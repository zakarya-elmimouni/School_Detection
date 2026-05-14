# -*- coding: utf-8 -*-
import os
import random
import glob
from pathlib import Path
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou

# =========================
# CONFIGURATION (TO BE ADAPTED)
# =========================
DATA_YAML = "golden_data/data.yaml"  # change as needed
TEST_TXT = Path("golden_data/test.txt") # change as needed
BACKUP_TEST_TXT = TEST_TXT.parent / "test_backup.txt"

# Paths to Faster R-CNN models (to be adapted)
MODEL_PATHS = {
    "finetuned":   "best_model_global.pt", # change as needed
    "manual_only": "best_model_global.pt", # change as needed
#    "auto_only":   "Desktop/usa_school_mapping/results/rslt_faster_rcnn_on_auto_labeled/best_fasterrcnn_1.pt",  # change if needed
}

# Bootstrap parameters
N_ITER = 100
SAMPLE_SIZE = None          # None = use full original test size
RANDOM_SEED = 123

# Faster R-CNN evaluation parameters
NUM_CLASSES = 1
IMG_SIZE = 500
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# UTILITIES (unchanged)
# =========================
def compute_ci(values, alpha=0.95):
    values = np.asarray(values, dtype=float)
    lo = np.percentile(values, (1 - alpha) / 2 * 100)
    hi = np.percentile(values, (1 + alpha) / 2 * 100)
    return float(lo), float(values.mean()), float(hi)

def write_hist(values, title, xlabel, out_png):
    plt.figure()
    plt.hist(values, bins=20, color='skyblue', edgecolor='black')
    plt.axvline(np.mean(values), color='red', linestyle='--', label='Mean')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def pvalues_for_deltas(deltas):
    d = np.asarray(deltas, dtype=float)
    n = len(d)
    if n == 0:
        return float("nan"), float("nan")
    p_leq0 = np.mean(d <= 0.0)
    p_geq0 = np.mean(d >= 0.0)
    p_two = 2.0 * min(p_leq0, p_geq0)
    p_two = min(max(p_two, 0.0), 1.0)
    p_one_greater = p_leq0
    return float(p_two), float(p_one_greater)

# =========================
# DATA LOADING (compatible with test.txt)
# =========================
def load_yolo_txt(lbl_path, img_w, img_h):
    """Load a YOLO label file and return boxes (x1,y1,x2,y2) and labels (1-indexed)."""
    if not os.path.exists(lbl_path) or os.path.getsize(lbl_path) == 0:
        return torch.zeros((0,4)), torch.zeros((0,), dtype=torch.int64)
    boxes = []
    labels = []
    with open(lbl_path) as f:
        lines = f.readlines()
    for line in lines:
        cls, cx, cy, w, h = map(float, line.strip().split())
        x1 = (cx - w/2) * img_w
        y1 = (cy - h/2) * img_h
        x2 = (cx + w/2) * img_w
        y2 = (cy + h/2) * img_h
        boxes.append([x1, y1, x2, y2])
        labels.append(int(cls) + 1)   # Faster R-CNN expects 1 for the object
    if len(boxes) == 0:
        return torch.zeros((0,4)), torch.zeros((0,), dtype=torch.int64)
    return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)

class TestDataset(Dataset):
    """Dataset that reads image paths from a test.txt file (or a list)."""
    def __init__(self, img_dir, lbl_dir, file_list=None):
        """
        file_list : path to a text file containing full image paths (one per line)
                    or a direct list of paths. If None, uses all .png files in img_dir.
        """
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        if file_list is not None:
            if isinstance(file_list, (str, Path)):
                with open(file_list, "r") as f:
                    self.img_paths = [line.strip() for line in f if line.strip()]
            else:
                self.img_paths = list(file_list)
        else:
            # fallback: all .png files in the directory
            self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img = np.array(img) / 255.0
        img = torch.tensor(img).permute(2,0,1).float()

        lbl_path = os.path.join(self.lbl_dir, Path(img_path).stem + ".txt")
        boxes, labels = load_yolo_txt(lbl_path, IMG_SIZE, IMG_SIZE)
        target = {"boxes": boxes, "labels": labels}
        return img, target

def collate_fn(batch):
    return tuple(zip(*batch))

# =========================
# LOADING A FASTER R-CNN MODEL
# =========================
def load_faster_rcnn_model(ckpt_path):
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES + 1)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# =========================
# FULL EVALUATION (mAP + PRF1)
# =========================
def evaluate_faster_rcnn(model, test_txt_path):
    """
    Evaluate metrics on the set of images listed in test_txt_path.
    Returns a dict with: map50, map50_95, precision, recall, f1.
    """
    # Hard-coded directories (adapt to your structure)
    IMG_DIR_TEST = "golden_data/images/test" # change as needed
    LBL_DIR_TEST = "golden_data/labels/test" # change as needed
    
    dataset = TestDataset(IMG_DIR_TEST, LBL_DIR_TEST, file_list=test_txt_path)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # ---- mAP using torchmetrics ----
    metric = MeanAveragePrecision(iou_type="bbox")
    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(DEVICE) for img in images]
            outputs = model(images)
            preds = []
            gts = []
            for output, target in zip(outputs, targets):
                preds.append({
                    "boxes": output["boxes"].cpu(),
                    "scores": output["scores"].cpu(),
                    "labels": output["labels"].cpu()
                })
                gts.append({
                    "boxes": target["boxes"],
                    "labels": target["labels"]
                })
            metric.update(preds, gts)
    map_metrics = metric.compute()
    map50 = map_metrics['map_50'].item()
    map50_95 = map_metrics['map'].item()

    # ---- Precision, recall, F1 (manual) ----
    def compute_pr_f1(model, loader, conf_thresh=CONF_THRESHOLD, iou_thresh=IOU_THRESHOLD):
        TP, FP, FN = 0, 0, 0
        with torch.no_grad():
            for images, targets in loader:
                images = [img.to(DEVICE) for img in images]
                outputs = model(images)
                for output, target in zip(outputs, targets):
                    pred_boxes = output["boxes"].cpu()
                    pred_scores = output["scores"].cpu()
                    gt_boxes = target["boxes"]
                    keep = pred_scores >= conf_thresh
                    pred_boxes = pred_boxes[keep]
                    pred_scores = pred_scores[keep]
                    # sort predictions by descending score
                    sorted_idx = torch.argsort(pred_scores, descending=True)
                    pred_boxes = pred_boxes[sorted_idx]
                    pred_scores = pred_scores[sorted_idx]

                    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
                        continue
                    if len(gt_boxes) == 0:
                        FP += len(pred_boxes)
                        continue
                    if len(pred_boxes) == 0:
                        FN += len(gt_boxes)
                        continue

                    ious = box_iou(pred_boxes, gt_boxes)
                    matched_gt = set()
                    for i in range(len(pred_boxes)):
                        max_iou, idx = torch.max(ious[i], dim=0)
                        if max_iou >= iou_thresh and idx.item() not in matched_gt:
                            TP += 1
                            matched_gt.add(idx.item())
                        else:
                            FP += 1
                    FN += len(gt_boxes) - len(matched_gt)
        precision = TP / (TP + FP + 1e-6)
        recall = TP / (TP + FN + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        return precision, recall, f1

    precision, recall, f1 = compute_pr_f1(model, loader)

    return {
        'map50': map50,
        'map50_95': map50_95,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# =========================
# MAIN BOOTSTRAP ROUTINE
# =========================
def main():
    # Backup original test.txt
    TEST_TXT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(TEST_TXT, BACKUP_TEST_TXT)
    current_sample_size = SAMPLE_SIZE
    try:
        # Read original test image list
        with open(TEST_TXT, "r") as f:
            original_test_paths = f.read().splitlines()
        if current_sample_size is None:
            current_sample_size = len(original_test_paths)
        if current_sample_size <= 0:
            raise RuntimeError("test.txt is empty or SAMPLE_SIZE <= 0.")

        # Pre-generate bootstrap samples (paired across models)
        rng = random.Random(RANDOM_SEED)
        bootstrap_samples = [rng.choices(original_test_paths, k=current_sample_size) for _ in range(N_ITER)]

        # Containers for metrics of each model
        results = {name: {m: [] for m in ['map50', 'map50_95', 'precision', 'recall', 'f1']}
                   for name in MODEL_PATHS}

        # Evaluate each model
        for model_name, ckpt_path in MODEL_PATHS.items():
            print(f"\n===== Bootstrap for {model_name} ({ckpt_path}) =====")
            model = load_faster_rcnn_model(ckpt_path)

            # Output directory for this model
            base_out = Path(ckpt_path).parent
            output_txt = base_out / f"bootstrap_metrics_{N_ITER}_conf25.txt"
            plot_dir = base_out / f"bootstrap_plots_{N_ITER}_conf25"
            plot_dir.mkdir(parents=True, exist_ok=True)

            # Bootstrap iterations
            for i, sample_paths in enumerate(bootstrap_samples):
                print(f"  Iteration {i+1}/{N_ITER}")
                # Write current sample to test.txt
                with open(TEST_TXT, "w") as f:
                    f.write('\n'.join(sample_paths))
                # Evaluate model on this sample
                metrics = evaluate_faster_rcnn(model, TEST_TXT)
                for key in results[model_name]:
                    results[model_name][key].append(metrics[key])

            # Save results for this model
            with open(output_txt, "w") as f:
                f.write(f"=== Bootstrap Evaluation Results (Faster R-CNN) ===\n")
                f.write(f"Model checkpoint: {ckpt_path}\n")
                f.write(f"Iterations: {N_ITER}, Sample size per iteration: {current_sample_size}\n\n")
                for metric in results[model_name]:
                    lo, mean, hi = compute_ci(results[model_name][metric])
                    f.write(f"{metric.upper()} : {mean:.4f} (95% CI: [{lo:.4f}, {hi:.4f}])\n")

            print(f"  Metrics saved to {output_txt}")

            # Histograms
            for metric, values in results[model_name].items():
                write_hist(values,
                           title=f"Bootstrap Distribution of {metric.upper()} ({model_name})",
                           xlabel=metric.upper(),
                           out_png=plot_dir / f"{metric}_hist.png")
            print(f"  Plots saved to {plot_dir}")

        # ===== PAIRED COMPARISON finetuned - manual_only =====
        compare_dir = Path("results/bootstrap_rcnn_compare_50_regime_conf25") / f"paired_{N_ITER}" # change as needed
        compare_dir.mkdir(parents=True, exist_ok=True)
        delta_txt = compare_dir / f"delta_finetuned_minus_manual_only_{N_ITER}.txt"

        metrics_list = ['map50', 'map50_95', 'precision', 'recall', 'f1']
        with open(delta_txt, "w") as f:
            f.write("=== Paired Bootstrap on Deltas (finetuned - manual_only) ===\n")
            f.write(f"Iterations: {N_ITER}, Sample size per iteration: {current_sample_size}\n")
            f.write(f"Seed: {RANDOM_SEED}\n\n")
            for metric in metrics_list:
                a = np.asarray(results['finetuned'][metric])
                b = np.asarray(results['manual_only'][metric])
                d = a - b
                lo, mean, hi = compute_ci(d)
                p_two, p_one_gt = pvalues_for_deltas(d)
                f.write(f"{metric.upper()}:\n")
                f.write(f"  mean ?: {mean:.6f}\n")
                f.write(f"  95% CI: [{lo:.6f}, {hi:.6f}]\n")
                f.write(f"  p(two-sided): {p_two:.6f}\n")
                f.write(f"  p(one-sided H1: ?>0): {p_one_gt:.6f}\n\n")

        print(f"\n? Delta stats saved to {delta_txt}")

    finally:
        # Restore original test.txt
        if BACKUP_TEST_TXT.exists():
            shutil.copy(BACKUP_TEST_TXT, TEST_TXT)
            print("\nRestored original test.txt.")

if __name__ == "__main__":
    main()