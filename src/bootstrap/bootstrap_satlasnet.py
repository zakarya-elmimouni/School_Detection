# -*- coding: utf-8 -*-
import os
import json
import random
import numpy as np
from pathlib import Path
import shutil
import matplotlib.pyplot as plt

# =========================
# CONFIGURATION (ADAPT AS NEEDED)
# =========================
# List of Satlas models to evaluate
MODEL_PATHS = {
    "with_ecp":   "satlas_smaller_train_ecp/best_test_preds.json", # path tp the json file (change as needed)
    "without_ecp": "rslt_satlas_on_smaller_train_golden/best_test_preds.json", # path to the json file (change as needed)
}

LABEL_DIR = "golden_data/labels/test"   # YOLO label directory (change as needed)
TEST_TXT = Path("golden_data/test.txt")  # file listing test images (change as needed)
BACKUP_TEST_TXT = TEST_TXT.parent / "test_backup.txt"

# Bootstrap parameters
N_ITER = 100
SAMPLE_SIZE = None          # None = use full original test size
RANDOM_SEED = 123

# Satlas evaluation parameters
MODEL_IMG_SIZE = 400
ORIGINAL_IMG_SIZE = 500
SCALE = MODEL_IMG_SIZE / ORIGINAL_IMG_SIZE
CONF_THRESHOLD_PRECISION = 0.25    # used only for precision/recall/F1
IOU_THRESHOLDS = np.arange(0.5, 1.0, 0.05)

# Output directory for bootstrap results
OUTPUT_BASE = Path("results/bootstrap_satlas_100_regime") # change as needed
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

# =========================
# UTILITY FUNCTIONS
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

# =========================
# SATLAS EVALUATION FUNCTIONS (adapted for a subset of images)
# =========================
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0

    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - inter

    return inter / union

def load_yolo_gt(lbl_path):
    boxes = []
    if not os.path.exists(lbl_path) or os.path.getsize(lbl_path) == 0:
        return boxes

    with open(lbl_path, "r") as f:
        lines = f.readlines()

    for ln in lines:
        parts = ln.strip().split()
        if len(parts) != 5:
            continue

        _, cx, cy, w, h = map(float, parts)

        x1 = (cx - w/2) * ORIGINAL_IMG_SIZE
        y1 = (cy - h/2) * ORIGINAL_IMG_SIZE
        x2 = (cx + w/2) * ORIGINAL_IMG_SIZE
        y2 = (cy + h/2) * ORIGINAL_IMG_SIZE

        x1 *= SCALE
        y1 *= SCALE
        x2 *= SCALE
        y2 *= SCALE

        boxes.append([x1, y1, x2, y2])

    return boxes

def evaluate_satlas_on_subset(pred_json_path, image_list):
    """
    Compute metrics (P, R, F1@0.5, mAP@0.5, mAP@0.5:0.95)
    using only images whose stem is in `image_list`.
    `image_list`: list of file names (without extension) to include.
    """
    with open(pred_json_path, 'r') as f:
        all_predictions = json.load(f)

    # Filter predictions to keep only images in the sample
    sample_set = set(image_list)
    predictions = [item for item in all_predictions
                   if Path(item["image_path"]).stem in sample_set]

    # ---------- 1) Precision / Recall / F1 @ IoU 0.5 ----------
    TP50, FP50, FN50 = 0, 0, 0

    for item in predictions:
        img_name = Path(item["image_path"]).stem
        lbl_path = os.path.join(LABEL_DIR, img_name + ".txt")
        gt_boxes = load_yolo_gt(lbl_path)

        pred_boxes = item["boxes"]
        pred_scores = item["scores"]

        # Apply confidence threshold
        keep = [i for i, s in enumerate(pred_scores) if s >= CONF_THRESHOLD_PRECISION]
        pred_boxes = [pred_boxes[i] for i in keep]
        pred_scores = [pred_scores[i] for i in keep]

        # Sort by descending score
        if len(pred_scores) > 0:
            order = np.argsort(-np.array(pred_scores))
            pred_boxes = [pred_boxes[i] for i in order]

        matched_gt = set()

        for pred_box in pred_boxes:
            matched = False
            for i, gt_box in enumerate(gt_boxes):
                if i in matched_gt:
                    continue
                if compute_iou(pred_box, gt_box) >= 0.5:
                    matched = True
                    matched_gt.add(i)
                    break
            if matched:
                TP50 += 1
            else:
                FP50 += 1

        FN50 += len(gt_boxes) - len(matched_gt)

    precision50 = TP50 / (TP50 + FP50 + 1e-6)
    recall50 = TP50 / (TP50 + FN50 + 1e-6)
    f1_50 = 2 * precision50 * recall50 / (precision50 + recall50 + 1e-6)

    # ---------- 2) COCO-style mAP (no confidence filtering) ----------
    def compute_ap(iou_thresh):
        tp, fp, scores = [], [], []
        total_gt = 0

        for item in predictions:
            img_name = Path(item["image_path"]).stem
            lbl_path = os.path.join(LABEL_DIR, img_name + ".txt")
            gt_boxes = load_yolo_gt(lbl_path)
            total_gt += len(gt_boxes)

            pred_boxes = item["boxes"]
            pred_scores = item["scores"]

            # Do not filter by confidence (COCO standard)
            order = np.argsort(-np.array(pred_scores))
            pred_boxes = [pred_boxes[i] for i in order]
            pred_scores = [pred_scores[i] for i in order]

            matched_gt = set()
            for pred_box, score in zip(pred_boxes, pred_scores):
                matched = False
                for i, gt_box in enumerate(gt_boxes):
                    if i in matched_gt:
                        continue
                    if compute_iou(pred_box, gt_box) >= iou_thresh:
                        matched = True
                        matched_gt.add(i)
                        break
                tp.append(1 if matched else 0)
                fp.append(0 if matched else 1)
                scores.append(score)

        if total_gt == 0:
            return 0.0

        tp = np.array(tp)
        fp = np.array(fp)
        scores = np.array(scores)

        order = np.argsort(-scores)
        tp = tp[order]
        fp = fp[order]

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)

        recalls = tp_cum / total_gt
        precisions = tp_cum / (tp_cum + fp_cum + 1e-6)

        ap = 0.0
        for t in np.linspace(0, 1, 101):
            p = precisions[recalls >= t]
            ap += max(p) if len(p) else 0
        return ap / 101

    ap50 = compute_ap(0.5)
    map5095 = np.mean([compute_ap(t) for t in IOU_THRESHOLDS])

    return {
        'precision': precision50,
        'recall': recall50,
        'f1': f1_50,
        'map50': ap50,
        'map50_95': map5095
    }

# =========================
# MAIN BOOTSTRAP ROUTINE
# =========================
def main():
    # Backup original test.txt
    TEST_TXT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(TEST_TXT, BACKUP_TEST_TXT)

    try:
        # Read original test image list
        with open(TEST_TXT, "r") as f:
            original_test_paths = f.read().splitlines()

        # Extract base names (without extension) for each path
        original_image_names = [Path(p).stem for p in original_test_paths]

        sample_size = SAMPLE_SIZE if SAMPLE_SIZE is not None else len(original_image_names)
        if sample_size <= 0:
            raise RuntimeError("test.txt is empty or SAMPLE_SIZE <= 0.")

        rng = random.Random(RANDOM_SEED)

        # Generate all bootstrap samples (list of lists of image names)
        bootstrap_samples = [
            rng.choices(original_image_names, k=sample_size)
            for _ in range(N_ITER)
        ]

        # Dictionary to store metrics for each model
        results = {}
        for model_name in MODEL_PATHS:
            results[model_name] = {metric: [] for metric in
                                   ['precision', 'recall', 'f1', 'map50', 'map50_95']}

        # Loop over models
        for model_name, pred_json in MODEL_PATHS.items():
            print(f"\n===== Bootstrap for {model_name} ({pred_json}) =====")

            # Output directory for this model
            model_out_dir = OUTPUT_BASE / model_name
            model_out_dir.mkdir(parents=True, exist_ok=True)

            # Text file for results
            out_txt = model_out_dir / f"bootstrap_metrics_{N_ITER}.txt"
            plot_dir = model_out_dir / f"bootstrap_plots_{N_ITER}"
            plot_dir.mkdir(parents=True, exist_ok=True)

            # Bootstrap iterations
            for i, sample_names in enumerate(bootstrap_samples):
                print(f"  Iteration {i+1}/{N_ITER}")
                # Evaluate model on this sample
                metrics = evaluate_satlas_on_subset(pred_json, sample_names)
                for key in results[model_name]:
                    results[model_name][key].append(metrics[key])

            # Save confidence intervals
            with open(out_txt, "w") as f:
                f.write(f"=== Bootstrap Evaluation for Satlas model '{model_name}' ===\n")
                f.write(f"Predictions file: {pred_json}\n")
                f.write(f"Iterations: {N_ITER}, Sample size per iteration: {sample_size}\n\n")
                for metric in results[model_name]:
                    lo, mean, hi = compute_ci(results[model_name][metric])
                    f.write(f"{metric.upper()} : {mean:.4f} (95% CI: [{lo:.4f}, {hi:.4f}])\n")

            print(f"  Results saved to {out_txt}")

            # Histograms
            for metric, values in results[model_name].items():
                write_hist(
                    values,
                    title=f"Bootstrap Distribution of {metric.upper()} ({model_name})",
                    xlabel=metric.upper(),
                    out_png=plot_dir / f"{metric}_hist.png"
                )
            print(f"  Plots saved to {plot_dir}")

        print("\n Bootstrap completed no model comparison performed.")

    finally:
        # Restore original test.txt
        if BACKUP_TEST_TXT.exists():
            shutil.copy(BACKUP_TEST_TXT, TEST_TXT)
            print("\nRestored original test.txt.")

if __name__ == "__main__":
    main()