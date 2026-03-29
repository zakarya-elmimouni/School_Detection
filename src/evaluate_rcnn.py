import os
import glob
from pathlib import Path
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou

# =========================
# CONFIG
# =========================
DATA_ROOT = "dataset/usa/golden_data_tiny_train"
IMG_DIR_TEST = f"{DATA_ROOT}/images/test"
LBL_DIR_TEST = f"{DATA_ROOT}/labels/test"

MODEL_PATH = "results/finetuning_rcnn_best_params_tiny_train/best_model_global.pt"
OUTPUT_TXT = "results/finetuning_rcnn_best_params_tiny_train/best_evaluation.txt"

NUM_CLASSES = 1
IMG_SIZE = 500

CONF_THRESHOLD = 0.1
IOU_THRESHOLD = 0.5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# YOLO ? Target
# =========================
def load_yolo_txt(lbl_path, img_w, img_h):

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
        labels.append(int(cls) + 1)

    if len(boxes) == 0:
        return torch.zeros((0,4)), torch.zeros((0,), dtype=torch.int64)

    return torch.tensor(boxes, dtype=torch.float32), \
           torch.tensor(labels, dtype=torch.int64)

# =========================
# Dataset
# =========================
class TestDataset(Dataset):
    def __init__(self, img_dir, lbl_dir):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.*")))
        self.lbl_dir = lbl_dir

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):

        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))

        img = np.array(img) / 255.0
        img = torch.tensor(img).permute(2,0,1).float()

        lbl_path = os.path.join(self.lbl_dir,
                                Path(img_path).stem + ".txt")

        boxes, labels = load_yolo_txt(lbl_path, IMG_SIZE, IMG_SIZE)

        target = {
            "boxes": boxes,
            "labels": labels
        }

        return img, target

def collate_fn(batch):
    return tuple(zip(*batch))

# =========================
# Load model
# =========================
def load_model():

    model = fasterrcnn_resnet50_fpn(weights=None)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, NUM_CLASSES + 1
    )

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    return model

# =========================
# Precision / Recall / F1 manual
# =========================
def compute_pr_f1(model, loader):

    TP = 0
    FP = 0
    FN = 0

    with torch.no_grad():
        for images, targets in loader:

            images = [img.to(DEVICE) for img in images]
            outputs = model(images)

            for output, target in zip(outputs, targets):
                
                pred_boxes = output["boxes"].cpu()
                pred_scores = output["scores"].cpu()
                gt_boxes = target["boxes"]
                
                # confidence filtering
                keep = pred_scores >= CONF_THRESHOLD
                pred_boxes = pred_boxes[keep]
                pred_scores = pred_scores[keep]
                
                # ?? IMPORTANT: sort predictions by score (descending)
                sorted_indices = torch.argsort(pred_scores, descending=True)
                pred_boxes = pred_boxes[sorted_indices]
                pred_scores = pred_scores[sorted_indices]

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

                    if max_iou >= IOU_THRESHOLD and idx.item() not in matched_gt:
                        TP += 1
                        matched_gt.add(idx.item())
                    else:
                        FP += 1

                FN += len(gt_boxes) - len(matched_gt)

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return precision, recall, f1

# =========================
# MAIN
# =========================
def main():

    dataset = TestDataset(IMG_DIR_TEST, LBL_DIR_TEST)
    loader = DataLoader(dataset, batch_size=4,
                        shuffle=False, collate_fn=collate_fn)

    model = load_model()

    # mAP
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

    results = metric.compute()

    # Precision / Recall / F1
    precision, recall, f1 = compute_pr_f1(model, loader)

    # Save to file
    with open(OUTPUT_TXT, "w") as f:

        f.write("===== TEST METRICS =====\n\n")

        f.write(f"mAP@0.5: {results['map_50'].item():.6f}\n")
        f.write(f"mAP@[0.5:0.95]: {results['map'].item():.6f}\n\n")

        f.write(f"Precision (IoU={IOU_THRESHOLD}, conf={CONF_THRESHOLD}): {precision:.6f}\n")
        f.write(f"Recall (IoU={IOU_THRESHOLD}, conf={CONF_THRESHOLD}): {recall:.6f}\n")
        f.write(f"F1-score: {f1:.6f}\n")

    print(f"\nMetrics saved in {OUTPUT_TXT}")

if __name__ == "__main__":
    main()
