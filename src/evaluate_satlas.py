import os
import json
import numpy as np
from pathlib import Path

# =========================
# CONFIG
# =========================
PRED_JSON = "results/usa/satlas_tiny_train_ecp/best_test_preds.json"
LABEL_DIR = "dataset/usa/golden_data_tiny_train/labels/test"
OUTPUT_DIR = "results/usa/satlas_tiny_train_ecp/evaluation_satlas"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "evaluation_metrics_on_tiny_train_golden.json")

IOU_THRESHOLDS = np.arange(0.5, 1.0, 0.05)

MODEL_IMG_SIZE = 400
ORIGINAL_IMG_SIZE = 500
SCALE = MODEL_IMG_SIZE / ORIGINAL_IMG_SIZE

CONF_THRESHOLD_PRECISION = 0.25  # only for precision/recall

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# IoU
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


# =========================
# Load GT
# =========================

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


# =========================
# Load predictions
# =========================

with open(PRED_JSON) as f:
    predictions = json.load(f)


# =========================
# Precision / Recall / F1 @ IoU 0.5
# =========================

TP50, FP50, FN50 = 0, 0, 0

for item in predictions:

    img_name = Path(item["image_path"]).stem
    lbl_path = os.path.join(LABEL_DIR, img_name + ".txt")

    gt_boxes = load_yolo_gt(lbl_path)

    pred_boxes = item["boxes"]
    pred_scores = item["scores"]

    # Apply confidence threshold only here
    keep = [i for i, s in enumerate(pred_scores) if s >= CONF_THRESHOLD_PRECISION]
    pred_boxes = [pred_boxes[i] for i in keep]
    pred_scores = [pred_scores[i] for i in keep]

    # Sort by score descending
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


# =========================
# COCO-style mAP
# =========================

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

        # DO NOT apply confidence threshold here (COCO compliant)

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


# =========================
# Save
# =========================

results = {
    "Precision50": float(precision50),
    "Recall50": float(recall50),
    "F1Score50": float(f1_50),
    "mAP50": float(ap50),
    "mAP50:95": float(map5095)
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=4)

print("Evaluation completed. Results saved to:", OUTPUT_FILE)