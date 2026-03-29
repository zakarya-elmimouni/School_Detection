import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path
from collections import defaultdict

model_name = 'yolo26n'

#MODEL_PATH = "runs/detect/results/usa/rslt_yolo26n_tiny_train_on_golden_best_params/test3/weights/best.pt"
#DATA_YAML = "dataset/usa/golden_data_tiny_train/data.yaml"
#IMAGES_TEST_DIR = "dataset/usa/golden_data_tiny_train/images/test"
#LABELS_TEST_DIR = "dataset/usa/golden_data_tiny_train/labels/test"
#OUTPUT_METRICS_TXT = "results/rslt_yolo26n_tiny_train_on_golden_best_params/test3/metrics.txt"
#OUTPUT_IMG_DIR = "results/rslt_yolo26n_tiny_train_on_golden_best_params/test3/visualizations"


MODEL_PATH = "runs/detect/results/usa/rslt_yolo26n_on_small_train_golden_best_params/test3/weights/best.pt"
DATA_YAML = "dataset/usa/golden_data/data.yaml"
IMAGES_TEST_DIR = "dataset/usa/golden_data/images/test"
LABELS_TEST_DIR = "dataset/usa/golden_data/labels/test"
OUTPUT_METRICS_TXT = "results/rslt_yolo26n_on_small_train_golden_best_params/test3/metrics.txt"
OUTPUT_IMG_DIR = "results/rslt_yolo26n_on_small_train_golden_best_params/test3/visualizations"


NUM_IMAGES = 5
IMAGE_SIZE = 500

os.makedirs("results/rslt_yolo26n_on_small_train_golden_best_params/test3", exist_ok=True)
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)
metrics = model.val(split="test", data=DATA_YAML, max_det=1)

# === Save YOLO metrics ===
with open(OUTPUT_METRICS_TXT, "w") as f:
    f.write(f"=== YOLO  trained on {model_name} and Evaluated on golden Test Set ===\n")
    f.write(f"mAP@0.5        : {metrics.box.map50:.4f}\n")
    f.write(f"mAP@0.5:0.95   : {metrics.box.map:.4f}\n")
    f.write(f"Mean Precision : {metrics.box.mp:.4f}\n")
    f.write(f"Mean Recall    : {metrics.box.mr:.4f}\n\n")
    
    # Mean F1-score
    if (metrics.box.mp + metrics.box.mr) > 0:
        mean_f1 = 2 * metrics.box.mp * metrics.box.mr / (metrics.box.mp + metrics.box.mr)
    else:
        mean_f1 = 0.0
    f.write(f"Mean F1-score  : {mean_f1:.4f}\n\n")

    f.write("=== Per-class Metrics ===\n")
    for i, name in model.names.items():
        p, r, ap50, ap = metrics.box.class_result(i)
        if (p + r) > 0:
            f1 = 2 * p * r / (p + r)
        else:
            f1 = 0.0
        f.write(f"{name:10s}: Precision={p:.3f}, Recall={r:.3f}, F1-score={f1:.3f}, mAP50={ap50:.3f}, mAP50-95={ap:.3f}\n")


print(f"Metrics saved to: {OUTPUT_METRICS_TXT}")

# === Helper to load YOLO labels ===
def load_yolo_labels(label_path, img_shape):
    h, w = img_shape[:2]
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                _, x, y, bw, bh = map(float, parts)
                cx, cy = x * w, y * h
                bw, bh = bw * w, bh * h
                x1, y1 = int(cx - bw/2), int(cy - bh/2)
                x2, y2 = int(cx + bw/2), int(cy + bh/2)
                boxes.append((x1, y1, x2, y2))
    return boxes

# === Visual sample predictions ===
all_imgs = [p for p in Path(IMAGES_TEST_DIR).glob("*.png") if cv2.imread(str(p)) is not None]
sample_imgs = random.sample(all_imgs, min(NUM_IMAGES, len(all_imgs)))
for img_path in sample_imgs:
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_name = img_path.stem
    label_path = os.path.join(LABELS_TEST_DIR, img_name + ".txt")

    gt_boxes = load_yolo_labels(label_path, img.shape)
    results = model(img, max_det=1)
    pred_boxes = results[0].boxes.xyxy.cpu().numpy()

    for (x1, y1, x2, y2) in gt_boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for (x1, y1, x2, y2) in pred_boxes:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

    output_path = os.path.join(OUTPUT_IMG_DIR, f"{img_name}_pred.png")
    cv2.imwrite(output_path, img)

# === IOU and AP ===
def compute_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    if inter_area == 0:
        return 0.0
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter_area / float(box1_area + box2_area - inter_area)

def average_precision(recalls, precisions):
    precisions = np.maximum.accumulate(precisions[::-1])[::-1]
    return np.trapz(precisions, recalls)

def compute_map_iou(model, img_dir, label_dir, iou_thresh=0.5):
    all_detections = []
    all_annotations = defaultdict(list)
    img_paths = list(Path(img_dir).glob("*.png"))

    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_name = img_path.stem
        label_path = os.path.join(label_dir, img_name + ".txt")

        gt_boxes = load_yolo_labels(label_path, img.shape)
        results = model(img, max_det=1)
        pred_boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()

        all_annotations[img_name] = gt_boxes
        for i in range(len(pred_boxes)):
            all_detections.append((img_name, scores[i], pred_boxes[i]))

    all_detections.sort(key=lambda x: x[1], reverse=True)
    image_gt_flags = {k: np.zeros(len(v)) for k, v in all_annotations.items()}
    tp = np.zeros(len(all_detections))
    fp = np.zeros(len(all_detections))
    total_gts = sum(len(v) for v in all_annotations.values())

    for i, (img_name, conf, pred_box) in enumerate(all_detections):
        matched = False
        gt_boxes = all_annotations[img_name]
        for j, gt_box in enumerate(gt_boxes):
            iou = compute_iou(pred_box, gt_box)
            if iou >= iou_thresh and image_gt_flags[img_name][j] == 0:
                matched = True
                image_gt_flags[img_name][j] = 1
                break
        if matched:
            tp[i] = 1
        else:
            fp[i] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / total_gts
    precisions = tp_cum / (tp_cum + fp_cum + 1e-16)
    ap = average_precision(recalls, precisions)
    final_recall = recalls[-1] if len(recalls) else 0
    final_precision = precisions[-1] if len(precisions) else 0
    return ap, final_recall, final_precision

def compute_mse_pred_vs_gt(img_dir, label_dir, model):
    distances = []
    img_paths = list(Path(img_dir).glob("*.png"))

    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_name = img_path.stem
        label_path = os.path.join(label_dir, img_name + ".txt")
        gt_boxes = load_yolo_labels(label_path, img.shape)
        if not gt_boxes:
            continue

        results = model(img, max_det=1)
        pred_boxes = results[0].boxes.xyxy.cpu().numpy()
        if len(pred_boxes) == 0:
            continue

        # Take only the first predicted box and first gt box
        x1_p, y1_p, x2_p, y2_p = pred_boxes[0]
        x1_g, y1_g, x2_g, y2_g = gt_boxes[0]

        cx_pred = (x1_p + x2_p) / 2
        cy_pred = (y1_p + y2_p) / 2
        cx_gt = (x1_g + x2_g) / 2
        cy_gt = (y1_g + y2_g) / 2

        dist_sq = (cx_pred - cx_gt) ** 2 + (cy_pred - cy_gt) ** 2
        distances.append(dist_sq)

    if distances:
        return np.mean(distances)
    return None

map30, r30, p30 = compute_map_iou(model, IMAGES_TEST_DIR, LABELS_TEST_DIR, iou_thresh=0.3)
map75, r75, p75 = compute_map_iou(model, IMAGES_TEST_DIR, LABELS_TEST_DIR, iou_thresh=0.75)
iou_thresholds = np.arange(0.5, 1.0, 0.05)
all_ap = [compute_map_iou(model, IMAGES_TEST_DIR, LABELS_TEST_DIR, iou_thresh=iou)[0] for iou in iou_thresholds]
map_50_95 = np.mean(all_ap)
mse_center = compute_mse_pred_vs_gt(IMAGES_TEST_DIR, LABELS_TEST_DIR, model)

with open(OUTPUT_METRICS_TXT, "a") as f:
    f.write("\n=== Full mAP@0.3 Evaluation ===\n")
    f.write(f"mAP@0.3       : {map30:.4f}\n")
    f.write(f"Precision@0.3 : {p30:.4f}\n")
    f.write(f"Recall@0.3    : {r30:.4f}\n")

    f.write("\n=== Full mAP@0.75 Evaluation ===\n")
    f.write(f"mAP@0.75      : {map75:.4f}\n")
    f.write(f"Precision@0.75: {p75:.4f}\n")
    f.write(f"Recall@0.75   : {r75:.4f}\n")

    f.write("\n=== Custom mAP@[0.5:0.95] Evaluation ===\n")
    f.write(f"mAP@[.5:.95]  : {map_50_95:.4f}\n")

    f.write("\n=== Center Distance MSE (Prediction vs Ground Truth) ===\n")
    if mse_center is not None:
        f.write(f"MSE_center_distance: {mse_center:.2f}\n")
        f.write(f"RMSE_center_distance: {np.sqrt(mse_center):.2f}\n")
    else:
        f.write("MSE_center_distance: N/A\n")

print(f"\n? Custom Evaluation Complete. Results saved in {OUTPUT_METRICS_TXT}")


