import os
import sys
import csv
import glob
import numpy as np
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou

# =========================================================
# Clone ECP if needed
# =========================================================
if not os.path.exists("ECP"):
    os.system("git clone https://github.com/fouratifares/ECP.git")

sys.path.append(os.path.abspath("ECP"))
from optimizers.ECP import ECP

# =========================================================
# CONFIG
# =========================================================
DATA_ROOT = "dataset/usa/golden_data_tiny_train"
PRETRAINED_MODEL_PATH = "results/rslt_faster_rcnn_on_auto_labeled/best_fasterrcnn_1.pt"

IMG_SIZE = 500
BATCH_SIZE = 4
EPOCHS = 10  # small for ECP speed
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_DIR_TRAIN = f"{DATA_ROOT}/images/train"
LBL_DIR_TRAIN = f"{DATA_ROOT}/labels/train"
IMG_DIR_VAL   = f"{DATA_ROOT}/images/val"
LBL_DIR_VAL   = f"{DATA_ROOT}/labels/val"

IOU_THRESHOLD = 0.5

# =========================================================
# DATASET
# =========================================================
def load_yolo_txt(lbl_path, img_w, img_h):
    if not os.path.exists(lbl_path) or os.path.getsize(lbl_path) == 0:
        return torch.zeros((0,4)), torch.zeros((0,), dtype=torch.int64)

    boxes, labels = [], []

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

    return torch.tensor(boxes, dtype=torch.float32), \
           torch.tensor(labels, dtype=torch.int64)

class YoloDataset(Dataset):
    def __init__(self, img_dir, lbl_dir):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.*")))
        self.lbl_dir = lbl_dir

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img = torch.tensor(np.array(img)/255.).permute(2,0,1).float()

        lbl_path = os.path.join(self.lbl_dir, Path(img_path).stem + ".txt")
        boxes, labels = load_yolo_txt(lbl_path, IMG_SIZE, IMG_SIZE)

        return img, {"boxes": boxes, "labels": labels}

def collate_fn(batch):
    return tuple(zip(*batch))

# =========================================================
# Precision@0.5
# =========================================================
def compute_precision50(model, loader, score_thresh):

    TP, FP = 0, 0

    model.eval()

    with torch.no_grad():
        for images, targets in loader:

            images = [img.to(DEVICE) for img in images]
            outputs = model(images)

            for output, target in zip(outputs, targets):

                pred_boxes = output["boxes"].cpu()
                pred_scores = output["scores"].cpu()
                gt_boxes = target["boxes"]

                # confidence filtering
                keep = pred_scores >= score_thresh
                pred_boxes = pred_boxes[keep]
                pred_scores = pred_scores[keep]

                # sort by score
                sorted_idx = torch.argsort(pred_scores, descending=True)
                pred_boxes = pred_boxes[sorted_idx]
                pred_scores = pred_scores[sorted_idx]

                if len(gt_boxes) == 0:
                    FP += len(pred_boxes)
                    continue

                if len(pred_boxes) == 0:
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

    precision = TP / (TP + FP + 1e-6)
    return precision
    
def compute_f1_50(model, loader, score_thresh):

    TP, FP, FN = 0, 0, 0

    model.eval()

    with torch.no_grad():
        for images, targets in loader:

            images = [img.to(DEVICE) for img in images]
            outputs = model(images)

            for output, target in zip(outputs, targets):

                pred_boxes = output["boxes"].cpu()
                pred_scores = output["scores"].cpu()
                gt_boxes = target["boxes"]

                # confidence filtering
                keep = pred_scores >= score_thresh
                pred_boxes = pred_boxes[keep]
                pred_scores = pred_scores[keep]

                # sort by score
                sorted_idx = torch.argsort(pred_scores, descending=True)
                pred_boxes = pred_boxes[sorted_idx]

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

                # ground truths non matches = False Negatives
                FN += len(gt_boxes) - len(matched_gt)

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)

    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return f1

# =========================================================
# ECP Objective
# =========================================================
class RCNNObjective:

    def __init__(self):

        self.bounds = np.array([
            [1e-5, 5e-3],   # lr
            [0.85, 0.98],   # momentum
            [1e-6, 1e-3],   # weight_decay
            [0.4, 0.8],     # rpn_nms_thresh
            [0.2, 0.5],     # box_score_thresh
            [0.4, 0.7],     # box_nms_thresh
        ])

        self.dimensions = self.bounds.shape[0]
        self.log_path = "results/finetuning_rcnn_best_params_tiny_train/ecp_faster_rcnn_tiny_train_log.csv"

        os.makedirs("results/finetuning_rcnn_best_params_tiny_train/models_trials", exist_ok=True)

        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Precision50"] + [f"x{i}" for i in range(self.dimensions)])

        train_ds = YoloDataset(IMG_DIR_TRAIN, LBL_DIR_TRAIN)
        val_ds   = YoloDataset(IMG_DIR_VAL, LBL_DIR_VAL)

        self.train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                       shuffle=True, collate_fn=collate_fn)

        self.val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                                     shuffle=False, collate_fn=collate_fn)

        self.global_best_precision = 0.0
        self.global_best_model_path = "results/finetuning_rcnn_best_params_tiny_train/best_model_global.pt"

        self.PATIENCE = 3  # early stopping patience

    def __call__(self, x):

        lr, momentum, weight_decay, rpn_nms, score_thresh, box_nms = map(float, x)

        trial_name = (
            f"lr{lr:.1e}_mom{momentum:.2f}_wd{weight_decay:.1e}"
            f"_rpn{rpn_nms:.2f}_sc{score_thresh:.2f}_nms{box_nms:.2f}"
        )

        trial_model_path = f"results/finetuning_rcnn_best_params_tiny_train/models_trials/{trial_name}.pt"

        print(f"\n=== Trial: {trial_name} ===")

        model = fasterrcnn_resnet50_fpn(weights=None)

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

        model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=DEVICE))

        # inject hyperparameters
        model.rpn.nms_thresh = rpn_nms
        model.roi_heads.score_thresh = score_thresh
        model.roi_heads.nms_thresh = box_nms

        model.to(DEVICE)

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )

        best_precision = 0.0
        epochs_no_improve = 0

        # ---------------- TRAINING LOOP ----------------
        for epoch in range(EPOCHS):

            model.train()

            for images, targets in self.train_loader:
                images = [img.to(DEVICE) for img in images]
                targets = [{k: v.to(DEVICE) for k,v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                loss = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # --- validation ---
#            precision50 = compute_precision50(model, self.val_loader, score_thresh)
            F1_50=compute_f1_50(model, self.val_loader, score_thresh)

#            print(f"Epoch {epoch+1} - Precision50: {precision50:.4f}")
            print(f"Epoch {epoch+1} - F1_50: {F1_50:.4f}")

            # --- check improvement ---
            if F1_50 > best_precision + 1e-5:
                best_precision = F1_50
                epochs_no_improve = 0

                torch.save(model.state_dict(), trial_model_path)
                print("Saved best model for this trial.")
            else:
                epochs_no_improve += 1

            # --- early stopping ---
            if epochs_no_improve >= self.PATIENCE:
                print("Early stopping triggered.")
                break

        # ---------------- GLOBAL BEST ----------------
        if best_precision > self.global_best_precision:
            self.global_best_precision = best_precision
            torch.save(model.state_dict(), self.global_best_model_path)
            print("?? Updated GLOBAL BEST MODEL")

        # log
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([best_precision] + list(x))

        print(f"Best Precision for trial: {best_precision:.4f}")

        return best_precision

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    objective = RCNNObjective()

    n_evals = 30
    points, values, epsilons = ECP(objective, n=n_evals)

    best_index = np.argmax(values)
    best_point = points[best_index]
    best_precision = values[best_index]

    print("\nOptimization Complete")
    print(f"Best hyperparameters: {best_point}")
    print(f"Best f1_50: {best_precision:.4f}")