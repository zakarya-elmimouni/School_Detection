import os
import glob
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
DATA_ROOT = "dataset/usa/golden_data_tiny_train"

IMG_DIR_TRAIN = f"{DATA_ROOT}/images/train"
LBL_DIR_TRAIN = f"{DATA_ROOT}/labels/train"
IMG_DIR_VAL   = f"{DATA_ROOT}/images/val"
LBL_DIR_VAL   = f"{DATA_ROOT}/labels/val"
IMG_DIR_TEST   = f"{DATA_ROOT}/images/test"
LBL_DIR_TEST   = f"{DATA_ROOT}/labels/test"



PATIENCE = 8
BEST_MODEL_PATH = "results/rslt_faster_rcnn_on_tiny_train_golden/best_fasterrcnn.pt"
LOSS_PLOT_PATH =  "results/rslt_faster_rcnn_on_tiny_train_golden/loss_curves.png"

os.makedirs("results/rslt_faster_rcnn_on_tiny_train_golden", exist_ok=True)

NUM_CLASSES = 1  # school
BATCH_SIZE = 4
EPOCHS = 30
LR = 0.0005
rpn_nms=0.5
score_thresh=0.3
box_nms=0.4
IMG_SIZE = 500  # tu peux mettre 400 si tu veux matcher Satlas
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# YOLO ? FasterRCNN
# =========================
def load_yolo_txt(lbl_path, img_w, img_h):
    boxes, labels = [], []

    if not os.path.exists(lbl_path) or os.path.getsize(lbl_path) == 0:
        return np.zeros((0,4)), np.zeros((0,))

    with open(lbl_path) as f:
        lines = f.readlines()

    for line in lines:
        cls, cx, cy, w, h = map(float, line.strip().split())

        x1 = (cx - w/2) * img_w
        y1 = (cy - h/2) * img_h
        x2 = (cx + w/2) * img_w
        y2 = (cy + h/2) * img_h

        boxes.append([x1, y1, x2, y2])
        labels.append(int(cls) + 1)  # 0 = background

    if len(boxes) == 0:
        return np.zeros((0,4)), np.zeros((0,))

    return np.array(boxes), np.array(labels)

# =========================
# Dataset
# =========================

def clean_dataset(img_dir, lbl_dir):
    print(f"\nChecking dataset integrity in {img_dir}...")
    img_paths = glob.glob(os.path.join(img_dir, "*.*"))

    removed = 0

    for img_path in img_paths:
        try:
            img = Image.open(img_path)
            img.verify()  # 
        except Exception:
            print(f"Removing corrupted image: {img_path}")
            
            # Supprimer image
            os.remove(img_path)

            # Supprimer label correspondant
            label_path = os.path.join(
                lbl_dir,
                Path(img_path).stem + ".txt"
            )

            if os.path.exists(label_path):
                os.remove(label_path)
                print(f"Removed corresponding label: {label_path}")

            removed += 1

    print(f"Done. Removed {removed} corrupted image-label pairs.")




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

        img = np.array(img) / 255.0
        img = torch.tensor(img).permute(2,0,1).float()

        H, W = IMG_SIZE, IMG_SIZE
        lbl_path = os.path.join(self.lbl_dir, Path(img_path).stem + ".txt")
        boxes, labels = load_yolo_txt(lbl_path, W, H)

        target = {}
        target["boxes"] = torch.tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.tensor(labels, dtype=torch.int64)
        target["image_id"] = torch.tensor([idx])

        return img, target

def collate_fn(batch):
    return tuple(zip(*batch))

# =========================
# Model
# =========================
def get_model():
    model = fasterrcnn_resnet50_fpn(pretrained=True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, NUM_CLASSES + 1
    )
    model.rpn.nms_thresh = rpn_nms
    model.roi_heads.score_thresh = score_thresh
    model.roi_heads.nms_thresh = box_nms
    return model

# =========================
# Train Loop
# =========================
def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0

    for images, targets in loader:
        images = list(img.to(DEVICE) for img in images)
        targets = [{k: v.to(DEVICE) for k,v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader):
    model.train()  # important pour avoir les losses
    total_loss = 0

    for images, targets in loader:
        images = list(img.to(DEVICE) for img in images)
        targets = [{k: v.to(DEVICE) for k,v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()

    return total_loss / len(loader)


def save_loss_curves(train_losses, val_losses, path):
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# =========================
# MAIN
# =========================
def main():
    
    clean_dataset(IMG_DIR_TRAIN, LBL_DIR_TRAIN)
    clean_dataset(IMG_DIR_VAL, LBL_DIR_VAL)
    clean_dataset(IMG_DIR_TEST, LBL_DIR_TEST)
  
    train_ds = YoloDataset(IMG_DIR_TRAIN, LBL_DIR_TRAIN)
    val_ds   = YoloDataset(IMG_DIR_VAL, LBL_DIR_VAL)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_fn)

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                            shuffle=False, collate_fn=collate_fn)

    model = get_model().to(DEVICE)
    

    optimizer = torch.optim.SGD(
        model.parameters(), lr=LR, momentum=0.9, weight_decay=0.0005
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    
    train_losses = []
    val_losses = []
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3
    )
    
    for epoch in range(EPOCHS):
    
        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_loss = evaluate(model, val_loader)
    
        scheduler.step(val_loss)
    
        train_losses.append(train_loss)
        val_losses.append(val_loss)
    
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"Train loss: {train_loss:.4f}")
        print(f"Val loss: {val_loss:.4f}")
    
        # ?? Check improvement
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            epochs_without_improvement = 0
    
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print("? Best model saved.")
    
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epochs.")
    
        # ?? Early stopping
        if epochs_without_improvement >= PATIENCE:
            print("?? Early stopping triggered.")
            break
    
    # Save curves at the end
    save_loss_curves(train_losses, val_losses, LOSS_PLOT_PATH)
    
    print("\nTraining finished.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved at: {BEST_MODEL_PATH}")
    print(f"Loss curves saved at: {LOSS_PLOT_PATH}")


if __name__ == "__main__":
    main()
