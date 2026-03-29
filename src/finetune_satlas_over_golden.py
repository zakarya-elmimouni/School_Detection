# -*- coding: utf-8 -*-
# finetune_satlas_on_golden.py

import os
import glob
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

import torch
from torch.utils.data import Dataset, DataLoader

import satlaspretrain_models as spm

# =========================
# CONFIG
# =========================

# ---- Golden dataset ----
DATA_ROOT = "dataset/usa/golden_data_small_train"
IMG_DIR_TRAIN = f"{DATA_ROOT}/images/train"
LBL_DIR_TRAIN = f"{DATA_ROOT}/labels/train"
IMG_DIR_VAL = f"{DATA_ROOT}/images/val"
LBL_DIR_VAL = f"{DATA_ROOT}/labels/val"
IMG_DIR_TEST = f"{DATA_ROOT}/images/test"
LBL_DIR_TEST = f"{DATA_ROOT}/labels/test"

# ---- Pretrained weights from auto-labeled training ----
PRETRAINED_WEIGHTS = "results/usa/rslt_satlas_auto_labeled/best.pt"

MODEL_ID = "Aerial_SwinB_SI"
NUM_CLASSES = 1

# ?? Lower LR for finetuning
LR = 5e-5
EPOCHS = 30
BATCH_SIZE = 4
WEIGHT_DECAY = 1e-4
PATIENCE = 8
IMG_SIZE = 400

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DIR = "results/usa/finetune_satlas_pret_auto_on_small_train_golden"
os.makedirs(OUT_DIR, exist_ok=True)
BEST_WEIGHTS = os.path.join(OUT_DIR, "best_finetuned.pt")

ImageFile.LOAD_TRUNCATED_IMAGES = True

# =========================
# Utils
# =========================

def load_yolo_txt(lbl_path, img_w, img_h):
    boxes, labels = [], []

    if not os.path.exists(lbl_path):
        return np.zeros((0,4)), np.zeros((0,))

    with open(lbl_path) as f:
        for ln in f.readlines():
            cls, cx, cy, w, h = map(float, ln.split())
            x1 = (cx - w/2) * img_w
            y1 = (cy - h/2) * img_h
            x2 = (cx + w/2) * img_w
            y2 = (cy + h/2) * img_h
            boxes.append([x1, y1, x2, y2])
            labels.append(int(cls) + 1)

    if len(boxes) == 0:
        return np.zeros((0,4)), np.zeros((0,))

    return np.array(boxes), np.array(labels)


class YoloDataset(Dataset):
    def __init__(self, img_dir, lbl_dir):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.*")))
        self.lbl_dir = lbl_dir

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        img = Image.open(path).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img = np.asarray(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2,0,1)

        H, W = IMG_SIZE, IMG_SIZE
        lbl_path = os.path.join(self.lbl_dir, Path(path).stem + ".txt")
        boxes, labels = load_yolo_txt(lbl_path, W, H)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }

        return img, target


def collate_fn(batch):
    imgs, targets = list(zip(*batch))
    return torch.stack(imgs,0), list(targets)


# =========================
# Build Model
# =========================

def build_model():
    weights = spm.Weights()
    model = weights.get_pretrained_model(
        MODEL_ID,
        fpn=True,
        head=spm.Head.DETECT,
        num_categories=NUM_CLASSES + 1,
        device=DEVICE
    )
    return model

def total_loss_from_model_output(out):
    """
    Improved version with optional debug
    """
    if isinstance(out, dict):
        print("c'est ca ce qu'il faut regarder",out)
        total_loss = 0.0
        loss_count = 0
        for key, value in out.items():
            if 'loss' in key.lower() and torch.is_tensor(value):
                total_loss += value
                loss_count += 1
                # print(f"Loss {key}: {value.item():.4f}")  # Uncomment for debug
        if loss_count == 0:
            for key, value in out.items():
                if torch.is_tensor(value):
                    total_loss += value
                    loss_count += 1
        return total_loss
    elif isinstance(out, (list, tuple)):
        if len(out) >= 1 and isinstance(out[0], dict):
            return total_loss_from_model_output(out[0])
        else:
            total_loss = 0.0
            for item in out:
                if torch.is_tensor(item):
                    total_loss += item
            return total_loss
    elif torch.is_tensor(out):
        return out
    else:
        print(f"WARNING: Unrecognized output format: {type(out)}")
        return torch.tensor(0.0, device=DEVICE)


# =========================
# MAIN
# =========================

def main():

    print("Loading Golden dataset...")
    train_ds = YoloDataset(IMG_DIR_TRAIN, LBL_DIR_TRAIN)
    val_ds   = YoloDataset(IMG_DIR_VAL, LBL_DIR_VAL)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_fn)

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                            shuffle=False, collate_fn=collate_fn)

    print("Loading model...")
    model = build_model().to(DEVICE)

    # ?? Load previously trained weights
    print("Loading pretrained weights from auto-labeled training...")
    model.load_state_dict(torch.load(PRETRAINED_WEIGHTS, map_location=DEVICE))

    # OPTIONAL: Freeze backbone (uncomment if needed)
    """
    for name, param in model.backbone.named_parameters():
        param.requires_grad = False
    """

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)

    best_val = float("inf")
    patience_counter = 0

    for epoch in range(EPOCHS):

        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        model.train()
        train_loss = 0

        for imgs, targets in train_loader:
            imgs = imgs.to(DEVICE)
            targets = [{k:v.to(DEVICE) for k,v in t.items()} for t in targets]

            out = model(imgs, targets)
#            loss = sum(loss_dict.values())
            loss = total_loss_from_model_output(out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ---- Validation ----
        model.train()  # important for detection loss
        val_loss = 0

        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(DEVICE)
                targets = [{k:v.to(DEVICE) for k,v in t.items()} for t in targets]

                out = model(imgs, targets)
#                loss = sum(loss_dict.values())
                loss = total_loss_from_model_output(out)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # ---- Early stopping ----
        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), BEST_WEIGHTS)
            print("? New best model saved")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping")
                break

    print("\nFinetuning completed.")
    print(f"Best model saved at: {BEST_WEIGHTS}")


if __name__ == "__main__":
    main()