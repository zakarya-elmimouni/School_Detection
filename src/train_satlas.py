# -*- coding: latin-1 -*-
# train_satlas_detect_full.py - PREDICTION FIX with Image Validation

import os
import glob
import json
import math
import time
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

import satlaspretrain_models as spm

# =========================
# Config - MODIFIED
# =========================
DATA_ROOT = "dataset/usa/golden_data_tiny_train"
IMG_DIR_TRAIN = f"{DATA_ROOT}/images/train"
LBL_DIR_TRAIN = f"{DATA_ROOT}/labels/train"
IMG_DIR_VAL = f"{DATA_ROOT}/images/val"
LBL_DIR_VAL = f"{DATA_ROOT}/labels/val"
IMG_DIR_TEST = f"{DATA_ROOT}/images/test"
LBL_DIR_TEST = f"{DATA_ROOT}/labels/test"

MODEL_ID = "Aerial_SwinB_SI"
NUM_CLASSES = 1

EPOCHS = 30  # Increased epochs
BATCH_SIZE = 8
LR = 2e-4  # Increased learning rate
WARMUP_EPOCHS = 3
PATIENCE = 8
WEIGHT_DECAY = 1e-4
IMG_SIZE = 400

# Confidence threshold for predictions - ADDED
CONFIDENCE_THRESHOLD = 0.01  # Lower threshold to see more predictions

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DIR = "results/usa/rslt_satlas_on_tiny_train_golden"
os.makedirs(OUT_DIR, exist_ok=True)
BEST_WEIGHTS = os.path.join(OUT_DIR, "best1.pt")
VAL_PREDS_JSON = os.path.join(OUT_DIR, "best_val_preds1.json")
TEST_PREDS_JSON = os.path.join(OUT_DIR, "best_test_preds1.json")
LOSS_PNG = os.path.join(OUT_DIR, "loss_curves1.png")

# =========================
# Image Validation & Recovery - ADDED
# =========================
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Allow loading truncated images

def safe_image_open(img_path):
    """
    Safely open an image file, handling truncated/corrupted files
    Returns a valid PIL Image object even for corrupted files
    """
    try:
        # First, check if file exists and has content
        if not os.path.exists(img_path):
            print(f"Warning: Image file does not exist: {img_path}")
            return Image.new('RGB', (IMG_SIZE, IMG_SIZE), color='white')
        
        if os.path.getsize(img_path) == 0:
            print(f"Warning: Image file is empty: {img_path}")
            return Image.new('RGB', (IMG_SIZE, IMG_SIZE), color='white')
        
        # Try to open and verify the image
        img = Image.open(img_path)
        
        # Try to load the image data to catch truncation errors
        try:
            img.load()
        except (OSError, IOError) as e:
            print(f"Warning: Truncated image {img_path}: {e}")
            # Try to recover by converting to RGB and copying
            img = img.convert("RGB")
        
        return img.convert("RGB")
        
    except Exception as e:
        print(f"Warning: Failed to open image {img_path}: {e}")
        # Create a blank image as fallback
        return Image.new('RGB', (IMG_SIZE, IMG_SIZE), color='white')

def validate_dataset_files(img_dir, lbl_dir):
    """
    Validate all images in dataset and report issues
    """
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.*")))
    valid_count = 0
    corrupted_count = 0
    
    print(f"Validating dataset in {img_dir}...")
    
    for img_path in img_paths:
        try:
            # Test image opening
            img = safe_image_open(img_path)
            img.verify()  # Verify it's a valid image
            
            # Check corresponding label file
            lbl_path = os.path.join(lbl_dir, Path(img_path).stem + ".txt")
            if os.path.exists(lbl_path):
                valid_count += 1
            else:
                print(f"Warning: No label file for {img_path}")
                
        except Exception as e:
            corrupted_count += 1
            print(f"Corrupted image: {img_path} - {e}")
    
    print(f"Dataset validation: {valid_count} valid, {corrupted_count} corrupted images")
    return valid_count, corrupted_count

# =========================
# Utils - IMPROVED
# =========================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED)

def debug_model_output(out, prefix=""):
    """Debug function to understand the structure of the outputs - FIXED"""
    print(f"{prefix} Type: {type(out)}")
    if isinstance(out, dict):
        print(f"{prefix} Dict keys: {list(out.keys())}")
        for k, v in out.items():
            if torch.is_tensor(v):
                # FIX: Only calculate mean for float tensors
                if v.dtype in [torch.float16, torch.float32, torch.float64]:
                    print(f"{prefix}   {k}: tensor with shape {v.shape}, mean: {v.mean().item():.4f}")
                else:
                    print(f"{prefix}   {k}: tensor with shape {v.shape}, dtype: {v.dtype}")
            else:
                print(f"{prefix}   {k}: {type(v)}")
    elif isinstance(out, (list, tuple)):
        print(f"{prefix} Sequence length: {len(out)}")
        for i, item in enumerate(out):
            debug_model_output(item, prefix + f"  [{i}]")
    elif torch.is_tensor(out):
        # FIX: Only calculate mean for float tensors
        if out.dtype in [torch.float16, torch.float32, torch.float64]:
            print(f"{prefix} Tensor shape: {out.shape}, mean: {out.mean().item():.4f}")
        else:
            print(f"{prefix} Tensor shape: {out.shape}, dtype: {out.dtype}")
    return out

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

def load_yolo_txt(lbl_path, img_w, img_h):
    boxes, labels = [], []
    if not os.path.exists(lbl_path) or os.path.getsize(lbl_path) == 0:
        return np.zeros((0,4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    
    with open(lbl_path, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    
    for ln in lines:
        parts = ln.split()
        if len(parts) != 5:
            continue
        cls, cx, cy, w, h = map(float, parts)
        x1 = (cx - w/2.0) * img_w
        y1 = (cy - h/2.0) * img_h
        x2 = (cx + w/2.0) * img_w
        y2 = (cy + h/2.0) * img_h
        
        if x2 <= x1 or y2 <= y1:
            continue
        if x1 >= img_w or y1 >= img_h or x2 <= 0 or y2 <= 0:
            continue
            
        boxes.append([x1, y1, x2, y2])
        labels.append(int(cls) + 1)  # 0 is background in detection models
    
    if not boxes:
        return np.zeros((0,4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    
    boxes = np.array(boxes, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    
    boxes[:, [0,2]] = boxes[:, [0,2]].clip(0, img_w)
    boxes[:, [1,3]] = boxes[:, [1,3]].clip(0, img_h)
    
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes = boxes[valid]
    labels = labels[valid]
    
    return boxes, labels

class YoloDetectDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, augment=True):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.*")))
        self.lbl_dir = lbl_dir
        self.augment = augment
        print(f"Found {len(self.img_paths)} images in {img_dir}")
        
        # Validate dataset files
        valid_count, corrupted_count = validate_dataset_files(img_dir, lbl_dir)
        
        # Debug: check labels
        label_counts = 0
        for img_path in self.img_paths[:10]:  # Check first 10
            lbl_path = os.path.join(self.lbl_dir, Path(img_path).stem + ".txt")
            if os.path.exists(lbl_path) and os.path.getsize(lbl_path) > 0:
                label_counts += 1
        print(f"Debug: {label_counts}/10 images have non-empty labels")
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        p = self.img_paths[idx]
        
        # Use safe image loading - FIXED
        img = safe_image_open(p)
        img = img.resize((IMG_SIZE, IMG_SIZE), resample=Image.BILINEAR)
        arr = np.asarray(img).astype(np.float32) / 255.0
        
        arr = (arr - np.array(MEAN)) / np.array(STD)
        
        flipped = False
        if self.augment and random.random() < 0.5:
            arr = np.ascontiguousarray(arr[:, ::-1, :])
            flipped = True
        
        H, W = arr.shape[:2]
        lbl_path = os.path.join(self.lbl_dir, Path(p).stem + ".txt")
        boxes, labels = load_yolo_txt(lbl_path, W, H)
        
        if flipped and boxes.shape[0] > 0:
            boxes[:, [0, 2]] = W - boxes[:, [2, 0]]
        
        img_t = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float()
        
        if boxes.shape[0] == 0:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.int64),
                "image_id": torch.tensor([idx]),
                "img_path": p,
            }
        else:
            target = {
                "boxes": torch.from_numpy(boxes).float(),
                "labels": torch.from_numpy(labels).long(),
                "image_id": torch.tensor([idx]),
                "img_path": p,
            }
        
        return img_t, target

def collate_fn(batch):
    imgs, targets = list(zip(*batch))
    return torch.stack(imgs, 0), list(targets)
#
def build_model():
    weights = spm.Weights()
    model = weights.get_pretrained_model(
        MODEL_ID, fpn=True, head=spm.Head.DETECT, num_categories=NUM_CLASSES + 1,device=DEVICE
    )
    return model




@torch.no_grad()
def predict_dataset(model, loader, device):
    """
    CRITICAL FIX: Handle tuple output from model
    """
    model.eval()
    outputs = []
    
    for batch_idx, (imgs, targets) in enumerate(loader):
        imgs = imgs.to(device)
        preds = model(imgs)
        
        # FIX: Handle tuple output - take the first element which contains predictions
        if isinstance(preds, tuple) and len(preds) >= 1:
            preds = preds[0]  # Take the predictions part of the tuple
        
        # Debug first batch
        if batch_idx == 0:
            print(f"DEBUG - Prediction structure analysis:")
            print(f"Type: {type(preds)}")
            if isinstance(preds, list):
                print(f"List length: {len(preds)}")
                if len(preds) > 0:
                    first_pred = preds[0]
                    print(f"First prediction type: {type(first_pred)}")
                    if isinstance(first_pred, dict):
                        print(f"Keys: {list(first_pred.keys())}")
                        for k, v in first_pred.items():
                            if torch.is_tensor(v):
                                print(f"  {k}: shape {v.shape}, dtype {v.dtype}")
                                if v.numel() > 0 and v.dtype in [torch.float16, torch.float32, torch.float64]:
                                    print(f"    value range: {v.min().item():.3f} to {v.max().item():.3f}")
        
        # Handle different prediction formats
        batch_preds = []
        if isinstance(preds, list):
            batch_preds = preds
        elif isinstance(preds, dict):
            batch_preds = [preds]
        elif torch.is_tensor(preds):
            print(f"WARNING: Unexpected tensor prediction format")
            continue
        else:
            print(f"WARNING: Unknown prediction type: {type(preds)}")
            continue
        
        for i, pred in enumerate(batch_preds):
            # Extract predictions with confidence filtering
            if isinstance(pred, dict):
                boxes = pred.get("boxes", torch.empty(0, 4))
                scores = pred.get("scores", torch.empty(0))
                labels = pred.get("labels", torch.empty(0, dtype=torch.int64))
                
                # Apply confidence threshold
                if len(scores) > 0:
                    keep = scores >= CONFIDENCE_THRESHOLD
                    boxes = boxes[keep]
                    scores = scores[keep]
                    labels = labels[keep]
                    
                    # Debug: print number of detections
                    if batch_idx == 0 and i == 0:
                        print(f"DEBUG: First image - {len(scores)} detections after thresholding")
                        if len(scores) > 0:
                            print(f"DEBUG: Scores range: {scores.min().item():.3f} to {scores.max().item():.3f}")
                            print(f"DEBUG: Labels: {labels.cpu().numpy()}")
                
            elif hasattr(pred, 'bbox') and hasattr(pred, 'scores') and hasattr(pred, 'labels'):
                boxes = pred.bbox
                scores = pred.scores  
                labels = pred.labels
                
                # Apply confidence threshold
                if len(scores) > 0:
                    keep = scores >= CONFIDENCE_THRESHOLD
                    boxes = boxes[keep]
                    scores = scores[keep]
                    labels = labels[keep]
            else:
                print(f"WARNING: Unexpected prediction format: {type(pred)}")
                boxes = torch.empty(0, 4)
                scores = torch.empty(0)
                labels = torch.empty(0, dtype=torch.int64)
            
            # Ensure all tensors are on CPU for serialization
            out = {
                "image_path": targets[i]["img_path"] if i < len(targets) else f"unknown_{batch_idx}_{i}",
                "boxes": boxes.cpu().numpy().tolist(),
                "scores": scores.cpu().numpy().tolist(),
                "labels": labels.cpu().numpy().tolist(),
                "img_size": [IMG_SIZE, IMG_SIZE],
            }
            outputs.append(out)
    
    # Debug: count non-empty predictions
    non_empty = sum(1 for o in outputs if len(o["boxes"]) > 0)
    print(f"Prediction summary: {non_empty}/{len(outputs)} images have detections")
    
    return outputs

def save_loss_curves(train_losses, val_losses, out_path):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses)+1), train_losses, 'b-', label="Train Loss", linewidth=2)
    if any(val_loss > 0 for val_loss in val_losses):
        plt.plot(range(1, len(val_losses)+1), val_losses, 'r-', label="Val Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

# =========================
# Training improvements
# =========================
def main():
    print("Initialization...")
    print(f"Device: {DEVICE}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    
    # Data loading with safe image handling
    print("Loading datasets with safe image validation...")
    train_ds = YoloDetectDataset(IMG_DIR_TRAIN, LBL_DIR_TRAIN, augment=True)
    val_ds = YoloDetectDataset(IMG_DIR_VAL, LBL_DIR_VAL, augment=False)
    test_ds = YoloDetectDataset(IMG_DIR_TEST, LBL_DIR_TEST, augment=False)
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    if len(train_ds) == 0:
        raise ValueError("No training images found!")
    
    # Use fewer workers to avoid data loading issues
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, collate_fn=collate_fn)
    
    # Model
    print("Loading Satlas model...")
    model = build_model().to(DEVICE)
    
    # Print model structure for debugging
    print("Model structure:")
    print(model)
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    train_losses, val_losses = [], []
    best_val = float("inf")
    epochs_no_improve = 0
    
    def train_one_epoch():
        model.train()
        running_loss = 0.0
        num_batches = 0
        
        for batch_idx, (imgs, targets) in enumerate(train_loader):
            imgs = imgs.to(DEVICE)
            processed_targets = []
            
            for t in targets:
                target_dict = {}
                for k, v in t.items():
                    if k == "img_path":
                        target_dict[k] = v
                    elif torch.is_tensor(v):
                        target_dict[k] = v.to(DEVICE)
                    else:
                        target_dict[k] = v
                processed_targets.append(target_dict)
            
            optimizer.zero_grad()
            
            try:
                out = model(imgs, processed_targets)
#                print("regarde ca ",out)
                loss = total_loss_from_model_output(out)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Batch {batch_idx}: Loss NaN/Inf - skipping")
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                running_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")
                    
            except Exception as e:
                print(f"Error batch {batch_idx}: {e}")
                continue
        
        avg_loss = running_loss / max(1, num_batches)
        print(f"  Epoch Train Loss: {avg_loss:.4f} from {num_batches} batches")
        return avg_loss
    
    @torch.no_grad()
    def val_one_epoch():
        model.train()  # Important so Faster R-CNN returns losses
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (imgs, targets) in enumerate(val_loader):
            imgs = imgs.to(DEVICE)
            processed_targets = []
            
            for t in targets:
                target_dict = {}
                for k, v in t.items():
                    if k == "img_path":
                        target_dict[k] = v
                    elif torch.is_tensor(v):
                        target_dict[k] = v.to(DEVICE)
                    else:
                        target_dict[k] = v
                processed_targets.append(target_dict)
            
            try:
                out = model(imgs, processed_targets)
                loss = total_loss_from_model_output(out)
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    total_loss += loss.item()
                    num_batches += 1
                
                if batch_idx % 5 == 0:
                    print(f"  Val Batch {batch_idx} - Loss: {loss.item():.4f}")
                    
            except Exception as e:
                print(f"Validation error batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / max(1, num_batches)
        print(f"  Epoch Val Loss: {avg_loss:.4f} from {num_batches} batches")
        return avg_loss
    
    # Training loop
    print("Start training...")
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        
        # Warmup LR
        if epoch < WARMUP_EPOCHS:
            warm_lr = LR * (epoch + 1) / WARMUP_EPOCHS
            for g in optimizer.param_groups:
                g["lr"] = warm_lr
        
        # Training
        tr_loss = train_one_epoch()
        
        # Validation
        va_loss = val_one_epoch()
        
        # Scheduler update
        scheduler.step(va_loss)
        
        train_losses.append(tr_loss)
        val_losses.append(va_loss)
        
        print(f"Epoch {epoch+1:03d}/{EPOCHS} | train={tr_loss:.4f} | val={va_loss:.4f} | LR={optimizer.param_groups[0]['lr']:.2e}")
        
        # Save
        save_loss_curves(train_losses, val_losses, LOSS_PNG)
        
        # Early stopping
        if va_loss + 1e-9 < best_val:
            best_val = va_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), BEST_WEIGHTS)
            print(f"  ? New best model saved (loss: {va_loss:.4f})")
            
            try:
                # Predictions with the best model
                print("  Generating predictions...")
                best_model = build_model().to(DEVICE)
                best_model.load_state_dict(torch.load(BEST_WEIGHTS, map_location=DEVICE))
                
                val_preds = predict_dataset(best_model, val_loader, DEVICE)
                with open(VAL_PREDS_JSON, "w") as f:
                    json.dump(val_preds, f, indent=2)
                print(f"  ? Validation predictions saved: {VAL_PREDS_JSON}")
                
                test_preds = predict_dataset(best_model, test_loader, DEVICE)
                with open(TEST_PREDS_JSON, "w") as f:
                    json.dump(test_preds, f, indent=2)
                print(f"  ? Test predictions saved: {TEST_PREDS_JSON}")
                
            except Exception as e:
                print(f"  ? Error while generating predictions: {e}")
                print("  But the model is saved; you can generate predictions later.")
                
        else:
            epochs_no_improve += 1
            print(f"  ? No improvement for {epochs_no_improve} epochs")
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping after {epoch+1} epochs")
                break
    
    print("\n" + "="*50)
    print("TRAINING FINISHED")
    print(f"Best validation loss: {best_val:.4f}")
    print(f"Model saved: {BEST_WEIGHTS}")

if __name__ == "__main__":
    main()