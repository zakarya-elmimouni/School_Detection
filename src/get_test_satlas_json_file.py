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


import satlaspretrain_models as spm


SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE=8
NUM_CLASSES=1
CONFIDENCE_THRESHOLD = 0.01 
IMG_SIZE=400


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

OUT_DIR = "results/usa/satlas_tiny_train_ecp"
BEST_WEIGHTS = os.path.join(OUT_DIR, "best_global_model.pt")
DATA_ROOT = "dataset/usa/golden_data_tiny_train"
IMG_DIR_TEST = f"{DATA_ROOT}/images/test"
LBL_DIR_TEST = f"{DATA_ROOT}/labels/test"
TEST_PREDS_JSON = os.path.join(OUT_DIR, "best_test_preds.json")


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
    

MODEL_ID = "Aerial_SwinB_SI"
best_model = build_model().to(DEVICE)
best_model.load_state_dict(torch.load(BEST_WEIGHTS, map_location=DEVICE))

test_ds = YoloDetectDataset(IMG_DIR_TEST, LBL_DIR_TEST, augment=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, collate_fn=collate_fn)

test_preds = predict_dataset(best_model, test_loader, DEVICE)
with open(TEST_PREDS_JSON, "w") as f:
  json.dump(test_preds, f, indent=2)
print(f"  ? Test predictions saved: {TEST_PREDS_JSON}")