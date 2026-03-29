# -*- coding: utf-8 -*- 

import os
import json
import cv2
import numpy as np
import torch
from PIL import Image
from samgeo.text_sam import LangSAM

# ---------------- Configuration ----------------
INPUT_DIR  = "usa_school_mapping/data/usa/school"
OUTPUT_DIR = "dataset/usa/satellite"
CROP_SIZE  = 400

GLOBAL_VEG_THRESHOLD = 0.8
SAHARA_THRESHOLD = 0.8
SEA_THRESHOLD = 0.8
BLUR_VAR_THRESHOLD   = 100
FALLBACK_BOX_SIZE    = 100
MASK_OUTLIER_RATIO   = 0.80

CONF_THRESHOLD       = 0.25
MIN_COMPONENT_AREA   = 300
MAX_COMPONENT_AREA   = int(0.40 * CROP_SIZE * CROP_SIZE)
MAX_CENTER_DISTANCE  = 120
MIN_SOLIDITY         = 0.30
MIN_ASPECT_RATIO     = 0.2
MAX_VEGETATION       = 0.35
MIN_EDGE_DENSITY     = 0.08
MAX_VEG_OUTLIER      = 0.60
MAX_MASK_RATIO       = 0.90

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

# ---------------- Output directories ----------------
VALID_IMG_DIR = os.path.join(OUTPUT_DIR, "school")
VALID_LABEL_DIR = os.path.join(OUTPUT_DIR, "school_labels")
OUTLIER_DIR = os.path.join(OUTPUT_DIR, "outliers")

os.makedirs(VALID_IMG_DIR, exist_ok=True)
os.makedirs(VALID_LABEL_DIR, exist_ok=True)
os.makedirs(OUTLIER_DIR, exist_ok=True)

# ---------------- LangSAM model ----------------
sam = LangSAM(model_type="vit_h")
print("LangSAM loaded")

# ---------------- Mask helper with IoU decision tree ----------------
IOU_INTERSECT_HARD = 0.90
IOU_UNION_SOFT     = 0.30
TOP_K_CLOSEST      = 2

def get_building_mask(crop_bgr: np.ndarray) -> np.ndarray:
    """Generate building mask using LangSAM with prompt fusion"""
    img_pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    crop_area = float(CROP_SIZE * CROP_SIZE)
    cx = cy = CROP_SIZE // 2

    # Collect mask candidates
    candidates = []
    prompts = ["building", "roof", "school"]
    
    for word in prompts:
        try:
            masks, _, _, scores = sam.predict(
                img_pil,
                text_prompt=word,
                box_threshold=0.24,
                text_threshold=0.24,
                return_results=True,
            )
            if masks is None:
                continue
                
            # Handle different mask formats
            if isinstance(masks, torch.Tensor):
                masks = [m.cpu().numpy() for m in masks]
            elif isinstance(masks, np.ndarray) and masks.ndim == 3:
                masks = [masks]
                
            # Filter masks by size and position
            for m, scr in zip(masks, scores):
                area_ratio = m.sum() / crop_area
                if area_ratio >= MAX_MASK_RATIO or m.sum() < MIN_COMPONENT_AREA:
                    continue
                    
                ys, xs = np.where(m)
                if xs.size == 0:
                    continue
                    
                dist = np.hypot(xs.mean() - cx, ys.mean() - cy)
                candidates.append((dist, -float(scr), m))
        except Exception:
            torch.cuda.empty_cache()
            continue

    if not candidates:
        return np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.uint8)

    # Select top candidates by distance and score
    candidates.sort(key=lambda t: (t[0], t[1]))
    selected = [c[2] for c in candidates[:TOP_K_CLOSEST]]

    # Single mask case
    if len(selected) == 1:
        return (selected[0] > CONF_THRESHOLD).astype(np.uint8)

    # Multi-mask fusion
    m1 = selected[0] > CONF_THRESHOLD
    m2 = selected[1] > CONF_THRESHOLD
    inter = m1 & m2
    union = m1 | m2
    iou = inter.sum() / (union.sum() + 1e-5)

    # Fusion decision tree
    if iou >= IOU_INTERSECT_HARD:
        return inter.astype(np.uint8)
    elif iou >= IOU_UNION_SOFT:
        kernel = np.ones((3, 3), np.uint8)
        return cv2.morphologyEx(union.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    else:
        return m1.astype(np.uint8)

# --------------- Shape & texture helpers ---------------
def is_valid_building(cnt):
    """Check if contour meets building shape criteria"""
    area = cv2.contourArea(cnt)
    if area == 0:
        return False, 0.0, 0.0
        
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0
    
    x, y, w, h = cv2.boundingRect(cnt)
    aspect = min(w, h) / max(w, h) if max(w, h) > 0 else 0.0
    
    return (solidity >= MIN_SOLIDITY and aspect >= MIN_ASPECT_RATIO), solidity, aspect

def has_building_texture(crop, mask_region):
    """Check if masked region has building-like texture"""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    edge_density = edges[mask_region > 0].sum() / (mask_region.sum() + 1.0)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    green_mask = ((hsv[..., 0] > 30) & (hsv[..., 0] < 106) & (hsv[..., 1] > 40))
    veg_ratio = (green_mask & (mask_region > 0)).sum() / (mask_region.sum() + 1.0)
    
    
    return (edge_density > MIN_EDGE_DENSITY and veg_ratio < MAX_VEGETATION), edge_density, veg_ratio

# --------------- Main processing loop ---------------
results = []
image_files = [f for f in os.listdir(INPUT_DIR) 
              if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))]
print(f"Found {len(image_files)} images to process")

for idx, filename in enumerate(image_files, 1):
    print(f"\nProcessing [{idx}/{len(image_files)}]: {filename}")
    img_path = os.path.join(INPUT_DIR, filename)
    img = cv2.imread(img_path)
    if img is None:
        print("  Skipped: Unable to read image")
        continue

    height, width = img.shape[:2]
    info = {
        "image": filename, 
        "original_center": (width // 2, height // 2), 
        "original_size": (width, height)
    }

    # 1) Image size check
    if height < CROP_SIZE or width < CROP_SIZE:
        info.update(outlier=True, reason="image_too_small")
        results.append(info)
        cv2.imwrite(os.path.join(OUTLIER_DIR, filename), img)
        continue

    # 2) Central crop
    center_x, center_y = info["original_center"]
    left = max(0, center_x - CROP_SIZE // 2)
    top = max(0, center_y - CROP_SIZE // 2)
    right = min(width, left + CROP_SIZE)
    bottom = min(height, top + CROP_SIZE)
    
    # Adjust crop if near edges
    if right - left < CROP_SIZE:
        left = max(0, right - CROP_SIZE)
    if bottom - top < CROP_SIZE:
        top = max(0, bottom - CROP_SIZE)
    
    crop = img[top:bottom, left:right]

    # 3) Central region analysis (200x200 center)
    center_half = 100
    center_left = max(0, CROP_SIZE // 2 - center_half)
    center_top = max(0, CROP_SIZE // 2 - center_half)
    center_region = crop[center_top:center_top+200, center_left:center_left+200]
    
    hsv_center = cv2.cvtColor(center_region, cv2.COLOR_BGR2HSV)
    
    # Vegetation check
    green_mask = ((hsv_center[..., 0] > 30) & 
                 (hsv_center[..., 0] < 106) & 
                 (hsv_center[..., 1] > 40))
    green_ratio = np.sum(green_mask) / (200 * 200)
    if green_ratio > GLOBAL_VEG_THRESHOLD:
        info.update(outlier=True, reason="high_vegetation", vegetation_ratio=green_ratio)
        results.append(info)
        cv2.imwrite(os.path.join(OUTLIER_DIR, filename), img)
        continue
    
    # Desert check
    desert_mask = ((hsv_center[..., 0] > 15) & 
                  (hsv_center[..., 0] < 35) & 
                  (hsv_center[..., 1] > 50))
    desert_ratio = np.sum(desert_mask) / (200 * 200)
    if desert_ratio > SAHARA_THRESHOLD:
        info.update(outlier=True, reason="high_desert", desert_ratio=desert_ratio)
        results.append(info)
        cv2.imwrite(os.path.join(OUTLIER_DIR, filename), img)
        continue

    # Sea check
    sea_mask = ((hsv_center[..., 0] > 90) & 
               (hsv_center[..., 0] < 130) & 
               (hsv_center[..., 1] > 30))
    sea_ratio = np.sum(sea_mask) / (200 * 200)
    if sea_ratio > SEA_THRESHOLD:
        info.update(outlier=True, reason="high_sea", sea_ratio=sea_ratio)
        results.append(info)
        cv2.imwrite(os.path.join(OUTLIER_DIR, filename), img)
        continue

    # 4) Blur detection
    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur_variance = cv2.Laplacian(gray_crop, cv2.CV_64F).var()
    if blur_variance < BLUR_VAR_THRESHOLD:
        info.update(outlier=True, reason="blurry", blur_variance=blur_variance)
        results.append(info)
        cv2.imwrite(os.path.join(OUTLIER_DIR, filename), img)
        continue

    # 5) Building segmentation
    mask = get_building_mask(crop)

    # 6) Mask size check (fallback to fixed box)
    mask_ratio = np.sum(mask) / (CROP_SIZE * CROP_SIZE)
    if mask_ratio > MASK_OUTLIER_RATIO:
        half = FALLBACK_BOX_SIZE // 2
        box_x1 = max(0, center_x - half)
        box_y1 = max(0, center_y - half)
        box_x2 = min(width, box_x1 + FALLBACK_BOX_SIZE)
        box_y2 = min(height, box_y1 + FALLBACK_BOX_SIZE)
        
        info.update(
            outlier=False, 
            fallback="fixed_box_mask_too_big",
            bbox=[box_x1, box_y1, box_x2, box_y2],
            new_center=[center_x, center_y], 
            centroid_dist=0.0
        )
        results.append(info)
        cv2.imwrite(os.path.join(VALID_IMG_DIR, filename), img)
        
        # Save YOLO label
        x_center = (box_x1 + box_x2) / 2 / width
        y_center = (box_y1 + box_y2) / 2 / height
        box_width = (box_x2 - box_x1) / width
        box_height = (box_y2 - box_y1) / height
        
        label_path = os.path.join(VALID_LABEL_DIR, f"{os.path.splitext(filename)[0]}.txt")
        with open(label_path, 'w') as f:
            f.write(f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")
        continue

    # 7) Contour processing
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        info.update(outlier=True, reason="no_contours")
        results.append(info)
        cv2.imwrite(os.path.join(OUTLIER_DIR, filename), img)
        continue

    # Find best building contour
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    building_contour = None
    outlier_detected = False

    for contour in contours:
        area = cv2.contourArea(contour)
        if not (MIN_COMPONENT_AREA <= area <= MAX_COMPONENT_AREA):
            continue
            
        # Shape validation
        valid_shape, solidity, aspect = is_valid_building(contour)
        if not valid_shape:
            continue
            
        # Texture validation
        contour_mask = np.zeros_like(mask)
        cv2.drawContours(contour_mask, [contour], -1, 1, -1)
        valid_texture, edge_density, veg_ratio = has_building_texture(crop, contour_mask)
        
        # Vegetation inside mask check
        if veg_ratio > MAX_VEG_OUTLIER:
            info.update(outlier=True, reason="high_vegetation_inside", veg_ratio=veg_ratio)
            outlier_detected = True
            break
            
        if valid_shape and valid_texture:
            building_contour = contour
            break

    if outlier_detected:
        results.append(info)
        cv2.imwrite(os.path.join(OUTLIER_DIR, filename), img)
        continue
        
    if building_contour is None:
        info.update(outlier=True, reason="no_valid_building")
        results.append(info)
        cv2.imwrite(os.path.join(OUTLIER_DIR, filename), img)
        continue

    # 8) Centroid position validation
    M = cv2.moments(building_contour)
    if M["m00"] == 0:
        info.update(outlier=True, reason="invalid_centroid")
        results.append(info)
        cv2.imwrite(os.path.join(OUTLIER_DIR, filename), img)
        continue
        
    contour_cx = int(M["m10"] / M["m00"])
    contour_cy = int(M["m01"] / M["m00"])
    distance = np.hypot(contour_cx - CROP_SIZE//2, contour_cy - CROP_SIZE//2)
    
    if distance > MAX_CENTER_DISTANCE:
        info.update(outlier=True, reason="center_too_far", centroid_dist=distance)
        results.append(info)
        cv2.imwrite(os.path.join(OUTLIER_DIR, filename), img)
        continue

    # 9) Final validation - save image and label
    x, y, w, h = cv2.boundingRect(building_contour)
    abs_bbox = [left + x, top + y, left + x + w, top + y + h]
    abs_center = [left + contour_cx, top + contour_cy]
    
    info.update(
        outlier=False, 
        bbox=abs_bbox, 
        new_center=abs_center, 
        centroid_dist=distance
    )
    results.append(info)
    cv2.imwrite(os.path.join(VALID_IMG_DIR, filename), img)
    
    # Save YOLO label
    x1, y1, x2, y2 = abs_bbox
    x_center = (x1 + x2) / 2 / width
    y_center = (y1 + y2) / 2 / height
    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height
    
    label_path = os.path.join(VALID_LABEL_DIR, f"{os.path.splitext(filename)[0]}.txt")
    with open(label_path, 'w') as f:
        f.write(f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")

# --------------- Save results ---------------
with open(os.path.join(OUTPUT_DIR, "processing_results.json"), "w") as f:
    json.dump(results, f, indent=2)

valid_count = sum(1 for r in results if not r.get("outlier", True))
print(f"\nProcessing complete. Valid: {valid_count}/{len(results)}")
print(f"Output saved to: {OUTPUT_DIR}")