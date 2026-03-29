import os
import random
import cv2

# Paths
IMG_DIR = "dataset/usa/golden_data/images_samples/train"
LBL_DIR = "dataset/usa/golden_data/labels_samples/train"
OUT_DIR = "dataset/usa/golden_data/get_viz"

os.makedirs(OUT_DIR, exist_ok=True)

# Parameters
N_SAMPLES = 30
IMAGE_EXTS = (".jpg", ".jpeg", ".png")

# Get images
images = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(IMAGE_EXTS)]
selected_images = random.sample(images, min(N_SAMPLES, len(images)))

def draw_yolo_bboxes(img, label_path):
    h, w, _ = img.shape

    if not os.path.exists(label_path):
        return img

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        _, x_c, y_c, bw, bh = map(float, parts)

        # YOLO → pixel coords
        x1 = int((x_c - bw / 2) * w)
        y1 = int((y_c - bh / 2) * h)
        x2 = int((x_c + bw / 2) * w)
        y2 = int((y_c + bh / 2) * h)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return img

for img_name in selected_images:
    img_path = os.path.join(IMG_DIR, img_name)
    lbl_path = os.path.join(LBL_DIR, os.path.splitext(img_name)[0] + ".txt")

    img = cv2.imread(img_path)
    if img is None:
        continue

    img = draw_yolo_bboxes(img, lbl_path)

    out_path = os.path.join(OUT_DIR, img_name)
    cv2.imwrite(out_path, img)

print(f"✅ {len(selected_images)} images sauvegardées dans {OUT_DIR}")
