import os
import cv2
import glob
import numpy as np
import random

# Directories (train only)
IMG_DIR = "dataset/usa/golden_data/images_samples/val"
LABEL_DIR = "dataset/usa/golden_data/labels_samples/val"
IMG_EXT = ".png"

# Parameters
AUG_PER_IMAGE = 3
IMG_SIZE = 500
MAX_ROTATION = 15
MAX_TRANSLATION = 175
seed=42
random.seed(seed)

# Get all training images
images = glob.glob(os.path.join(IMG_DIR, "*" + IMG_EXT))

for img_path in images:
    img_name = os.path.basename(img_path).replace(IMG_EXT, "")
    label_path = os.path.join(LABEL_DIR, f"{img_name}.txt")

    # Load image and label
    img = cv2.imread(img_path)
    if img is None or not os.path.exists(label_path):
        print(f"Skipping {img_name} (image or label missing).")
        continue

    with open(label_path, "r") as f:
        label_line = f.readline().strip()

    if label_line:  # Positive sample
        class_id, x_center, y_center, w, h = map(float, label_line.split())
        bbox_x = x_center * IMG_SIZE
        bbox_y = y_center * IMG_SIZE
        bbox = np.array([[bbox_x, bbox_y]], dtype=np.float32)

        for aug_idx in range(AUG_PER_IMAGE):
            angle = random.uniform(-MAX_ROTATION, MAX_ROTATION)
            tx = random.uniform(-MAX_TRANSLATION, MAX_TRANSLATION)
            ty = random.uniform(-MAX_TRANSLATION, MAX_TRANSLATION)

            # Build affine transform: rotation + translation
            M = cv2.getRotationMatrix2D((IMG_SIZE / 2, IMG_SIZE / 2), angle, 1.0)
            M[:, 2] += [tx, ty]

            aug_img = cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE), borderMode=cv2.BORDER_REFLECT)

            # Apply transform to bbox center
            bbox_homogeneous = np.hstack([bbox, np.ones((bbox.shape[0], 1))])
            bbox_aug = np.dot(M, bbox_homogeneous.T).T[0]
            new_x, new_y = bbox_aug

            if 0 <= new_x <= IMG_SIZE and 0 <= new_y <= IMG_SIZE:
                new_x_norm = new_x / IMG_SIZE
                new_y_norm = new_y / IMG_SIZE
                new_label_line = f"{int(class_id)} {new_x_norm:.6f} {new_y_norm:.6f} {w:.6f} {h:.6f}"

                aug_img_name = f"{img_name}_aug{aug_idx}"
                cv2.imwrite(os.path.join(IMG_DIR, f"{aug_img_name}{IMG_EXT}"), aug_img)
                with open(os.path.join(LABEL_DIR, f"{aug_img_name}.txt"), "w") as f:
                    f.write(new_label_line)

                print(f"Positive aug: {aug_img_name}")
            else:
                print(f"Skipped {aug_idx} of {img_name} (bbox out of image)")

    else:  # Negative sample (empty label)
        for aug_idx in range(AUG_PER_IMAGE):
            angle = random.uniform(-MAX_ROTATION, MAX_ROTATION)
            tx = random.uniform(-MAX_TRANSLATION, MAX_TRANSLATION)
            ty = random.uniform(-MAX_TRANSLATION, MAX_TRANSLATION)

            M = cv2.getRotationMatrix2D((IMG_SIZE / 2, IMG_SIZE / 2), angle, 1.0)
            M[:, 2] += [tx, ty]

            aug_img = cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE), borderMode=cv2.BORDER_REFLECT)

            aug_img_name = f"{img_name}_aug{aug_idx}"
            cv2.imwrite(os.path.join(IMG_DIR, f"{aug_img_name}{IMG_EXT}"), aug_img)
            with open(os.path.join(LABEL_DIR, f"{aug_img_name}.txt"), "w") as f:
                pass  # empty label

            print(f"Negative aug: {aug_img_name}")