import os
import random
import shutil
from pathlib import Path

# =========================
# CONFIG
# =========================

root_directory="dataset/usa/golden_data_small_train/"
IMG_DIR = Path(root_directory+"images/train")
LBL_DIR = Path(root_directory+"labels/train")

OUTPUT_IMG_DIR = Path(root_directory+"subset/images/train")
OUTPUT_LBL_DIR = Path(root_directory+"subset/labels/train")

TARGET_TOTAL = 300
SEED = 42

random.seed(SEED)

# =========================
# CREATE OUTPUT FOLDERS
# =========================
OUTPUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_LBL_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# SEPARATE SCHOOL / NON-SCHOOL
# =========================
school_images = []
non_school_images = []

for label_file in LBL_DIR.glob("*.txt"):
    image_file = IMG_DIR / (label_file.stem + ".png")  # adapte si png
    
    if not image_file.exists():
        continue
    
    # Si le fichier label est vide => non school
    if os.path.getsize(label_file) == 0:
        non_school_images.append((image_file, label_file))
    else:
        school_images.append((image_file, label_file))

print(f"Total school images: {len(school_images)}")
print(f"Total non-school images: {len(non_school_images)}")

# =========================
# CALCULATE DISTRIBUTION
# =========================
total_images = len(school_images) + len(non_school_images)

school_ratio = len(school_images) / total_images
non_school_ratio = len(non_school_images) / total_images

target_school = int(TARGET_TOTAL * school_ratio)
target_non_school = TARGET_TOTAL - target_school

print(f"Keeping {target_school} school images")
print(f"Keeping {target_non_school} non-school images")

# =========================
# RANDOM SAMPLING
# =========================
selected_school = random.sample(school_images, target_school)
selected_non_school = random.sample(non_school_images, target_non_school)

selected_all = selected_school + selected_non_school

# =========================
# COPY FILES
# =========================
for img_path, lbl_path in selected_all:
    shutil.copy(img_path, OUTPUT_IMG_DIR / img_path.name)
    shutil.copy(lbl_path, OUTPUT_LBL_DIR / lbl_path.name)

print("Done ? Subset created successfully.")