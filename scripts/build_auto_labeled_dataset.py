import os
import shutil
import random

# ------------ CONFIGURATION ------------
school_img_dir = "dataset/usa/satellite/school"
school_lbl_dir = "dataset/usa/satellite/school_labels"
non_school_img_dir = "data/non_school"

base_output_dir = "dataset/usa/dataset_yolo_auto_labeling"
img_ext = ".png"
seed = 42
max_non_school = 3000  #limit non_school to 1500

# Split ratios
split_ratio = {"train": 0.8, "val": 0.15, "test": 0.05}
splits = ["train", "val", "test"]

## ------------ SETUP ------------
tmp_img_dir = os.path.join(base_output_dir, "images_all")
tmp_lbl_dir = os.path.join(base_output_dir, "labels_all")
#os.makedirs(tmp_img_dir, exist_ok=True)
#os.makedirs(tmp_lbl_dir, exist_ok=True)
#
#for split in splits:
#    os.makedirs(os.path.join(base_output_dir, f"images/{split}"), exist_ok=True)
#    os.makedirs(os.path.join(base_output_dir, f"labels/{split}"), exist_ok=True)

# ------------ STEP 1: Fusionner les images school et non-school ------------

## Copy school images and their labels
#for img_file in os.listdir(school_img_dir):
#    if img_file.endswith(img_ext):
#        base = os.path.splitext(img_file)[0]
#        img_path = os.path.join(school_img_dir, img_file)
#        lbl_path = os.path.join(school_lbl_dir, f"{base}.txt")
#
#        shutil.copy(img_path, os.path.join(tmp_img_dir, img_file))
#        if os.path.exists(lbl_path):
#            shutil.copy(lbl_path, os.path.join(tmp_lbl_dir, f"{base}.txt"))
#
## Copy limited number of non-school images and create empty labels
#non_school_files = [f for f in os.listdir(non_school_img_dir) if f.endswith(img_ext)]
#random.seed(seed)
#random.shuffle(non_school_files)
#non_school_files = non_school_files[:max_non_school]  # <<<<< LIMITATION ICI
#
#for img_file in non_school_files:
#    base = os.path.splitext(img_file)[0]
#    img_path = os.path.join(non_school_img_dir, img_file)
#
#    shutil.copy(img_path, os.path.join(tmp_img_dir, img_file))
#    open(os.path.join(tmp_lbl_dir, f"{base}.txt"), "w").close()
#
#print(f"Step 1 done: {len(os.listdir(tmp_img_dir))} images prepared.")

# ------------ STEP 2: Split train/val/test ------------
random.seed(seed)

# Shuffle image files
all_images = [f for f in os.listdir(tmp_img_dir) if f.endswith(img_ext)]
random.shuffle(all_images)

total = len(all_images)
train_end = int(split_ratio["train"] * total)
val_end = train_end + int(split_ratio["val"] * total)

split_files = {
    "train": all_images[:train_end],
    "val": all_images[train_end:val_end],
    "test": all_images[val_end:]
}

# Move image/label pairs
for split_name, img_list in split_files.items():
    for img_file in img_list:
        base = os.path.splitext(img_file)[0]
        label_file = f"{base}.txt"

        src_img = os.path.join(tmp_img_dir, img_file)
        src_lbl = os.path.join(tmp_lbl_dir, label_file)

        dst_img = os.path.join(base_output_dir, f"images/{split_name}", img_file)
        dst_lbl = os.path.join(base_output_dir, f"labels/{split_name}", label_file)

        shutil.copy(src_img, dst_img)
        shutil.copy(src_lbl, dst_lbl)

print("Step 2 done: Train/Val/Test split completed.")
