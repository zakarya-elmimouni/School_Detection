import os

""" this script removes images and labels from the train auto-labeled set that are also present in the val and test sets of the golden dataset"""


# ------------ CONFIGURATION ------------
base_path = "dataset/usa/golden_data"   # change as needed
target_path="dataset/usa/dataset_yolo_auto_labeling" #change as needed
  # Allowed image extensions
img_extensions = (".jpg", ".jpeg", ".png")
# ------------ STEP 1: Collect all image filenames from val and test ------------
val_test_images = set()
for split in ["val", "test"]:
    split_dir = os.path.join(base_path, "images", split)
    for filename in os.listdir(split_dir):
        if filename.endswith(img_extensions):
            val_test_images.add(filename)

# ------------ STEP 2: Remove duplicates from train ------------
train_img_dir = os.path.join(target_path, "images", "train")
train_lbl_dir = os.path.join(target_path, "labels", "train")

removed_count = 0

for filename in os.listdir(train_img_dir):
    if filename in val_test_images:
        # Remove image from train
        img_path = os.path.join(train_img_dir, filename)
        os.remove(img_path)

        # Remove corresponding label from train (same filename but .txt)
        label_name = os.path.splitext(filename)[0] + ".txt"
        label_path = os.path.join(train_lbl_dir, label_name)
        if os.path.exists(label_path):
            os.remove(label_path)

        removed_count += 1
        print(f"Removed from train: {filename} and label {label_name}")

print(f"\nDone. Total removed from train: {removed_count} images and labels.")
