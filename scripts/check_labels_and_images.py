import os

IMAGE_EXTS = (".jpg", ".jpeg", ".png")

IMG_ROOT = "dataset/usa/golden_data/images_samples"
LBL_ROOT = "dataset/usa/golden_data/labels_samples"

splits = ["train", "val", "test"]

def get_images(folder):
    return {
        os.path.splitext(f)[0]: f
        for f in os.listdir(folder)
        if f.lower().endswith(IMAGE_EXTS)
    }

def get_labels(folder):
    return {
        os.path.splitext(f)[0]: f
        for f in os.listdir(folder)
        if f.lower().endswith(".txt")
    }

for split in splits:
    print(f"\n🔍 Checking split: {split}")

    img_dir = os.path.join(IMG_ROOT, split)
    lbl_dir = os.path.join(LBL_ROOT, split)

    images = get_images(img_dir)
    labels = get_labels(lbl_dir)

    # Images without labels
    missing_labels = set(images.keys()) - set(labels.keys())
    # Labels without images
    missing_images = set(labels.keys()) - set(images.keys())

    if not missing_labels and not missing_images:
        print("✅ OK: images et labels bien appariés")
        continue

    if missing_labels:
        print(f"❌ Images sans label ({len(missing_labels)}):")
        for k in sorted(missing_labels):
            print(f"   - {images[k]}")

    if missing_images:
        print(f"❌ Labels sans image ({len(missing_images)}):")
        for k in sorted(missing_images):
            print(f"   - {labels[k]}")

print("\n✅ Vérification terminée.")
