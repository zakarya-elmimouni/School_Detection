#!/usr/bin/env python3
import shutil
import random
from pathlib import Path

# ------------- CONFIGURATION ------------------------------------------------
BASE_DIR       = Path("dataset/usa/golden_data_small")
IMAGE_DIR      = BASE_DIR / "images"
LABEL_DIR      = BASE_DIR / "labels"

SPLIT_RATIOS   = {"train": 0.6, "val": 0.15, "test": 0.25}
USE_COPY       = True   # True = copy files, False = move files
RANDOM_SEED    = 42     # set None for non-deterministic split

IMG_EXTS       = {".png"}
# ----------------------------------------------------------------------------

def make_subset_dirs() -> dict[str, dict[str, Path]]:
    
    paths = {}
    for subset in SPLIT_RATIOS.keys():
        img_sub = IMAGE_DIR / subset
        lbl_sub = LABEL_DIR / subset
        img_sub.mkdir(parents=True, exist_ok=True)
        lbl_sub.mkdir(parents=True, exist_ok=True)
        paths[subset] = {"img": img_sub, "lbl": lbl_sub}
    return paths

def collect_pairs() -> list[tuple[Path, Path]]:
  
    images = [p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in IMG_EXTS and p.is_file()]
    pairs  = []
    for img in images:
        label = LABEL_DIR / f"{img.stem}.txt"
        if label.exists():
            pairs.append((img, label))
        else:
            print(f"[WARNING] No label found for {img.name} skipped.")
    return pairs

def split_pairs(pairs: list[tuple[Path, Path]]):

    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
    random.shuffle(pairs)

    n_total = len(pairs)
    n_train = int(SPLIT_RATIOS["train"] * n_total)
    n_val   = int(SPLIT_RATIOS["val"]   * n_total)

    boundaries = {
        "train": (0, n_train),
        "val"  : (n_train, n_train + n_val),
        "test" : (n_train + n_val, n_total)
    }

    for subset, (start, end) in boundaries.items():
        for pair in pairs[start:end]:
            yield subset, pair

def transfer(src: Path, dst: Path) -> None:
   
    if USE_COPY:
        shutil.copy2(src, dst)
    else:
        shutil.move(src, dst)

def main() -> None:
    subset_dirs = make_subset_dirs()
    pairs = collect_pairs()
    print(f"Total valid pairs found: {len(pairs)}")

    counts = {k: 0 for k in SPLIT_RATIOS}
    for subset, (img, lbl) in split_pairs(pairs):
        transfer(img, subset_dirs[subset]["img"] / img.name)
        transfer(lbl, subset_dirs[subset]["lbl"] / lbl.name)
        counts[subset] += 1

    print("Split complete:")
    for subset, n in counts.items():
        print(f"  {subset:<5}: {n} pairs")

if __name__ == "__main__":
    main()
