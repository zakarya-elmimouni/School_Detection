import os
import shutil

SOURCE_DIR = "data/school"
TARGET_DIR = "dataset/school"
TXT_FILE = "school_cleaned.txt"

# Create target directory if it does not exist
os.makedirs(TARGET_DIR, exist_ok=True)

# Read filenames from text file
with open(TXT_FILE, "r") as f:
    filenames = [line.strip() for line in f.readlines()]

moved = 0
missing = 0

for filename in filenames:
    source_path = os.path.join(SOURCE_DIR, filename)
    target_path = os.path.join(TARGET_DIR, filename)

    if os.path.exists(source_path):
        shutil.move(source_path, target_path)
        moved += 1
    else:
        print(f"Warning: File not found -> {filename}")
        missing += 1

print(f"\n? {moved} files moved successfully.")
print(f"? {missing} files were missing.")