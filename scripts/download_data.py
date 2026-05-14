# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np

from pystac_client import Client
import rasterio
from rasterio.windows import Window
from pyproj import Transformer
from PIL import Image

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = "data/usa" # change as needed

CSV_PATH = os.path.join(BASE_DIR, "remaining_schools_usa.csv")   # input CSV
SAT_DIR = os.path.join(BASE_DIR, "remaining_schools")
LABELS_CSV = os.path.join(BASE_DIR, "remaining_files_naip.csv")

# Target ML-safe configuration
TARGET_GSD = 0.6                 # meters / pixel
TARGET_SIZE_PX = 500
TARGET_SIZE_M = TARGET_GSD * TARGET_SIZE_PX  # 300 meters

# NAIP search
NAIP_START_YEAR = 2023
NAIP_END_YEAR = 2025
SEARCH_DELTA = 0.003              # ~666 m bbox

# ============================================================
# PREPARE DIRECTORIES
# ============================================================

for sub in ("school", "non_school"):
    os.makedirs(os.path.join(SAT_DIR, sub), exist_ok=True)

# ============================================================
# OPEN STAC CATALOG
# ============================================================

catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1"
)



df = pd.read_csv(CSV_PATH)
labels = []

# ============================================================
# MAIN LOOP
# ============================================================

for idx, row in df.iterrows():
    lat = float(row["Latitude"])
    lon = float(row["Longitude"])
    label = row["label"]

    sub_folder = "school" if label == "school" else "non_school"
    filename = f"{label}_lat={lat}_lon={lon}.png"
    filepath = os.path.join(SAT_DIR, sub_folder, filename)

    if os.path.exists(filepath):
        continue

    try:
        # ----------------------------------------------------
        #  SEARCH NAIP TILE
        # ----------------------------------------------------
        bbox = [
            lon - SEARCH_DELTA,
            lat - SEARCH_DELTA,
            lon + SEARCH_DELTA,
            lat + SEARCH_DELTA,
        ]

        search = catalog.search(
            collections=["naip"],
            bbox=bbox,
            datetime=f"{NAIP_START_YEAR}-01-01/{NAIP_END_YEAR}-12-31",
            limit=1,
        )

        items = list(search.items())
        if not items:
            print(f"[WARN] No NAIP tile for ({lat}, {lon})")
            continue

        asset_url = items[0].assets["image"].href

        # ----------------------------------------------------
        #  OPEN IMAGE AND NORMALIZED CROP
        # ----------------------------------------------------
        with rasterio.open(asset_url) as src:
            src_res = src.res[0]  # meters / pixel

            transformer = Transformer.from_crs(
                "EPSG:4326", src.crs, always_xy=True
            )
            x, y = transformer.transform(lon, lat)
            row_px, col_px = src.index(x, y)

            crop_px = int(TARGET_SIZE_M / src_res)
            half = crop_px // 2

            window = Window(
                col_px - half,
                row_px - half,
                crop_px,
                crop_px,
            )

            img = src.read([1, 2, 3], window=window)
            img = np.transpose(img, (1, 2, 0))

        
        if src_res != TARGET_GSD:
            img = np.array(
                Image.fromarray(img).resize(
                    (TARGET_SIZE_PX, TARGET_SIZE_PX),
                    resample=Image.BILINEAR,
                )
            )
        else:
            img = np.array(
                Image.fromarray(img).resize(
                    (TARGET_SIZE_PX, TARGET_SIZE_PX),
                    resample=Image.NEAREST,
                )
            )

        # ----------------------------------------------------
        #  SAVE IMAGE
        # ----------------------------------------------------
        Image.fromarray(img).save(filepath)

        labels.append([
            filename,
            "satellite_naip",
            label,
            lat,
            lon,
            src_res,
        ])

        print(f"[OK] {filename} | native_res={src_res:.2f} m")

    except Exception as e:
        print(f"[ERROR] ({lat}, {lon}) ? {e}")

# ============================================================
# EXPORT LABELS
# ============================================================

pd.DataFrame(
    labels,
    columns=[
        "filename",
        "modality",
        "label",
        "latitude",
        "longitude",
        "native_resolution_m",
    ],
).to_csv(LABELS_CSV, index=False)

print(f"\n[INFO] Dataset ready. Labels saved to {LABELS_CSV}")
