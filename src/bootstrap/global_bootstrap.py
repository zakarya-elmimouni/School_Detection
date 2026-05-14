# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path
import shutil

# Directory containing the .png images
image_dir = "golden_data/images/test" # change as needed
# Output file path
output_txt = "/golden_data/test.txt" # change as needed

# List all .png files in the folder
png_files = [f for f in os.listdir(image_dir) if f.endswith(".png")]

# Write full relative paths to the output file
with open(output_txt, "w") as f:
    for file_name in png_files:
        relative_path = os.path.join(image_dir, file_name)
        f.write(f"{relative_path}\n")

print(f"{output_txt} file successfully created.")

# =========================
# COMMON CONFIG
# =========================
DATA_YAML = "golden_data/data.yaml"
TEST_TXT = Path("golden_data/test.txt")
BACKUP_TEST_TXT = TEST_TXT.parent / "test_backup.txt"

# Models (update paths if needed)
MODEL_PATHS = {
    "finetuned":   "best_finetuned/weights/best.pt", # change as needed
#    "auto_only":   "auto_labeling/exp/weights/best.pt", # change as needed 
    "manual_only": "manual_only/weights/best.pt", # change as needed
}

N_ITER = 100
SAMPLE_SIZE = None  # default: use full original test set size
RANDOM_SEED = 123

# Ultralytics eval params (same as your script)
IMGSZ = 500
BATCH = 64
MAX_DET = 1
RECT = False
VERBOSE = False

# =========================
# UTILS
# =========================
def out_base_from_pt(pt_path: str) -> Path:
    
    return Path(pt_path).parents[1]

def compute_ci(values, alpha=0.95):
    values = np.asarray(values, dtype=float)
    lo = np.percentile(values, (1 - alpha) / 2 * 100)
    hi = np.percentile(values, (1 + alpha) / 2 * 100)
    return float(lo), float(values.mean()), float(hi)

def write_hist(values, title, xlabel, out_png):
    plt.figure()
    plt.hist(values, bins=20, color='skyblue', edgecolor='black')
    plt.axvline(np.mean(values), color='red', linestyle='--', label='Mean')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def pvalues_for_deltas(deltas):
    """
    deltas: array of ? = finetuned - manual_only over bootstrap iterations
    Returns (p_two_sided, p_one_sided_greater) where:
      - p_two_sided = 2 * min(Pr(?<=0), Pr(?>=0))
      - p_one_sided_greater = Pr(?<=0)  (test H1: ?>0)
    """
    d = np.asarray(deltas, dtype=float)
    n = len(d)
    if n == 0:
        return float("nan"), float("nan")
    p_leq0 = np.mean(d <= 0.0)
    p_geq0 = np.mean(d >= 0.0)
    p_two = 2.0 * min(p_leq0, p_geq0)
    p_two = min(max(p_two, 0.0), 1.0)
    p_one_greater = p_leq0
    return float(p_two), float(p_one_greater)

# =========================
# MAIN
# =========================
# Save the original test.txt
TEST_TXT.parent.mkdir(parents=True, exist_ok=True)
shutil.copy(TEST_TXT, BACKUP_TEST_TXT)

try:
    # Load the original test list (same as your script)
    with open(TEST_TXT, "r") as f:
        original_test_paths = f.read().splitlines()
    if SAMPLE_SIZE is None:
        SAMPLE_SIZE = len(original_test_paths)
    if SAMPLE_SIZE <= 0:
        raise RuntimeError("test.txt is empty or SAMPLE_SIZE <= 0.")

    # Pre-generate bootstrap samples ONCE (PAIRED across all models)
    rng = random.Random(RANDOM_SEED)
    bootstrap_samples = [rng.choices(original_test_paths, k=SAMPLE_SIZE) for _ in range(N_ITER)]

    # Prepare metric containers per model
    results = {}
    for name in MODEL_PATHS.keys():
        results[name] = {
            'map50': [],
            'map50_95': [],
            'precision': [],
            'recall': [],
            'f1': []
        }

    # Evaluate each model SEPARATELY (your workflow),
    # but reusing the same i-th sample for pairing.
    for model_name, MODEL_PATH in MODEL_PATHS.items():
        print("\n" + "="*80)
        print(f"Bootstrap evaluation for model: {model_name} -> {MODEL_PATH}")
        print("="*80)

        # Outputs for THIS model
        base_out = out_base_from_pt(MODEL_PATH)
        OUTPUT_TXT = base_out / f"bootstrap_metrics{N_ITER}.txt"
        PLOT_DIR = base_out / f"bootstrap_plots{N_ITER}"
        PLOT_DIR.mkdir(parents=True, exist_ok=True)

        # Load model
        model = YOLO(MODEL_PATH)

        # Run bootstrap (same logic as your code, using pre-generated samples)
        for i in range(N_ITER):
            print(f"[{model_name}] Bootstrap iteration {i+1}/{N_ITER}")
            sampled_paths = bootstrap_samples[i]

            # Overwrite test.txt with the i-th sample
            with open(TEST_TXT, "w") as f:
                f.write('\n'.join(sampled_paths))

            # Evaluate
            metrics = model.val(
                data=DATA_YAML,
                split="test",
                rect=RECT,
                save=False,
                max_det=MAX_DET,
                imgsz=IMGSZ,
                batch=BATCH,
                verbose=VERBOSE
            )
            box = metrics.box

            # F1
            p, r = float(box.mp), float(box.mr)
            f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

            # Store
            results[model_name]['map50'].append(float(box.map50))
            results[model_name]['map50_95'].append(float(box.map))
            results[model_name]['precision'].append(p)
            results[model_name]['recall'].append(r)
            results[model_name]['f1'].append(float(f1))

        # Save per-model summary (same format as your script)
        with open(OUTPUT_TXT, "w") as f:
            f.write("=== Bootstrap Evaluation Results (YOLO) ===\n")
            f.write(f"Model checkpoint: {MODEL_PATH}\n")
            f.write(f"Iterations: {N_ITER}, Sample size per iteration: {SAMPLE_SIZE}\n\n")
            for metric, values in results[model_name].items():
                ci_low, mean, ci_high = compute_ci(values)
                f.write(f"{metric.upper()} : {mean:.4f} (95% CI: [{ci_low:.4f}, {ci_high:.4f}])\n")

        print(f"? Metrics saved to: {OUTPUT_TXT}")

        # Plots (same as your script)
        for metric, values in results[model_name].items():
            write_hist(
                values,
                title=f"Bootstrap Distribution of {metric.upper()} ({model_name})",
                xlabel=metric.upper(),
                out_png=PLOT_DIR / f"{metric}_hist.png",
            )
        print(f"?? Plots saved to: {PLOT_DIR}")

    # =========================
    # COMPARISON: ? = finetuned - manual_only
    # =========================
    compare_dir = Path("results/bootstrap_yolo_compare_100_regime") / f"paired_{N_ITER}" # change as needed
    compare_dir.mkdir(parents=True, exist_ok=True)
    delta_txt = compare_dir / f"delta_finetuned_minus_manual_only_{N_ITER}.txt"

    metrics_list = ['map50', 'map50_95', 'precision', 'recall', 'f1']
    with open(delta_txt, "w") as f:
        f.write("=== Paired Bootstrap on Deltas (finetuned - manual_only) ===\n")
        f.write(f"Iterations: {N_ITER}, Sample size per iteration: {SAMPLE_SIZE}\n")
        f.write(f"Seed: {RANDOM_SEED}\n\n")
        for metric in metrics_list:
            a = np.asarray(results['finetuned'][metric], dtype=float)
            b = np.asarray(results['manual_only'][metric], dtype=float)
            d = a - b  # paired deltas
            lo, mean, hi = compute_ci(d, 0.95)
            p_two, p_one_gt = pvalues_for_deltas(d)
            f.write(f"{metric.upper()}:\n")
            f.write(f"  mean ?: {mean:.6f}\n")
            f.write(f"  95% CI: [{lo:.6f}, {hi:.6f}]\n")
            f.write(f"  p(two-sided): {p_two:.6f}\n")
            f.write(f"  p(one-sided H1: ?>0): {p_one_gt:.6f}\n\n")

    print(f"?? Delta stats saved to: {delta_txt}")

finally:
    # Restore original test.txt
    if BACKUP_TEST_TXT.exists():
        shutil.copy(BACKUP_TEST_TXT, TEST_TXT)
        print("\nRestored original test.txt.")
