import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Base directory where models are stored
BASE_DIR = Path(r"C:\Users\marku\OneDrive - Københavns Erhvervsakademi\Desktop\main_proj\models_bulk")
PLOT_DIR = Path("plots")
PLOT_DIR.mkdir(exist_ok=True)

# Regex to extract parameters from folder names
img_proc_pattern = re.compile(r"img(?P<resize>\d+)_nc(?P<n_components>\d+)")
model_pattern = re.compile(r"ne(?P<n_estimators>\d+)_md(?P<max_depth>\d+)_fl(?P<feature_limit>[\d.]+)")

# Gather evaluation data
records = []

for img_proc_dir in BASE_DIR.iterdir():
    if not img_proc_dir.is_dir():
        continue
    img_match = img_proc_pattern.match(img_proc_dir.name)
    if not img_match:
        continue
    resize_to = int(img_match.group("resize"))
    n_components = int(img_match.group("n_components"))

    for model_dir in img_proc_dir.iterdir():
        if not model_dir.is_dir():
            continue
        model_match = model_pattern.match(model_dir.name)
        if not model_match:
            continue
        n_estimators = int(model_match.group("n_estimators"))
        max_depth = int(model_match.group("max_depth"))
        feature_limit = float(model_match.group("feature_limit"))

        eval_file = model_dir / f"{model_dir.name}_evaluation_stats.json"
        if not eval_file.exists():
            continue

        with open(eval_file, "r") as f:
            data = json.load(f)
            records.append({
                "img_proc": img_proc_dir.name,
                "resize_to": resize_to,
                "n_components": n_components,
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "feature_limit": feature_limit,
                "mean_iou": data.get("mean_iou", None)
            })

# Create DataFrame
df = pd.DataFrame(records)
print("Loaded evaluations:\n", df.head())

# Visualizations
sns.set(style="whitegrid")
for param in ["n_estimators", "max_depth", "feature_limit"]:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x=param, y="mean_iou", hue="img_proc")
    plt.title(f"Mean IoU by {param}")
    plt.ylabel("Mean IoU")
    plt.xlabel(param)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plot_filename = PLOT_DIR / f"iou_vs_{param}.png"
    plt.savefig(plot_filename)
    print(f"Saved plot to {plot_filename}")
    plt.close()

