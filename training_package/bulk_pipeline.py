"""
Central training pipeline: prepares data and trains a RandomForest model.
"""
from os import makedirs
from pathlib import Path
from pprint import pprint

from training_package.prepare_training_data import prepare_training_data
from training_package.train_model_from_dataset import train_model
import argparse
medium_config = {
    "feature_limit": 20,
    "max_images": 8,
    "n_estimators": 20,
    "n_components": 25,
    "max_depth": 5,
    "n_jobs": -1,
    "resize_to": (512, 512),
    "skip_existing": False,
    "verbosity": 2,
    "random_seed": 0,
    "model_name": "rf_model_mid"

    }
light_training_config = {
    "feature_limit": 5,
    "max_images": 6,
    "n_estimators": 20,
    "n_components": 10,
    "max_depth": 5,
    "n_jobs": -1,
    "resize_to": (512, 512),
    "skip_existing": False,
    "verbosity": 2,
    "random_seed": 0,
    "model_name": "rf_model_light_2"
}
config_50 = {
    "feature_limit": 50,
    "max_images": None,
    "n_estimators": 100,
    "n_components": 50,
    "max_depth": None,
    "n_jobs": -1,
    "verbosity": 2,
    "random_seed": 42,
    "resize_to": (1024, 1024),
    "channel_index": 0,
    "skip_existing": False,
    "dry_run": False,
    "model_name": "rf_model_50"
}
default_config = {
    "model_name": "rf_model_default",
    
    "feature_limit": 2,
    "max_images": 10,
    "n_estimators": 50,
    "n_components": 10,
    "max_depth": 20,
    "n_jobs": 2,
    "random_seed": 0,
    "verbosity": 2,
    "resize_to": (256, 256),
    "channel_index": 0,
    "skip_existing": False,
    "feature_source": "reduced",
    "keep_output": False,

    "dry_run": False,


    # "test_predict_path" : Path(__file__).parent / "training_data" / "TubeImage.czi",
    "test_predict_path" : None,
}

import argparse
import itertools
from datetime import datetime
from train_model_from_dataset import train_model
def int_or_float(value):
    try:
        return int(value)
    except ValueError:
        return float(value)

def parse_args():
    p = argparse.ArgumentParser(
        description="Grid-search wrapper around train_model_from_dataset.py"
    )
    # allow arbitrary lists on the CLI, e.g. --n-estimators 50 100 200
    p.add_argument(
        "--n_estimators", nargs="+", type=int, default=[50, 100, 200],
        help="Number of trees in the forest"
    )
    p.add_argument(
        "--n_components", nargs="+", type=int, default=[10, 20, 50],
        help="Number of components for GaussianRandomProjection"
    )
    p.add_argument(
        "--max_depth", nargs="+", type=int, default=[10, 20, None],
        help="Maximum depth of each tree"
    )
    p.add_argument(
        "--feature_limit", nargs="+", type=int_or_float, default=[50, 100, None],
        help="How many VGG features to keep"
    )
    p.add_argument(
        "--image_size", nargs="+" ,type=int, default=[256, 512],
        help="Size of resized image feature maps"
    )
    p.add_argument(
        "--max_images", nargs="+", type=int, default=[10, 20, None],
        help="Maximum number of images to process"
    )
    p.add_argument(
        "--n_jobs", type=int, default=[1, 2, -1],
        help="Parallel jobs"
    )
    p.add_argument(
        "--dry_run", action="store_true", default=False,

    )
    p.add_argument(
        "--verbosity", type=int, default=2,
        help="Verbosity level: 0 (silent), 1 (info), 2 (detailed)"
    )
    p.add_argument(
        "--input-dir", "-i", type=Path, default=Path.cwd() / "training_package" / "training_data",
        help="Path to input directory containing images and labels."
    )

    p.add_argument(
        "--output-dir", "-o", type=Path, default=Path.cwd() / "output",
        help="(Optional) Path to output directory for trained model."
    )
    p.add_argument(
        "--model-dir", "-m", type=Path, default=Path.cwd() / "models",
        help="(Optional) Path to output directory for trained model."
    )


    return p.parse_args()

def main():
    args = parse_args()
    dry_run = False

    # build a dict of name→list_of_values
    param_grid = {
        "n_estimators": args.n_estimators,
        "n_components": args.n_components,
        "max_depth": args.max_depth,
        "feature_limit": args.feature_limit,
        "n_jobs": [args.n_jobs],
        "max_images": args.max_images,
        "image_size": args.image_size,
        "resize_to": args.image_size,
        "verbosity": [args.verbosity],
        "dry_run": [args.dry_run],

    }

    # iterate over the Cartesian product of all hyper-parameter lists
    combos = list(itertools.product(*param_grid.values()))
    total = len(combos)
    print(f"Starting {total} runs")
    for idx, combo in enumerate(combos, start=1):
        # map back to parameter names
        config = dict(zip(param_grid.keys(), combo))

        # give each run a unique folder/name
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # e.g. rf_ne50_md20_fl100_nj2_20250519_235501
        config["model_name"] = (
            f"rf_ne{config['n_estimators']}"
            f"_md{config['n_components']}"
            f"_md{config['max_depth']}"
            f"_fl{config['feature_limit']}"
            f"_im{config['max_images']}"
            f"x{config['resize_to']}"
            f"_nj{config['n_jobs']}"
            f"_{stamp}"
        )
        config["resize_to"] = (config["resize_to"],config["resize_to"])

        config["input_dir"] = args.input_dir
        config["models_dir"] = args.model_dir
        config["model_folder"] = args.model_dir / config["model_name"]
        config["output_dir"] = args.output_dir / config["model_name"]

        config["models_dir"].mkdir(parents=True, exist_ok=True)
        config["model_folder"].mkdir(parents=False, exist_ok=True)
        config["output_dir"].mkdir(parents=True, exist_ok=True)

        run_config = {**default_config, **config}
        print(f"[{idx}/{total}] starting run: Config:")
        pprint(run_config)
        if dry_run:
            print("Dry run bulk_pipeline: skipping training")
            continue
        prepare_training_data(run_config)
        train_model(run_config)

if __name__ == "__main__":
    main()
