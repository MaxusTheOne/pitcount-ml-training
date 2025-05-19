"""
Central training pipeline: prepares data and trains a RandomForest model.
"""
from os import makedirs
from pathlib import Path
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
    "random_seed": 42,
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
        "--max-depth", nargs="+", type=int, default=[10, 20, None],
        help="Maximum depth of each tree"
    )
    p.add_argument(
        "--feature-limit", nargs="+", type=int, default=[50, 100, None],
        help="How many VGG features to keep"
    )
    p.add_argument(
        "--n-jobs", nargs="+", type=int, default=[1, 2, -1],
        help="Parallel jobs"
    )
    p.add_argument(
        "--output-dir", type=str, default=None,
        help="(Optional) override the DEFAULT out_dir in your script"
    )
    return p.parse_args()

def main():
    args = parse_args()

    # build a dict of name→list_of_values
    param_grid = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "feature_limit": args.feature_limit,
        "n_jobs": args.n_jobs,
    }

    # iterate over the Cartesian product of all hyper-parameter lists
    combos = list(itertools.product(*param_grid.values()))
    total = len(combos)

    for idx, combo in enumerate(combos, start=1):
        # map back to parameter names
        config = dict(zip(param_grid.keys(), combo))

        # give each run a unique folder/name
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # e.g. rf_ne50_md20_fl100_nj2_20250519_235501
        config["model_name"] = (
            f"rf_ne{config['n_estimators']}"
            f"_md{config['max_depth']}"
            f"_fl{config['feature_limit']}"
            f"_nj{config['n_jobs']}"
            f"_{stamp}"
        )
        if args.output_dir:
            config["output_dir"] = args.output_dir

        print(f"[{idx}/{total}] starting run: {config['model_name']}")
        train_model(config)

if __name__ == "__main__":
    main()
