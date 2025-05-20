"""
Central training pipeline: prepares data and trains a RandomForest model.
"""
from os import makedirs
from pathlib import Path
from pprint import pprint
import random

from training_package.prepare_training_data import prepare_training_data
from training_package.train_model_from_dataset import train_model
from training_package.evaluate_model import evaluate_model
import argparse
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

    p.add_argument(
        "--n_estimators", nargs="+", type=int, default=[50, 100, 200],
        help="Number of trees in the forest"
    ) # The number of trees in the forest
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
        "--max_images", type=int, default=10,
        help="Maximum number of images to process"
    )
    p.add_argument(
        "--n_jobs", type=int, default=[1, 2, -1],
        help="Parallel jobs"
    )
    p.add_argument(
        "--random_seed", type=int, default=0,
    )
    p.add_argument(
        "--dry_run", action="store_true", default=False,

    )
    p.add_argument(
        "--verbosity", type=int, default=2,
        help="Verbosity level: 0 (silent), 1 (info), 2 (detailed)"
    )
    p.add_argument(
        "--input_dir", "-i", type=Path, default=Path.cwd() / "training_package" / "training_data",
        help="Path to input directory containing images and labels."
    )
    p.add_argument(
        "--output_dir", "-o", type=Path, default=Path.cwd() / "output",
        help="(Optional) Path to output directory for trained model."
    )
    p.add_argument(
        "--model_dir", "-m", type=Path, default=Path.cwd() / "models",
        help="(Optional) Path to output directory for trained model."
    )
    p.add_argument(
        "--test_dir", "-t", type=Path, default=None,
        help="(Optional) If set, runs an evaluation from the test dataset."
    )


    return p.parse_args()

def make_image_proc_name(ip_config):
    # Example: img256_nc20
    return f"img{ip_config['resize_to']}_nc{ip_config['n_components']}"

def make_train_name(tr_config):
    # Example: ne100_md20_fl50
    return f"ne{tr_config['n_estimators']}_md{tr_config['max_depth']}_fl{tr_config['feature_limit']}"

def main():
    args = parse_args()
    dry_run = False

    resize_to = (args.image_size, args.image_size)

    image_processing_iter_params = {
        "resize_to": resize_to,
        "n_components": args.n_components,
    }
    image_processing_params = {
        "max_images": args.max_images,
        "input_dir": args.input_dir,
        "verbosity": args.verbosity,
        "dry_run": args.dry_run,
        "random_seed": args.random_seed,
    }

    training_iter_params = {
        "feature_limit": args.feature_limit,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
    }
    training_params = {
        "verbosity": args.verbosity,
        "dry_run": args.dry_run,
        "n_jobs": args.n_jobs,
        "random_seed": args.random_seed,
    }

    output_dir = args.output_dir
    model_dir = args.model_dir

    image_processing_combos = list(itertools.product(*image_processing_iter_params.values()))
    training_combos = list(itertools.product(*training_iter_params.values()))

    total = len(image_processing_combos) * len(training_combos)
    idx = 0
    for ip_combo in image_processing_combos:
        ip_config = dict(zip(image_processing_iter_params.keys(), ip_combo))
        img_folder_name = make_image_proc_name(ip_config)
        processed_dir = output_dir / img_folder_name
        export_dir = model_dir / img_folder_name

        if image_processing_params["random_seed"] == 0:
            if image_processing_params["random_seed"] == 0:
                seed = random.randint(1, 1000)
                image_processing_params["random_seed"] = seed

        # Prepare data
        prep_config = {
            **image_processing_params,
            **ip_config,
            "output_dir": processed_dir,
            "export_dir": export_dir,
        }
        if dry_run:
            print(f"[DRY RUN] Would prepare data in {processed_dir} with {ip_config}")
        else:
            prepare_training_data(prep_config)

        for tr_combo in training_combos:
            tr_config = dict(zip(training_iter_params.keys(), tr_combo))
            train_folder_name = make_train_name(tr_config)
            model_name = f"{img_folder_name}_{train_folder_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            if training_params["random_seed"] == 0:
                seed = random.randint(1, 1000)
                training_params["random_seed"] = seed

            run_config = {
                **default_config,
                **prep_config,
                **training_params,
                **tr_config,
                "model_name": model_name,
                "output_dir": processed_dir,
                "models_dir": export_dir,
            }
            idx += 1
            if dry_run:
                print(f"[{idx}/{total}] [DRY RUN] Would train model {model_name} using data from {processed_dir} with {tr_config}")
            else:
                train_model(run_config)
                if args.test_dir:
                    evaluate_model(run_config)
if __name__ == "__main__":
    main()
