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
    "random_seed": 0,
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
def int_float_sqrt_or_log2(value):
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            if value == "sqrt" or value == "log2":
                return value
            return None
def int_or_none(value):
    try:
        return int(value)
    except ValueError:
        return None

def parse_args():
    p = argparse.ArgumentParser(
        description="Grid-search wrapper around train_model_from_dataset.py"
    )

    p.add_argument(
        "--n_estimators", nargs="+", type=int, default=[10, 20, 30],
        help="Number of trees in the forest"
    ) # The number of trees in the forest
    p.add_argument(
        "--n_components", nargs="+", type=int, default=[10, 20, 50],
        help="Number of components for GaussianRandomProjection"
    )
    p.add_argument(
        "--max_depth", nargs="+", type=int_or_none, default=[10, 20, None],
        help="Maximum depth of each tree"
    )
    p.add_argument(
        "--feature_limit", nargs="+", type=int_float_sqrt_or_log2, default=[50, 100, None],
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
        "--verbosity", type=int, default=1,
        help="Verbosity level: 0 (silent), 1 (info), 2 (detailed)"
    )
    p.add_argument(
        "--skip_existing", action="store_true", default=False,
    )
    p.add_argument(
        "--input_dir", "-i", type=Path, default=Path.cwd() / "training_package" / "training_data",
        help="Path to input directory containing images and labels."
    )
    p.add_argument(
        "--output_dir", "--models_dir", "-o", type=Path, default=Path.cwd() / "output",
        help="(Optional) Path to output directory for data and models."
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


    image_processing_iter_params = {
        "resize_to": args.image_size,
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

    verbosity = args.verbosity
    skip_existing = args.skip_existing

    model_dir = args.output_dir

    image_processing_combos = list(itertools.product(*image_processing_iter_params.values()))
    training_combos = list(itertools.product(*training_iter_params.values()))

    total_ic = len(image_processing_combos)
    total_tc = len(training_combos)
    total = total_ic * total_tc

    print(f"Launching Bulk Training, processing {total_ic} VGG19 feature sets \n         with {total_tc} RFC each \n             Totaling {total} fits")

    idx = 0
    for ip_combo in image_processing_combos:
        idx += 1
        ip_config = dict(zip(image_processing_iter_params.keys(), ip_combo))
        img_folder_name = make_image_proc_name(ip_config)
        data_folder = model_dir / img_folder_name
        data_folder.mkdir(parents=True, exist_ok=True)
        if verbosity > 0:
            print(f"[{idx}/{total_ic}] dataset: {img_folder_name}")

        ip_config["resize_to"] = (ip_config["resize_to"], ip_config["resize_to"])




        if image_processing_params["random_seed"] == 0:
            if image_processing_params["random_seed"] == 0:
                seed = random.randint(1, 1000)
                image_processing_params["random_seed"] = seed

        # Prepare data
        prep_config = {
            **image_processing_params,
            **ip_config,
            "data_folder": data_folder,
        }
        if skip_existing and (data_folder / "transformer.joblib").exists():
            print(f"    Skipping existing feature set {img_folder_name}")
        else:
            if dry_run:
                print(f"    [DRY RUN] Would prepare data in {data_folder} with {ip_config}")
            else:
                prepare_training_data(prep_config)
        ydx = 0
        for tr_combo in training_combos:
            ydx += 1
            tr_config = dict(zip(training_iter_params.keys(), tr_combo))
            train_folder_name = make_train_name(tr_config)
            if verbosity > 0:
                print(f"[{idx}/{total_ic}][{ydx}/{total_tc}] model: {train_folder_name}")
            model_name = f"{img_folder_name}_{train_folder_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            model_folder = data_folder / train_folder_name
            model_file_path = model_folder / (model_name + ".joblib")
            model_folder.mkdir(parents=True, exist_ok=True)

            if skip_existing and model_file_path.exists():
                print(f"    Skipping existing model {model_name}")
                continue

            run_config = {
                **default_config,
                **prep_config,
                **training_params,
                **tr_config,
                "model_name": model_name,
                "model_name_short": train_folder_name,

                "model_folder": model_folder,
                "model_file_path": model_file_path,
            }
            
            if dry_run:
                print(f"    [DRY RUN] Would train model {model_name} using data from {data_folder} with {tr_config}")
            else:
                train_model(run_config)
                if args.test_dir:
                    evaluate_model({**run_config, "test_dir": args.test_dir})
if __name__ == "__main__":
    main()
