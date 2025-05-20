import os
import logging
from pathlib import Path
import numpy as np
from joblib import load
from imageio import imread
import cv2
import json
from training_package.transformers import czi_to_fmap

try:
    import czifile  # for reading .czi image files
except ImportError:
    raise ImportError("Please install the 'czifile' package to read .czi files")

try:
    from skimage.measure import label
except ImportError:
    raise ImportError("Please install 'scikit-image' for connected component analysis")


def evaluate_model(config: dict) -> None:
    """
    Evaluate a trained model on a set of test images and save evaluation statistics.

    Args:
        config (dict): Configuration dictionary with keys:
            - models_dir (str or Path): Base directory containing all model folders.
            - model_folder (str): Sub-directory under models_dir for the specific model.
            - model_name (str): Filename (without extension) of the saved model (joblib format).
            - output_dir (str or Path): Directory where evaluation stats will be saved.
            - test_data (dict): Dictionary with keys 'images' and 'labels' pointing to test image and label directories.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    models_dir = config['models_dir']
    model_folder = models_dir / config['model_folder']
    model_path = model_folder / f"{config['model_name']}.joblib"
    transformer_path = model_folder / "transformer.joblib"
    output_dir = config['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = config['test_dir']
    images_dir = data_dir / "images"
    labels_dir = data_dir / 'labels'

    if config["dry_run"]:
        print("Dry run: skipping actual evaluation.")
        return

    # Load model and transformer
    model = load(model_path)
    logging.info(f"Loaded model from {model_path}")
    transformer = load(transformer_path)
    logging.info(f"Loaded transformer from {transformer_path}")

    # Metrics
    ious, dices, count_errors = [], [], []
    RESIZE_TO = config["resize_to"]

    for filename in os.listdir(images_dir):
        if not filename.lower().endswith((".czi", ".png", ".tif", ".tiff")):
            continue

        image_path = images_dir / filename
        base_name = image_path.stem
        # find corresponding label
        label_path = None
        for ext in [".tif", ".tiff", ".png", ".jpg", ".czi"]:
            candidate = labels_dir / f"{base_name}{ext}"
            if candidate.exists():
                label_path = candidate
                break
        if label_path is None:
            logging.error(f"No label found for {filename}, skipping")
            continue

        # feature map and raw image
        feat_map, image_npy = czi_to_fmap(czi_path=str(image_path), size=RESIZE_TO)
        H, W, _ = feat_map.shape
        flat = feat_map.reshape(-1, feat_map.shape[-1])
        trans_flat = transformer.transform(flat)
        pred_flat = model.predict(trans_flat)
        pred_map = pred_flat.reshape(H, W)

        # resize prediction back to original size if needed
        if (H, W) != image_npy.shape[:2]:
            pred_mask = cv2.resize(pred_map.astype(np.uint8),
                                   (image_npy.shape[1], image_npy.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
        else:
            pred_mask = pred_map.astype(np.uint8)

        # load GT mask
        if label_path.suffix.lower() == '.czi':
            gt_array = czifile.imread(str(label_path))
            gt_array = np.squeeze(gt_array)
        else:
            gt_array = imread(str(label_path))
        gt_mask = (gt_array > 0).astype(np.uint8)
        if gt_mask.ndim > 2:
            gt_mask = gt_mask[..., 0]

        # compute metrics
        intersection = np.logical_and(pred_mask == 1, gt_mask == 1).sum()
        union = np.logical_or(pred_mask == 1, gt_mask == 1).sum()
        iou = 1.0 if union == 0 else intersection / union
        pred_sum, gt_sum = pred_mask.sum(), gt_mask.sum()
        dice = 1.0 if (pred_sum + gt_sum) == 0 else (2 * intersection) / (pred_sum + gt_sum)

        # count components
        pred_labeled, pred_count = label(pred_mask, connectivity=2, return_num=True)
        gt_labeled, gt_count = label(gt_mask, connectivity=2, return_num=True)
        count_error = pred_count - gt_count

        logging.debug(f"{filename}: IoU={iou:.4f}, Dice={dice:.4f}, "
                      f"TrueCount={gt_count}, PredCount={pred_count}, CountError={count_error}")

        ious.append(iou)
        dices.append(dice)
        count_errors.append(count_error)

    if not ious:
        logging.error("No images processed, please check paths.")
        return

    # summarize
    mean_iou = float(np.mean(ious))
    mean_dice = float(np.mean(dices))
    mae_count = float(np.mean(np.abs(count_errors)))
    bias_count = float(np.mean(count_errors))

    logging.info(f"Mean IoU: {mean_iou:.4f}")
    logging.info(f"Mean Dice: {mean_dice:.4f}")
    logging.info(f"MAE Count: {mae_count:.2f}")
    logging.info(f"Bias Count: {bias_count:.2f}")

    # save stats
    stats = {
        "model": f"{config['model_name']}",
        "mean_iou": mean_iou,
        "mean_dice": mean_dice,
        "mae_count": mae_count,
        "bias_count": bias_count
    }
    stats_path = Path(output_dir) / f"{config['model_name']}_evaluation_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=4)
    logging.info(f"Saved evaluation stats to {stats_path}")

