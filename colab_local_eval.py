"""
yolo_local_eval.py

This module provides a local (non-Colab) workflow for:

1. (Optionally) training a YOLOv8 model using the Ultralytics API.
2. Evaluating image-level class accuracy on a validation set, where an
   image is considered correct if all ground-truth classes appear in
   the model's predictions (bounding box locations are ignored).
3. Reporting per-class accuracy statistics.

It is written to be compatible with pydoc, so all executable logic is
contained in functions and guarded under the usual
`if __name__ == "__main__":` block.
"""

from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, Tuple

import numpy as np
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Class names for the YOLO model, indexed by numerical class ID.
CLASS_NAMES = ["Dalek", "cats", "dog", "person", "Lightsaber", "Red Lightsaber"]

#: Path to your YOLO data configuration (data.yaml) for training/validation.
#: Update this path to match your local filesystem.
DATA_CONFIG = Path("all_images/data.yaml")

#: Directory containing validation images for custom evaluation.
#: Update this path to match your local filesystem.
VAL_IMAGES_DIR = Path("all_images/images/val")

#: Directory containing YOLO-format label files for the validation images.
#: Each image file `name.jpg`/`name.png` should have a corresponding
#: `name.txt` label file in this folder.
VAL_LABELS_DIR = Path("all_images/labels/val")

#: Path to the trained model weights to be used for evaluation.
#: This should be a .pt file produced by Ultralytics YOLO training.
TRAINED_MODEL_WEIGHTS = Path("all_images_training/run_2/weights/best.pt")

#: Number of epochs to train when using `train_model()` (optional).
DEFAULT_EPOCHS = 100


# ---------------------------------------------------------------------------
# Colab-specific code (COMMENTED OUT for local use)
# ---------------------------------------------------------------------------

# The following code is useful when running in Google Colab, but will not work
# in a standard local Python environment. It is left here as a reference and
# is commented out so that pydoc and local imports will not fail.
#
# In Colab, you might have used:
#
# !pip install ultralytics
# !yolo settings sync=False
#
# from IPython import display
# display.clear_output()
#
# import ultralytics
# ultralytics.checks()
#
# from google.colab import drive
# drive.mount('/content/drive')
#
# import kagglehub
# path = kagglehub.dataset_download("shaunthesheep/microsoft-catsvsdogs-dataset")
# print("Path to dataset files:", path)
#
# All of this is commented out because:
# - `!` shell commands are not valid Python syntax in a normal script.
# - `google.colab` is not available outside of Colab.
# - `kagglehub` may not be installed locally.
#
# Instead, install packages via:
#   pip install ultralytics numpy
# and place your data on your local filesystem.


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def read_gt_classes(label_path: Path) -> Set[int]:
    """
    Read ground-truth classes from a YOLO-format label file.

    The expected format for each line in the label file is:
        class_id x_center y_center width height

    Only the `class_id` (the first value on each line) is used here.

    Parameters
    ----------
    label_path : Path
        Path to the YOLO label file (.txt).

    Returns
    -------
    Set[int]
        A set of integer class IDs present in this annotation file.
        Returns an empty set if the file does not exist or contains no labels.
    """
    if not label_path.exists():
        return set()

    classes: Set[int] = set()
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            # The first value is the class ID
            cls_id = int(float(parts[0]))
            classes.add(cls_id)

    return classes


def evaluate_image_level_accuracy(
    model: YOLO,
    images_dir: Path,
    labels_dir: Path,
    class_names: list,
    conf_threshold: float = 0.25,
) -> Tuple[float, Dict[int, float]]:
    """
    Evaluate image-level class accuracy over a directory of images.

    For each image, this function:
      1. Loads the ground-truth classes from the YOLO label file.
      2. Uses the model to predict bounding boxes for the image.
      3. Extracts the unique set of predicted class IDs.
      4. Counts the image as correct if *all* ground-truth classes are present
         in the predicted set (bounding box locations are ignored).

    Additionally, it tracks per-class accuracy: for each class,
    "correct" means the class was present in the ground truth for an image
    and also appeared at least once in the model's predictions.

    Parameters
    ----------
    model : YOLO
        A loaded Ultralytics YOLO model ready to perform `.predict()`.
    images_dir : Path
        Directory containing the validation images (jpg/png/jpeg).
    labels_dir : Path
        Directory containing YOLO-format annotation files matching
        the images in `images_dir`.
    class_names : list
        List of class names, indexed by numeric class ID.
    conf_threshold : float, optional
        Confidence threshold for predictions, by default 0.25.

    Returns
    -------
    Tuple[float, Dict[int, float]]
        A tuple `(overall_accuracy, per_class_accuracy)` where:

        - `overall_accuracy` is the fraction of images for which
          all ground-truth classes appeared in the predictions.
        - `per_class_accuracy` is a dictionary mapping class ID to
          the probability that the class is predicted at least once,
          given that it appears in the ground truth for that image.
    """
    # Collect all image paths in the directory
    image_paths = sorted(
        list(images_dir.glob("*.jpg"))
        + list(images_dir.glob("*.png"))
        + list(images_dir.glob("*.jpeg"))
    )

    total_images = 0
    correct_images = 0

    # Per-class statistics
    class_totals: Dict[int, int] = defaultdict(int)
    class_correct: Dict[int, int] = defaultdict(int)

    for img_path in image_paths:
        label_path = labels_dir / (img_path.stem + ".txt")
        gt_classes = read_gt_classes(label_path)

        # Skip images that have no ground-truth labels
        if len(gt_classes) == 0:
            continue

        # Run the model on this single image
        results = model.predict(
            source=str(img_path),
            conf=conf_threshold,
            verbose=False,
        )

        r = results[0]

        # Extract predicted class IDs (if any)
        if r.boxes is not None and len(r.boxes) > 0:
            pred_classes = set(r.boxes.cls.cpu().numpy().astype(int).tolist())
        else:
            pred_classes = set()

        # Overall image-level accuracy
        total_images += 1
        if gt_classes.issubset(pred_classes):
            correct_images += 1

        # Per-class statistics: if class appears in GT, did we predict it?
        for cls_id in gt_classes:
            class_totals[cls_id] += 1
            if cls_id in pred_classes:
                class_correct[cls_id] += 1

    # Compute final overall accuracy
    overall_accuracy = (
        correct_images / total_images if total_images > 0 else 0.0
    )

    # Compute per-class accuracy
    per_class_accuracy: Dict[int, float] = {}
    for cls_id, total in class_totals.items():
        if total > 0:
            per_class_accuracy[cls_id] = class_correct[cls_id] / total
        else:
            per_class_accuracy[cls_id] = 0.0

    # Print a summary to the console
    print(f"Images evaluated: {total_images}")
    print(f"Images with all GT classes predicted: {correct_images}")
    print(f"Image-level class accuracy: {overall_accuracy * 100:.2f}%")

    print("\nPer-class accuracy (GT present → predicted at least once):\n")
    for cls_id in sorted(class_totals.keys()):
        total = class_totals[cls_id]
        correct = class_correct[cls_id]
        acc = per_class_accuracy[cls_id]
        name = class_names[cls_id] if cls_id < len(class_names) else f"cls_{cls_id}"
        print(f"{cls_id} ({name}): {correct}/{total} = {acc * 100:.2f}%")

    return overall_accuracy, per_class_accuracy


def train_model(
    data_config: Path,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = 16,
    project_dir: Path = Path("all_images_training"),
    run_name: str = "run_2",
    device: str = "cpu",
) -> YOLO:
    """
    Train a YOLOv8 model using the Ultralytics API.

    This function is optional and provided for completeness. In many
    workflows, you may choose to train in Colab or elsewhere, and only
    load the resulting `.pt` weights locally.

    Parameters
    ----------
    data_config : Path
        Path to the YOLO data configuration file (data.yaml).
    epochs : int, optional
        Number of training epochs, by default DEFAULT_EPOCHS.
    batch_size : int, optional
        Training batch size, by default 16.
    project_dir : Path, optional
        Directory where Ultralytics will store run artifacts, by default
        "all_images_training".
    run_name : str, optional
        Subdirectory name for this training run, by default "run_2".
    device : str, optional
        Device to train on ("cpu" or "cuda"), by default "cpu".

    Returns
    -------
    YOLO
        The trained YOLO model instance.
    """
    # Load a base YOLO model (e.g., YOLOv8 nano variant)
    model = YOLO("yolov8n.pt")

    results = model.train(
        data=str(data_config),
        epochs=epochs,
        batch=batch_size,
        save_period=5,
        device=device,
        cls=2.0,
        project=str(project_dir),
        name=run_name,
        save=True,
        exist_ok=True,
    )

    # Evaluate on the validation set defined in the data config
    metrics = model.val()
    print(f"mAP@0.5: {metrics.box.map}")

    return model


def run_example_predictions(model: YOLO, source: Path) -> None:
    """
    Run example predictions on a directory or a single image
    and display the results.

    Parameters
    ----------
    model : YOLO
        A loaded YOLO model ready for `.predict()`.
    source : Path
        Path to either a directory of images or a single image file.
    """
    results = model.predict(
        source=str(source),
        iou=0.3,
        conf=0.35,
    )

    # Note: In a local environment, `r.show()` will open image windows.
    # This behavior depends on your OS and display backend.
    for r in results:
        r.show()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Example end-to-end workflow for local evaluation.

    This function demonstrates:

    1. Loading a trained YOLO model from disk.
    2. Evaluating image-level and per-class accuracy.
    3. (Optionally) Running example predictions on the validation set.
    """
    # If you want to train locally, uncomment the following and
    # comment out the direct loading of TRAINED_MODEL_WEIGHTS:
    #
    # model = train_model(
    #     data_config=DATA_CONFIG,
    #     epochs=DEFAULT_EPOCHS,
    #     batch_size=16,
    #     project_dir=Path("all_images_training"),
    #     run_name="run_2",
    #     device="cuda",  # or "cpu"
    # )

    # Load an already-trained YOLO model from weights file
    model = YOLO(str(TRAINED_MODEL_WEIGHTS))

    # Evaluate image-level class accuracy
    evaluate_image_level_accuracy(
        model=model,
        images_dir=VAL_IMAGES_DIR,
        labels_dir=VAL_LABELS_DIR,
        class_names=CLASS_NAMES,
        conf_threshold=0.25,
    )

    # Optionally, run predictions and display them
    # run_example_predictions(model, VAL_IMAGES_DIR)


if __name__ == "__main__":
    main()
