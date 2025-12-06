"""
modelTest.py

Evaluate a trained YOLO model on a validation dataset and compute:

1. Image-level accuracy:
   - An image is counted as correct if *all* ground-truth classes
     in that image appear in the model's predictions
     (bounding box positions are ignored).

2. Per-class accuracy:
   - For each class, we count how often it was predicted at least once
     in images where it appears in the ground truth.

If any class has an accuracy below a specified threshold, the script
exits with status code 1. Otherwise, it exits with status code 0.

This script is structured and documented for use with pydoc.
"""

from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, List, Tuple
import sys

from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Directory containing validation images.
IMAGES_DIR = Path("images/val")

#: Directory containing YOLO-format label files (.txt) for the validation images.
LABELS_DIR = Path("labels/val")

#: Minimum allowed per-class accuracy (0.0 - 1.0). If any class falls below
#: this value, the script will exit with status code 1.
ACCURACY_THRESHOLD = 0.75

#: YOLO model weights file to evaluate.
MODEL_WEIGHTS = Path("best.pt")

#: Class names for the dataset, indexed by numerical class ID.
CLASS_NAMES: List[str] = [
    "Dalek",
    "cats",
    "dog",
    "person",
    "Lightsaber",
    "Red Lightsaber",
]


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def read_gt_classes(label_path: Path) -> Set[int]:
    """
    Read ground-truth class IDs from a YOLO label file.

    Each line in a YOLO label file has the format:
        class_id x_center y_center width height

    Only the `class_id` (the first value in each line) is used here.

    Parameters
    ----------
    label_path : Path
        Path to a YOLO-format label file (.txt).

    Returns
    -------
    Set[int]
        A set of integer class IDs present in this annotation file.
        If the file does not exist, an empty set is returned.
    """
    if not label_path.exists():
        return set()

    classes: Set[int] = set()

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cls_id = int(float(parts[0]))
            classes.add(cls_id)

    return classes


def evaluate_model_accuracy(
    model: YOLO,
    images_dir: Path,
    labels_dir: Path,
    class_names: List[str],
    threshold: float,
) -> Tuple[bool, float, Dict[int, float]]:
    """
    Evaluate image-level and per-class accuracy for a YOLO model.

    An image is considered correct if all of its ground-truth classes
    are found in the model's predicted classes (location is ignored).

    Per-class accuracy is computed as:
        (# of images where class is GT and predicted at least once)
        /
        (# of images where class appears in GT)

    If any class has accuracy below `threshold`, the overall result is
    considered a failure.

    Parameters
    ----------
    model : YOLO
        A loaded YOLO model ready to perform predictions.
    images_dir : Path
        Directory containing validation images (jpg/png/jpeg).
    labels_dir : Path
        Directory containing YOLO-format annotation files.
    class_names : list of str
        List of class names corresponding to numeric class IDs.
    threshold : float
        Minimum allowed per-class accuracy (0.0 - 1.0).

    Returns
    -------
    Tuple[bool, float, Dict[int, float]]
        A tuple `(success, overall_accuracy, per_class_accuracy)` where:

        - `success` is True if all classes meet or exceed the threshold,
          False otherwise.
        - `overall_accuracy` is the fraction of images for which all
          ground-truth classes appear in the predictions.
        - `per_class_accuracy` is a dictionary mapping class ID to the
          per-class accuracy value (0.0 - 1.0).
    """
    # Collect all images in the directory
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

        # Skip unlabeled images
        if not gt_classes:
            continue

        # Run YOLO prediction on this image
        results = model.predict(
            source=str(img_path),
            conf=0.25,
            verbose=False,
        )
        r = results[0]

        # Extract predicted class IDs
        if r.boxes is not None and len(r.boxes) > 0:
            pred_classes = set(r.boxes.cls.cpu().numpy().astype(int).tolist())
        else:
            pred_classes = set()

        total_images += 1

        # Image-level accuracy: GT classes must be a subset of predicted classes
        if gt_classes.issubset(pred_classes):
            correct_images += 1

        # Per-class stats
        for cls_id in gt_classes:
            class_totals[cls_id] += 1
            if cls_id in pred_classes:
                class_correct[cls_id] += 1

    # Compute overall image-level accuracy
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

    # Print summary
    print(f"Images evaluated: {total_images}")
    print(f"Images with exactly correct class set: {correct_images}")
    print(f"Image-level class accuracy: {overall_accuracy * 100:.2f}%")

    print("\nPer-class accuracy (GT present → predicted at least once):\n")

    success = True
    for cls_id in sorted(class_totals.keys()):
        total = class_totals[cls_id]
        correct = class_correct[cls_id]
        acc = per_class_accuracy[cls_id]
        name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"

        print(f"{cls_id} ({name}): {correct}/{total} = {acc * 100:.2f}%")

        if acc < threshold:
            success = False
            print(
                f"Class {cls_id} ({name}) accuracy {acc * 100:.2f}% "
                f"fails threshold of {threshold * 100:.2f}%"
            )

    return success, overall_accuracy, per_class_accuracy


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Main function to load the model, run the evaluation, and set exit code.

    This function:
    1. Loads the YOLO model from `MODEL_WEIGHTS`.
    2. Evaluates image-level and per-class accuracy on the dataset
       located under `IMAGES_DIR` and `LABELS_DIR`.
    3. Exits with code 0 if all classes meet the accuracy threshold
       defined by `ACCURACY_THRESHOLD`, or 1 otherwise.
    """
    model = YOLO(str(MODEL_WEIGHTS))

    success, _, _ = evaluate_model_accuracy(
        model=model,
        images_dir=IMAGES_DIR,
        labels_dir=LABELS_DIR,
        class_names=CLASS_NAMES,
        threshold=ACCURACY_THRESHOLD,
    )

    # Exit code reflects pass/fail status
    if success:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
