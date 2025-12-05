from pathlib import Path
from collections import defaultdict
import ultralytics
from ultralytics import YOLO

success = True
images_dir = Path("images/val")
labels_dir = Path("labels/val")
threshold = 0.75  # minimum per-class accuracy
image_paths = sorted(list(images_dir.glob("*.jpg")) +
                     list(images_dir.glob("*.png")) +
                     list(images_dir.glob("*.jpeg")))

trained_model = YOLO("best.pt")
total_images = 0
correct_images = 0

# Your class names:
class_names = ['Dalek', 'cats', 'dog', 'person', 'Lightsaber', 'Red Lightsaber']

# Per-class stats
class_totals = defaultdict(int)   # how many images contain this class
class_correct = defaultdict(int)  # how many of those images we predicted this class in

def read_gt_classes(label_path: Path):
    """Reads a YOLO txt label file and returns a set of class IDs."""
    if not label_path.exists():
        return set()
    classes = set()
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cls_id = int(float(parts[0]))
            classes.add(cls_id)
    return classes

for img_path in image_paths:
    label_path = labels_dir / (img_path.stem + ".txt")
    gt_classes = read_gt_classes(label_path)

    if len(gt_classes) == 0:
        continue  # skip images with no labels

    results = trained_model.predict(
        source=str(img_path),
        conf=0.25,
        verbose=False
    )
    r = results[0]
    # predicted classes
    if r.boxes is not None and len(r.boxes) > 0:
        pred_classes = set(r.boxes.cls.cpu().numpy().astype(int).tolist())
    else:
        pred_classes = set()

    total_images += 1
    if gt_classes.issubset(pred_classes):
        correct_images += 1

    # per-class metrics
    for cls_id in gt_classes:
        class_totals[cls_id] += 1
        if cls_id in pred_classes:
            class_correct[cls_id] += 1
# Overall accuracy
accuracy = correct_images / total_images if total_images > 0 else 0.0

print(f"Images evaluated: {total_images}")
print(f"Images with exactly correct class set: {correct_images}")
print(f"Image-level class accuracy: {accuracy * 100:.2f}%")

# Per-class accuracy
print("\nPer-class accuracy (GT present → predicted at least once):\n")
for cls_id in sorted(class_totals.keys()):
    total = class_totals[cls_id]
    correct = class_correct[cls_id]
    acc = correct / total if total > 0 else 0.0
    print(f"{cls_id} ({class_names[cls_id]}): {correct}/{total} = {acc*100:.2f}%")
    if acc < threshold:
        success = False
        print(f"Class {cls_id} accuracy {acc*100:.2f}% fails threshold of {threshold*100:.2f}%")
if success:
    exit(0)
else:
    exit(1)