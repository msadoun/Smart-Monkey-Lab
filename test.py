"""Score a trained identity model against automatic-annotation ground truth.

annotate.py is the ground-truth step: its label files are the boxes and the
monkey ids. This script runs the identity model on the test images and lets
Ultralytics match each prediction to those boxes. A match counts only when the
class is the same monkey and the boxes overlap enough. mAP50 is that match at
IoU 0.50. mAP50-95 is the average of the same score from IoU 0.50 to 0.95.

Confidence stays at the Ultralytics validation default, as in validate.py.
The annotation threshold of 0.75 is not reused here.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write per-monkey mAP for a test split whose labels came from annotate.py."
    )
    parser.add_argument("--model", required=True, help="Trained identity checkpoint, for example P4 weights/best.pt.")
    parser.add_argument("--data", type=Path, required=True, help="Dataset YAML with path, test, and names.")
    parser.add_argument("--csv", type=Path, default=Path("test_metrics.csv"), help="Per-monkey results table.")
    parser.add_argument("--imgsz", type=int, default=640, help="Test image size.")
    parser.add_argument("--batch", type=int, default=16, help="Test batch size.")
    parser.add_argument("--device", default="0", help="CUDA device index, or 'cpu'.")
    parser.add_argument("--project", default=None, help="Run folder parent. Omit to use the Ultralytics default.")
    parser.add_argument("--name", default=None, help="Run name. Omit to let Ultralytics name the run.")
    return parser.parse_args(argv)


def load_dataset(data_yaml: Path) -> tuple[dict, Path]:
    with data_yaml.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise SystemExit(f"{data_yaml} is not a YAML mapping.")
    if "test" not in data:
        raise SystemExit(f"{data_yaml} has no 'test' entry.")
    if "names" not in data:
        raise SystemExit(f"{data_yaml} has no 'names' block.")

    root = Path(str(data.get("path") or data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    return data, root


def names_by_id(names) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(class_id): str(name).strip() for class_id, name in names.items()}
    if isinstance(names, list):
        return {index: str(name).strip() for index, name in enumerate(names)}
    raise SystemExit("'names' must be a list or a mapping.")


def split_dirs(root: Path, test_entry: str) -> tuple[Path, Path]:
    image_dir = Path(test_entry)
    if not image_dir.is_absolute():
        image_dir = root / image_dir
    parts = list(image_dir.parts)
    if "images" not in parts:
        raise SystemExit(f"The test path must contain an 'images' directory: {image_dir}")
    label_parts = parts.copy()
    label_parts[len(parts) - 1 - parts[::-1].index("images")] = "labels"
    return image_dir, Path(*label_parts)


def count_ground_truth(label_dir: Path) -> dict[int, tuple[int, int]]:
    """Return class id -> (image count, box count) from the annotation labels."""
    if not label_dir.is_dir():
        raise SystemExit(
            f"No ground-truth labels at {label_dir}. "
            "Put test videos in Dataset/test and run annotate.py before test.py."
        )
    label_files = sorted(label_dir.glob("*.txt"))
    if not label_files:
        raise SystemExit(f"No label files in {label_dir}.")

    images: dict[int, int] = {}
    instances: dict[int, int] = {}
    for label_file in label_files:
        seen: set[int] = set()
        for line in label_file.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if not fields:
                continue
            class_id = int(float(fields[0]))
            instances[class_id] = instances.get(class_id, 0) + 1
            seen.add(class_id)
        for class_id in seen:
            images[class_id] = images.get(class_id, 0) + 1
    return {class_id: (images[class_id], instances[class_id]) for class_id in instances}


def class_scores(metrics) -> dict[int, tuple[float, float, float, float]]:
    """Map class id -> precision, recall, mAP50, mAP50-95 from an Ultralytics val result."""
    box = metrics.box
    try:
        class_ids = [int(class_id) for class_id in box.ap_class_index]
        precision = box.p
        recall = box.r
        all_ap = box.all_ap
    except AttributeError as exc:
        raise SystemExit(f"This Ultralytics version did not return per-class AP: {exc}") from exc

    scores = {}
    for row, class_id in enumerate(class_ids):
        ap_row = list(all_ap[row])
        if not ap_row:
            continue
        map50 = float(ap_row[0])
        map50_95 = float(sum(ap_row) / len(ap_row))
        scores[class_id] = (float(precision[row]), float(recall[row]), map50, map50_95)
    return scores


def write_table(
    csv_path: Path,
    names: dict[int, str],
    counts: dict[int, tuple[int, int]],
    scores: dict[int, tuple[float, float, float, float]],
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    class_ids = sorted(set(names) | set(counts) | set(scores))
    scored = [scores[class_id] for class_id in class_ids if class_id in scores]

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_id", "name", "images", "instances", "precision", "recall", "mAP50", "mAP50-95"])
        for class_id in class_ids:
            image_count, instance_count = counts.get(class_id, (0, 0))
            row = [class_id, names.get(class_id, ""), image_count, instance_count]
            if class_id in scores:
                row.extend(f"{value:.6f}" for value in scores[class_id])
            else:
                row.extend(["", "", "", ""])
            writer.writerow(row)
        if scored:
            means = [sum(column) / len(scored) for column in zip(*scored)]
            writer.writerow(["", "mean", "", "", *(f"{value:.6f}" for value in means)])


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_yaml = args.data.expanduser().resolve()
    csv_path = args.csv.expanduser().resolve()
    if not data_yaml.is_file():
        raise SystemExit(f"Dataset YAML does not exist: {data_yaml}")

    data, root = load_dataset(data_yaml)
    names = names_by_id(data["names"])
    image_dir, label_dir = split_dirs(root, str(data["test"]))
    if not image_dir.is_dir():
        raise SystemExit(f"Test images not found: {image_dir}")
    counts = count_ground_truth(label_dir)

    model = YOLO(args.model)
    val_args = {
        "data": str(data_yaml),
        "split": "test",
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
    }
    if args.project is not None:
        val_args["project"] = args.project
    if args.name is not None:
        val_args["name"] = args.name

    metrics = model.val(**val_args)
    write_table(csv_path, names, counts, class_scores(metrics))
    print(f"Wrote per-monkey scores to {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
