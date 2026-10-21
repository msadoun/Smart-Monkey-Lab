"""Train a YOLO detector on a dataset YAML.

The dataset YAML must use the same `names` order as the file passed to
annotate.py. Class ids in the label files are indexes into that list.

Default of one epoch matches the Smart Monkey Lab runs: each collection period
is large enough that a single pass is the training run. Raise --epochs for a
smaller set. --batch -1 asks Ultralytics AutoBatch to pick a batch size that
fits the GPU.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO detection model.")
    parser.add_argument("--model", required=True, help="Model checkpoint or architecture, for example yolov9e.pt.")
    parser.add_argument("--data", type=Path, required=True, help="Ultralytics dataset YAML (path, train, val, names).")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs. Default 1 matches the period-wise runs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=-1, help="Batch size. -1 uses AutoBatch.")
    parser.add_argument("--device", default="0", help="CUDA device index, or 'cpu'.")
    parser.add_argument("--workers", type=int, default=None, help="Dataloader workers. Omit to use the Ultralytics default.")
    parser.add_argument("--project", default=None, help="Run folder parent. Omit to use runs/detect.")
    parser.add_argument("--name", default=None, help="Run name. Omit to let Ultralytics name the run.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data = args.data.expanduser().resolve()
    if not data.is_file():
        raise SystemExit(f"Dataset YAML does not exist: {data}")
    if args.epochs < 1:
        raise SystemExit("--epochs must be at least 1.")

    model = YOLO(args.model)
    train_args = {
        "data": str(data),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
    }
    if args.workers is not None:
        train_args["workers"] = args.workers
    if args.project is not None:
        train_args["project"] = args.project
    if args.name is not None:
        train_args["name"] = args.name

    model.train(**train_args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
