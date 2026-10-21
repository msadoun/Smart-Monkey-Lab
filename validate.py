"""Validate a trained YOLO detector.

Confidence is intentionally left at the Ultralytics validation default. That
default is what produces precision, recall, and mAP for a detection run. Setting
it to the annotation threshold (0.75) would score a different operating point
and would not match the tables in the paper.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a YOLO detection model.")
    parser.add_argument("--model", required=True, help="Trained checkpoint, usually weights/best.pt.")
    parser.add_argument("--data", type=Path, required=True, help="Ultralytics dataset YAML used for training.")
    parser.add_argument("--imgsz", type=int, default=640, help="Validation image size.")
    parser.add_argument("--batch", type=int, default=16, help="Validation batch size.")
    parser.add_argument("--device", default="0", help="CUDA device index, or 'cpu'.")
    parser.add_argument("--split", default="val", choices=("train", "val", "test"), help="Split named in the dataset YAML.")
    parser.add_argument("--project", default=None, help="Run folder parent. Omit to use the Ultralytics default.")
    parser.add_argument("--name", default=None, help="Run name. Omit to let Ultralytics name the run.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data = args.data.expanduser().resolve()
    if not data.is_file():
        raise SystemExit(f"Dataset YAML does not exist: {data}")

    model = YOLO(args.model)
    val_args = {
        "data": str(data),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "split": args.split,
    }
    if args.project is not None:
        val_args["project"] = args.project
    if args.name is not None:
        val_args["name"] = args.name

    model.val(**val_args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
