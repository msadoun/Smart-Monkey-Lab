"""Build identity-labeled face crops from isolation videos.

Each video is one macaque, and the animal's name is a field in the filename
(by default the fourth underscore-separated field: 2024_02_06_Augustin_RH_01.mp4).
A single-class face detector marks every face it finds as class 0. This script
keeps a frame only when that detector returns exactly one face, and it writes
the label with the named animal's class id from the YAML file instead of 0.

Frames with no face, or with more than one face, are skipped. Nothing is
written for them. Running the detector in a stream means the video is opened
once: the kept frame is the original image already in memory, not a second
decode from a saved text file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
SPLIT_NAMES = ("train", "val", "test")

# 2024_02_06_Augustin_RH_01 -> index 3 is the identity used in the lab filenames.
DEFAULT_NAME_INDEX = 3


def load_name_to_id(yaml_path: Path) -> dict[str, int]:
    """Read the Ultralytics `names` block and return name -> class id."""
    with yaml_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict) or "names" not in data:
        raise SystemExit(f"{yaml_path} has no 'names' block.")

    names = data["names"]
    if isinstance(names, dict):
        pairs = [(int(class_id), str(name).strip()) for class_id, name in names.items()]
    elif isinstance(names, list):
        pairs = [(index, str(name).strip()) for index, name in enumerate(names)]
    else:
        raise SystemExit(f"'names' in {yaml_path} must be a list or a mapping.")

    name_to_id: dict[str, int] = {}
    for class_id, name in pairs:
        if not name:
            raise SystemExit(f"Class {class_id} in {yaml_path} has an empty name.")
        if name in name_to_id:
            raise SystemExit(f"Duplicate name {name!r} in {yaml_path}.")
        name_to_id[name] = class_id
    if not name_to_id:
        raise SystemExit(f"{yaml_path} does not list any names.")
    return name_to_id


def list_videos(folder: Path) -> list[Path]:
    videos = [
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    return sorted(videos, key=lambda path: path.name.lower())


def video_groups(video_dir: Path) -> list[tuple[str | None, list[Path]]]:
    """Pair each video list with its split name.

    A folder that contains train, val, or test subfolders keeps that split:
    videos in train/ are labeled as the train group. A folder of video files,
    with none of those subfolders, is one group and has no split name.
    """
    order = {name: index for index, name in enumerate(SPLIT_NAMES)}
    splits = [
        path
        for path in video_dir.iterdir()
        if path.is_dir() and path.name.lower() in order
    ]
    splits.sort(key=lambda path: order[path.name.lower()])
    loose = list_videos(video_dir)

    if splits and loose:
        names = ", ".join(path.name for path in splits)
        raise SystemExit(
            f"{video_dir} contains videos and split folders ({names}). "
            "Use a folder of videos, or a folder whose videos live only in train/ and val/."
        )
    if splits:
        groups = []
        for folder in splits:
            videos = list_videos(folder)
            if not videos:
                raise SystemExit(f"No videos found in {folder}.")
            groups.append((folder.name, videos))
        return groups
    if not loose:
        raise SystemExit(
            f"No videos found in {video_dir}. "
            "Put videos in this folder, or in train/ and val/ inside it."
        )
    return [(None, loose)]


def output_dirs(output: Path, split: str | None) -> tuple[Path, Path]:
    """YOLO replaces a directory named images with labels, so both names are fixed."""
    root = output / split if split else output
    return root / "images", root / "labels"


def identity_from_filename(video: Path, name_index: int) -> str:
    fields = video.stem.split("_")
    if name_index < 0 or name_index >= len(fields):
        raise SystemExit(
            f"{video.name} has {len(fields)} underscore-separated fields; "
            f"--name-index {name_index} is outside that name."
        )
    name = fields[name_index].strip()
    if not name:
        raise SystemExit(f"The name field in {video.name} is empty.")
    return name


def check_videos_open(videos: list[Path]) -> None:
    for video in videos:
        capture = cv2.VideoCapture(str(video))
        opened = capture.isOpened()
        capture.release()
        if not opened:
            raise SystemExit(f"Could not open video: {video}")


def write_pair(image_path: Path, label_path: Path, image, class_id: int, box) -> None:
    """Save the full-resolution frame and one normalized YOLO box."""
    x_center, y_center, width, height = (float(value) for value in box)
    if not cv2.imwrite(str(image_path), image):
        raise SystemExit(f"Could not write image: {image_path}")
    line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
    with label_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(line)


def annotate_video(
    model: YOLO,
    video: Path,
    class_id: int,
    image_dir: Path,
    label_dir: Path,
    conf: float,
    imgsz: int,
    device: str,
) -> int:
    """Return how many single-face frames were saved from this video."""
    kept = 0
    # stream=True yields one result at a time. Collecting them would hold every
    # frame in memory. vid_stride=1 keeps the loop index equal to the frame index.
    results = model.predict(
        source=str(video),
        stream=True,
        conf=conf,
        imgsz=imgsz,
        device=device,
        vid_stride=1,
        verbose=False,
        save=False,
    )
    for frame_index, result in enumerate(results):
        boxes = result.boxes
        if boxes is None or len(boxes) != 1:
            continue
        if result.orig_img is None:
            raise SystemExit(f"No image was returned for frame {frame_index} of {video.name}.")

        stem = f"{video.stem}_{frame_index}"
        write_pair(
            image_dir / f"{stem}.jpg",
            label_dir / f"{stem}.txt",
            result.orig_img,
            class_id,
            boxes.xywhn[0].tolist(),
        )
        kept += 1
    return kept


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect one face per frame in isolation videos and write YOLO labels "
            "whose class id comes from the filename and the names YAML."
        )
    )
    parser.add_argument(
        "--videos",
        type=Path,
        required=True,
        help="Folder of isolation videos, or a folder that contains train/ and val/ subfolders of videos.",
    )
    parser.add_argument("--weights", type=Path, required=True, help="Single-class face detector weights (.pt).")
    parser.add_argument("--names", type=Path, required=True, help="YAML file with a YOLO names block.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination. Split inputs are written to <split>/images and <split>/labels. A flat input is written to images/ and labels/.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.75,
        help="Detection confidence. 0.75 keeps profile faces without the body-part false positives seen below 0.5.",
    )
    parser.add_argument(
        "--name-index",
        type=int,
        default=DEFAULT_NAME_INDEX,
        help="Zero-based underscore field that holds the animal name. Default 3 matches YYYY_MM_DD_Name_....",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size. Saved frames stay full resolution.")
    parser.add_argument("--device", default="0", help="CUDA device index, or 'cpu'.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    video_dir = args.videos.expanduser().resolve()
    weights = args.weights.expanduser().resolve()
    names_path = args.names.expanduser().resolve()
    output = args.output.expanduser().resolve()

    for path, label in ((video_dir, "Video folder"), (weights, "Weights"), (names_path, "Names YAML")):
        if not path.exists():
            raise SystemExit(f"{label} does not exist: {path}")
    if not video_dir.is_dir():
        raise SystemExit(f"Video folder is not a directory: {video_dir}")

    name_to_id = load_name_to_id(names_path)
    groups = video_groups(video_dir)

    # Reject unknown filenames before loading the model, so a bad name does not
    # spend a GPU pass on the videos that came before it.
    class_ids = {}
    for _split, videos in groups:
        for video in videos:
            name = identity_from_filename(video, args.name_index)
            if name not in name_to_id:
                known = ", ".join(sorted(name_to_id))
                raise SystemExit(
                    f"{video.name} names {name!r}, which is not in {names_path.name}. Known names: {known}"
                )
            class_ids[video] = name_to_id[name]

    check_videos_open([video for _split, videos in groups for video in videos])

    for split, _videos in groups:
        image_dir, label_dir = output_dirs(output, split)
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    total = 0
    video_count = 0
    for split, videos in groups:
        image_dir, label_dir = output_dirs(output, split)
        for video in videos:
            kept = annotate_video(
                model,
                video,
                class_ids[video],
                image_dir,
                label_dir,
                args.conf,
                args.imgsz,
                args.device,
            )
            total += kept
            video_count += 1
            label = f"{split}/{video.name}" if split else video.name
            print(f"{label}: saved {kept} frames", flush=True)

    print(f"Saved {total} frames from {video_count} videos to {output}", flush=True)
    return 0 if total else 1


if __name__ == "__main__":
    sys.exit(main())
