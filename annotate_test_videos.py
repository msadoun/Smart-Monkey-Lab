"""Write ground truth for the test videos.

Single-animal files are labeled directly: a frame is kept when the face
detector finds exactly one face, and that box gets the name in the filename.

Multi-animal files are listed in configs/test_positions.csv. The sheet records
the direction in which the filmed faces separate, and the order of names from
first to last along that direction. When every animal is visible, boxes are
sorted that way and matched to that order. A slot named nonfocal is tracked so
it is not given a focal name, and its box is not written. If one animal leaves,
the box that continues a previous face keeps that face's name. A full view
later locks the order from the sheet again.

Labels are YOLO text files without confidence. Drawn videos show only the
boxes that were written.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
NAME_INDEX = 3
NONFOCAL = "nonfocal"


def load_names(yaml_path: Path) -> dict[str, int]:
    with yaml_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    names = data["names"]
    if isinstance(names, dict):
        pairs = ((str(name).strip(), int(class_id)) for class_id, name in names.items())
    else:
        pairs = ((str(name).strip(), index) for index, name in enumerate(names))
    return {name: class_id for name, class_id in pairs}


def load_positions(csv_path: Path) -> dict[str, dict]:
    """Read direction and the name order along it.

    `order` lists every animal from 1st to last. Horizontal 1st is the leftmost
    face. Vertical 1st is the topmost face. Any length is accepted.
    """
    layouts = {}
    with csv_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            stem = row["video"].strip()
            direction = row["direction"].strip().lower()
            if direction not in {"horizontal", "vertical"}:
                raise SystemExit(f"{stem} has direction {direction!r}. Use horizontal or vertical.")
            order = [name.strip() for name in row["order"].split(",") if name.strip()]
            if len(order) < 2:
                raise SystemExit(f"{stem} needs at least two names in order.")
            layouts[stem] = {"direction": direction, "order": order}
    return layouts


def identity_token(stem: str) -> str:
    fields = stem.split("_")
    if len(fields) <= NAME_INDEX:
        raise SystemExit(f"{stem} has no animal name in field {NAME_INDEX}.")
    return fields[NAME_INDEX].strip()


def slots_for(stem: str, layouts: dict[str, dict], names: dict[str, int]) -> list[dict] | None:
    layout = layouts.get(stem)
    if layout is None:
        return None
    filename_names = identity_token(stem).split("-")
    order = layout["order"]
    if sorted(name.lower() for name in filename_names) != sorted(name.lower() for name in order):
        raise SystemExit(f"{stem} lists {filename_names}, but the sheet order is {order}.")
    canonical = {name.lower(): name for name in filename_names}
    slots = []
    for rank, animal in enumerate(order):
        if animal.lower() == NONFOCAL:
            slots.append({"name": None, "rank": rank})
            continue
        proper = canonical[animal.lower()]
        if proper not in names:
            raise SystemExit(f"{stem} names {proper!r}, which is not in the class list.")
        slots.append({"name": proper, "rank": rank})
    return slots


def list_videos(folder: Path) -> list[Path]:
    videos = [
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    return sorted(videos, key=lambda path: path.name.lower())


def boxes_of(result) -> list[tuple[float, float, float, float]]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    return [tuple(float(value) for value in row) for row in boxes.xyxy.cpu().tolist()]


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def center_distance(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax = (a[0] + a[2]) / 2
    ay = (a[1] + a[3]) / 2
    bx = (b[0] + b[2]) / 2
    by = (b[1] + b[3]) / 2
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def assign_spatial(
    dets: list[tuple[float, float, float, float]],
    slots: list[dict],
    sorting: str,
) -> dict[int, int]:
    """Match each detection index to a slot index by the sheet's axis."""
    axis = 0 if sorting == "horizontal" else 1

    def coordinate(box: tuple[float, float, float, float]) -> float:
        return (box[axis] + box[axis + 2]) / 2

    det_order = sorted(range(len(dets)), key=lambda index: coordinate(dets[index]))
    slot_order = sorted(range(len(slots)), key=lambda index: slots[index]["rank"])
    return {det_index: slot_index for det_index, slot_index in zip(det_order, slot_order)}


def assign_partial(
    dets: list[tuple[float, float, float, float]],
    state: list[dict],
    frame_width: float,
    frame_height: float,
) -> dict[int, int]:
    """Carry names from earlier frames when a face has left the shot.

    A detection that still overlaps a remembered box keeps that identity.
    If every animal but one is already matched, the remaining box is that
    missing animal. Otherwise a box is used only when it is clearly nearer
    one remembered face than the others.
    """
    pairs = []
    for det_index, det in enumerate(dets):
        for slot_index, slot in enumerate(state):
            if slot["box"] is None:
                continue
            score = iou(det, slot["box"])
            if score >= 0.15:
                pairs.append((score, det_index, slot_index))
    pairs.sort(reverse=True)
    used_dets: set[int] = set()
    used_slots: set[int] = set()
    assignment: dict[int, int] = {}
    for _score, det_index, slot_index in pairs:
        if det_index in used_dets or slot_index in used_slots:
            continue
        used_dets.add(det_index)
        used_slots.add(slot_index)
        assignment[det_index] = slot_index

    unmatched_dets = [index for index in range(len(dets)) if index not in used_dets]
    unmatched_slots = [index for index in range(len(state)) if index not in used_slots and state[index]["box"] is not None]
    if len(unmatched_dets) == 1 and len(unmatched_slots) == 1:
        assignment[unmatched_dets[0]] = unmatched_slots[0]
        return assignment

    if len(unmatched_dets) == 1 and len(unmatched_slots) > 1:
        det = dets[unmatched_dets[0]]
        distances = sorted(
            (center_distance(det, state[slot_index]["box"]), slot_index) for slot_index in unmatched_slots
        )
        nearest, slot_index = distances[0]
        second = distances[1][0]
        diagonal = (frame_width**2 + frame_height**2) ** 0.5
        if nearest < second * 0.5 and nearest < diagonal * 0.2:
            assignment[unmatched_dets[0]] = slot_index
    return assignment


def box_diagonal(box: tuple[float, float, float, float]) -> float:
    return ((box[2] - box[0]) ** 2 + (box[3] - box[1]) ** 2) ** 0.5


def link_score(
    current: tuple[float, float, float, float],
    earlier: tuple[float, float, float, float],
    frame_gap: int,
) -> float:
    """How confidently two boxes are the same face across a short gap."""
    overlap = iou(current, earlier)
    if overlap >= 0.1:
        return overlap
    limit = max(box_diagonal(current), box_diagonal(earlier), 40.0) * (1.0 + 0.45 * frame_gap)
    distance = center_distance(current, earlier)
    if distance <= limit:
        return 0.05 * (1.0 - distance / limit)
    return 0.0


def identities_from_locks(
    frame_dets: list[list[tuple[float, float, float, float]]],
    slots: list[dict],
    sorting: str,
    gap: int = 15,
) -> tuple[list[dict[int, int]], int]:
    """Lock names on full frames, then carry them backward and forward.

    A full frame is one detection per animal in the filename. Those boxes get
    the sheet order. Every other detection keeps a name only when it
    continues one of those boxes. The walk runs in both directions, so a face
    seen before the first full frame is named from the later lock.
    """
    assigned: list[dict[int, int]] = [{} for _ in frame_dets]
    locks = 0
    for frame, dets in enumerate(frame_dets):
        if len(dets) == len(slots):
            assigned[frame] = assign_spatial(dets, slots, sorting)
            locks += 1
    if locks == 0:
        return assigned, locks

    def spread(direction: int) -> int:
        added = 0
        order = range(len(frame_dets) - 2, -1, -1) if direction < 0 else range(1, len(frame_dets))
        source_sign = 1 if direction < 0 else -1
        for frame in order:
            dets = frame_dets[frame]
            unlabeled = [index for index in range(len(dets)) if index not in assigned[frame]]
            if not unlabeled:
                continue
            source = None
            source_gap = 0
            for step in range(1, gap + 1):
                other = frame + source_sign * step
                if other < 0 or other >= len(frame_dets):
                    break
                if assigned[other]:
                    source = other
                    source_gap = step
                    break
            if source is None:
                continue
            pairs = []
            for det_index in unlabeled:
                for source_index, slot_index in assigned[source].items():
                    score = link_score(dets[det_index], frame_dets[source][source_index], source_gap)
                    if score > 0:
                        pairs.append((score, det_index, slot_index))
            pairs.sort(reverse=True)
            used_dets: set[int] = set()
            used_slots = set(assigned[frame].values())
            for _score, det_index, slot_index in pairs:
                if det_index in used_dets or slot_index in used_slots:
                    continue
                used_dets.add(det_index)
                used_slots.add(slot_index)
                assigned[frame][det_index] = slot_index
                added += 1
        return added

    for _ in range(8):
        if spread(-1) + spread(1) == 0:
            break
    return assigned, locks


def collect_detections(model: YOLO, video: Path, conf: float, imgsz: int, device: str):
    frames = []
    size = None
    for result in model.predict(
        source=str(video),
        stream=True,
        conf=conf,
        imgsz=imgsz,
        device=device,
        vid_stride=1,
        verbose=False,
        save=False,
    ):
        if result.orig_img is None:
            raise SystemExit(f"No image was returned for {video.name}.")
        if size is None:
            height, width = result.orig_img.shape[:2]
            size = (width, height)
        frames.append(boxes_of(result))
    return frames, size


def annotate_multi_video(
    model: YOLO,
    video: Path,
    slots: list[dict],
    sorting: str,
    names: dict[str, int],
    drawn_path: Path,
    label_dir: Path,
    imgsz: int,
    device: str,
) -> tuple[int, int, int, str]:
    """Detect every face, lock the sheet, then name faces before and after that lock."""
    # Low enough that a second, smaller face still exists for the position lock.
    # A frame is locked only when the count matches the animals in the filename,
    # so an extra body-part box does not get a name.
    frame_dets, _size = collect_detections(model, video, 0.25, imgsz, device)
    assigned, locks = identities_from_locks(frame_dets, slots, sorting)
    if label_dir.exists():
        for old in label_dir.glob("*.txt"):
            old.unlink()
    label_dir.mkdir(parents=True, exist_ok=True)
    if drawn_path.exists():
        drawn_path.unlink()

    fps = video_fps(video)
    capture = cv2.VideoCapture(str(video))
    writer = None
    labeled_frames = 0
    boxes_written = 0
    for frame_index, dets in enumerate(frame_dets):
        ok, image = capture.read()
        if not ok:
            break
        height, width = image.shape[:2]
        if writer is None:
            writer = open_writer(drawn_path, fps, width, height)
        drawn = []
        lines = []
        for det_index, slot_index in assigned[frame_index].items():
            name = slots[slot_index]["name"]
            if name is None:
                continue
            box = dets[det_index]
            drawn.append((box, name))
            lines.append(yolo_line(box, names[name], width, height))
        if lines:
            write_label_file(label_dir / f"{video.stem}_{frame_index}.txt", lines)
            labeled_frames += 1
            boxes_written += len(lines)
            draw_boxes(image, drawn)
        writer.write(image)
    capture.release()
    if writer is not None:
        writer.release()
    note = "" if locks else "never saw every animal in one frame"
    if locks:
        note = f"locked on {locks} frames"
    return len(frame_dets), labeled_frames, boxes_written, note


def yolo_line(box: tuple[float, float, float, float], class_id: int, width: int, height: int) -> str:
    x1, y1, x2, y2 = box
    x_center = min(1.0, max(0.0, ((x1 + x2) / 2) / width))
    y_center = min(1.0, max(0.0, ((y1 + y2) / 2) / height))
    box_w = min(1.0, max(0.0, (x2 - x1) / width))
    box_h = min(1.0, max(0.0, (y2 - y1) / height))
    return f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"


def draw_boxes(image, labels: list[tuple[tuple[float, float, float, float], str]]) -> None:
    for box, name in labels:
        x1, y1, x2, y2 = (int(round(value)) for value in box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 180, 0), 2)
        cv2.putText(image, name, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 0), 2, cv2.LINE_AA)


def focal_labels(
    dets: list[tuple[float, float, float, float]],
    assignment: dict[int, int],
    slots: list[dict],
) -> list[tuple[tuple[float, float, float, float], str]]:
    labels = []
    for det_index, slot_index in assignment.items():
        name = slots[slot_index]["name"]
        if name is None:
            continue
        labels.append((dets[det_index], name))
    return labels


def open_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"Could not open a video writer for {path}.")
    return writer


def video_fps(path: Path) -> float:
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    if fps != fps or fps <= 1 or fps > 120:
        return 30.0
    return float(fps)


def write_label_file(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")


def annotate_video(
    model: YOLO,
    video: Path,
    slots: list[dict] | None,
    sorting: str | None,
    single_name: str | None,
    names: dict[str, int],
    drawn_path: Path,
    label_dir: Path,
    conf: float,
    imgsz: int,
    device: str,
) -> tuple[int, int, int, str]:
    if slots is not None:
        return annotate_multi_video(
            model, video, slots, sorting or "horizontal", names, drawn_path, label_dir, imgsz, device
        )

    fps = video_fps(video)
    writer = None
    frames = 0
    labeled_frames = 0
    boxes_written = 0
    note = ""
    if label_dir.exists():
        for old in label_dir.glob("*.txt"):
            old.unlink()
    label_dir.mkdir(parents=True, exist_ok=True)

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
        image = result.orig_img
        if image is None:
            raise SystemExit(f"No image was returned for frame {frame_index} of {video.name}.")
        height, width = image.shape[:2]
        if writer is None:
            writer = open_writer(drawn_path, fps, width, height)
        dets = boxes_of(result)
        drawn: list[tuple[tuple[float, float, float, float], str]] = []

        if len(dets) == 1 and single_name is not None:
            drawn = [(dets[0], single_name)]

        if drawn:
            lines = [yolo_line(box, names[name], width, height) for box, name in drawn]
            write_label_file(label_dir / f"{video.stem}_{frame_index}.txt", lines)
            labeled_frames += 1
            boxes_written += len(lines)
            draw_boxes(image, drawn)
        writer.write(image)
        frames += 1

    if writer is not None:
        writer.release()
    return frames, labeled_frames, boxes_written, note


def write_summary(path: Path, fresh_rows: list[dict[str, str]], replace_only: bool) -> None:
    fields = ["video", "mode", "frames", "labeled_frames", "boxes", "note"]
    by_name = {row["video"]: row for row in fresh_rows}
    if replace_only and path.is_file():
        with path.open(encoding="utf-8", newline="") as handle:
            existing = list(csv.DictReader(handle))
        merged = []
        seen = set()
        for row in existing:
            replacement = by_name.get(row["video"])
            if replacement is None:
                merged.append(row)
            else:
                merged.append(replacement)
                seen.add(row["video"])
        for name, row in by_name.items():
            if name not in seen:
                merged.append(row)
        rows = merged
    else:
        rows = fresh_rows
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Annotate the test videos and draw the ground truth.")
    parser.add_argument("--videos", type=Path, default=root.parent / "Test videos")
    parser.add_argument("--weights", type=Path, default=root.parent / "Automatic annotation" / "weights" / "best.pt")
    parser.add_argument("--names", type=Path, default=root.parent / "YAML" / "monkey-ID.yml")
    parser.add_argument("--positions", type=Path, default=root / "configs" / "test_positions.csv")
    parser.add_argument("--output", type=Path, default=root.parent / "Test ground truth")
    parser.add_argument("--conf", type=float, default=0.65)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--multi-only",
        action="store_true",
        help="Redo only the videos listed in the position sheet.",
    )
    parser.add_argument(
        "--single-only",
        action="store_true",
        help="Redo only the videos that are not listed in the position sheet.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    video_dir = args.videos.expanduser().resolve()
    weights = args.weights.expanduser().resolve()
    names_path = args.names.expanduser().resolve()
    positions_path = args.positions.expanduser().resolve()
    output = args.output.expanduser().resolve()
    for path, label in (
        (video_dir, "Video folder"),
        (weights, "Weights"),
        (names_path, "Names YAML"),
        (positions_path, "Positions sheet"),
    ):
        if not path.exists():
            raise SystemExit(f"{label} does not exist: {path}")

    names = load_names(names_path)
    layouts = load_positions(positions_path)
    videos = list_videos(video_dir)
    if not videos:
        raise SystemExit(f"No videos found in {video_dir}.")

    jobs = []
    for video in videos:
        slots = slots_for(video.stem, layouts, names)
        if slots is None:
            animal = identity_token(video.stem)
            if animal not in names:
                raise SystemExit(f"{video.name} names {animal!r}, which is not in the class list.")
            jobs.append((video, None, None, animal))
        else:
            jobs.append((video, slots, layouts[video.stem]["direction"], None))
    if args.multi_only and args.single_only:
        raise SystemExit("Use either --multi-only or --single-only.")
    if args.multi_only:
        jobs = [job for job in jobs if job[1] is not None]
    elif args.single_only:
        jobs = [job for job in jobs if job[1] is None]

    drawn_dir = output / "drawn"
    label_root = output / "labels"
    drawn_dir.mkdir(parents=True, exist_ok=True)
    label_root.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    summary_path = output / "summary.csv"
    fresh_rows = []
    for index, (video, slots, sorting, single_name) in enumerate(jobs, start=1):
        mode = "multi" if slots is not None else "single"
        print(f"[{index}/{len(jobs)}] {video.name} ({mode})", flush=True)
        frames, labeled_frames, boxes_written, note = annotate_video(
            model,
            video,
            slots,
            sorting,
            single_name,
            names,
            drawn_dir / f"{video.stem}.mp4",
            label_root / video.stem,
            args.conf,
            args.imgsz,
            args.device,
        )
        fresh_rows.append(
            {
                "video": video.name,
                "mode": mode,
                "frames": str(frames),
                "labeled_frames": str(labeled_frames),
                "boxes": str(boxes_written),
                "note": note,
            }
        )
        print(
            f"  frames {frames}, labeled {labeled_frames}, boxes {boxes_written}"
            + (f" ({note})" if note else ""),
            flush=True,
        )
    write_summary(summary_path, fresh_rows, replace_only=args.multi_only or args.single_only)
    print(f"Drawn videos: {drawn_dir}", flush=True)
    print(f"Labels: {label_root}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
