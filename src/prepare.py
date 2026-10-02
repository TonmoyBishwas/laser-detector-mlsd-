"""Stage 1 - prepare: check the labels and resize frames larger than prepare.max_side.

Reads  data/raw/<dataset>/{train,valid,test}/{images,labels}
Writes data/prepared/                 (resized images + checked labels + data.yaml)
       metrics/data_stats.json        (images / laser / no-laser per split)

A frame with no label file, or an empty one, is a "no laser" frame (negative).
The stage stops with an error if any label line is broken, so a bad dataset
never reaches training.
"""

import json
import shutil
from pathlib import Path

import cv2
import yaml

SPLITS = ("train", "valid", "test")


def check_label_file(path: Path) -> list[str]:
    """Return the label lines, or raise if any line is not 'class cx cy w h' in 0..1."""
    if not path.exists():
        return []
    lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}: expected 5 values, got {line!r}")
        cls, *box = parts
        if cls != "0":
            raise ValueError(f"{path}: unknown class {cls} (only 0 = laser)")
        if not all(0.0 <= float(v) <= 1.0 for v in box):
            raise ValueError(f"{path}: box values must be between 0 and 1: {line!r}")
    return lines


def resize(img, max_side: int):
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return img
    return cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def main():
    params = yaml.safe_load(open("params.yaml"))
    p = params["prepare"]
    src = Path("data/raw") / params["dataset"]
    out = Path("data/prepared")
    if out.exists():
        shutil.rmtree(out)

    stats = {"dataset": params["dataset"]}
    for split in SPLITS:
        (out / split / "images").mkdir(parents=True)
        (out / split / "labels").mkdir(parents=True)
        n_img = n_pos = 0
        for img_path in sorted((src / split / "images").glob("*.jpg")):
            lines = check_label_file(src / split / "labels" / f"{img_path.stem}.txt")
            img = cv2.imread(str(img_path))
            if img is None:
                raise ValueError(f"cannot read image {img_path}")
            if max(img.shape[:2]) <= p["max_side"]:
                # Already small enough: copy the original bytes (re-encoding would blur the tiny dot).
                shutil.copyfile(img_path, out / split / "images" / img_path.name)
            else:
                # YOLO labels are relative to the image size, so they stay valid after resizing.
                cv2.imwrite(str(out / split / "images" / img_path.name), resize(img, p["max_side"]),
                            [cv2.IMWRITE_JPEG_QUALITY, p["jpeg_quality"]])
            (out / split / "labels" / f"{img_path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            n_img += 1
            n_pos += bool(lines)
        stats[split] = {"images": n_img, "laser": n_pos, "no_laser": n_img - n_pos}
        print(f"[prepare] {split:5s}: {n_img} images, {n_pos} laser, {n_img - n_pos} no laser")

    # Relative path: DVC runs every stage from the repo root, so this works on any machine.
    (out / "data.yaml").write_text(yaml.safe_dump({
        "path": "data/prepared", "train": "train/images", "val": "valid/images",
        "test": "test/images", "names": {0: "laser"},
    }, sort_keys=False))

    Path("metrics").mkdir(exist_ok=True)
    json.dump(stats, open("metrics/data_stats.json", "w"), indent=2)


if __name__ == "__main__":
    main()
