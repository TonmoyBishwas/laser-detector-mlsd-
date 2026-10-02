"""Live laser detection with the model the DVC pipeline produced (models/model.pt).

Which laser is decided by DVC, not by a flag:
    params.yaml `dataset: red | green`  ->  dvc repro  ->  models/model.pt
or switch branches (each branch's dvc.lock points at its own model in the DVC cache):
    git checkout green && dvc checkout
    git checkout main  && dvc checkout      # main = red

Then always just:
    python main.py
    python main.py --source "rtsp://user:pass@192.168.1.20:554/stream1"   # optional override

Settings (source, imgsz, per-laser conf) live in the `detect:` section of params.yaml.
"""

import argparse
import hashlib
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
import detect  # noqa: E402
from device import best_device  # noqa: E402


def check_model(laser):
    """Stop unless models/model.pt is the model dvc.lock records for this dataset.
    Catches an edited params.yaml without `dvc repro`, and a `git checkout` without
    `dvc checkout` (dvc.lock then describes one model while the old file is still on disk)."""
    train = yaml.safe_load(open(ROOT / "dvc.lock"))["stages"]["train"]
    built = train["params"]["params.yaml"]["dataset"]
    if laser != built:
        sys.exit(f"params.yaml says dataset: {laser}, but dvc.lock has the model for {built}.\n"
                 f"Run `dvc repro` to rebuild models/model.pt.")
    path = ROOT / "models/model.pt"
    if not path.exists():
        sys.exit("models/model.pt is missing - run `dvc pull` (or `dvc checkout`).")
    expected = next(o["md5"] for o in train["outs"] if o["path"] == "models/model.pt")
    if hashlib.md5(path.read_bytes()).hexdigest() != expected:
        sys.exit(f"models/model.pt is not the {laser} model that dvc.lock records.\n"
                 f"Run `dvc checkout` (after switching branches) or `dvc repro`.")
    return path


def main():
    params = yaml.safe_load(open(ROOT / "params.yaml"))
    d = params["detect"]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default=str(d["source"]), help=f"override detect.source (now {d['source']})")
    p.add_argument("--save", help="write the annotated output to this file")
    p.add_argument("--no-show", action="store_true", help="do not open a window")
    args = p.parse_args()

    laser = params["dataset"]
    model_path = check_model(laser)

    from ultralytics import YOLO
    args.conf, args.imgsz, args.device = d["conf"][laser], d["imgsz"], best_device()
    models = {laser: YOLO(str(model_path))}
    print(f"model: models/model.pt ({laser}) | conf: {args.conf} | device: {args.device} | "
          f"imgsz: {args.imgsz} | source: {args.source}")

    if Path(args.source).suffix.lower() in detect.IMAGE_EXT:
        detect.run_image(models, args)
    else:
        detect.run_stream(models, args)


if __name__ == "__main__":
    main()
