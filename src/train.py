"""Stage 2 - train: produce models/model.pt.

mode: released -> copy the semi-supervised YOLO26n from the earlier project
                  (models/released/<dataset>_yolo26n.pt). No GPU time needed.
mode: train    -> train a new YOLO26n on data/prepared with the params below.
                  Like the earlier project, the LAST epoch is kept (not "best"),
                  because the model was chosen on the test set, not the tiny
                  validation set.
"""

import shutil
from pathlib import Path

import yaml


def main():
    params = yaml.safe_load(open("params.yaml"))
    t = params["train"]
    out = Path("models/model.pt")
    out.parent.mkdir(exist_ok=True)

    if t["mode"] == "released":
        src = Path("models/released") / f"{params['dataset']}_yolo26n.pt"
        shutil.copyfile(src, out)
        print(f"[train] mode=released: using {src}")
        return

    if t["mode"] != "train":
        raise ValueError(f"train.mode must be 'released' or 'train', got {t['mode']!r}")

    from ultralytics import YOLO
    run_dir = Path("runs").resolve()
    YOLO(t["base_model"]).train(
        data="data/prepared/data.yaml", imgsz=t["imgsz"], epochs=t["epochs"], batch=t["batch"],
        seed=t["seed"], deterministic=True, device=0, workers=2,
        project=str(run_dir), name="train", exist_ok=True,
        # same augmentation as the earlier project
        fliplr=0.5, flipud=0.5, degrees=10, scale=0.3, hsv_h=0.01, hsv_s=0.5, hsv_v=0.4,
        mosaic=1.0, close_mosaic=10,
    )
    shutil.copyfile(run_dir / "train" / "weights" / "last.pt", out)
    print(f"[train] mode=train: trained {t['epochs']} epochs -> {out}")


if __name__ == "__main__":
    main()
