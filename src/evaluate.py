"""Stage 3 - evaluate: score models/model.pt on the held-out test set.

Writes metrics/metrics.json with
  - box metrics on test and valid: precision, recall, mAP50, mAP50-95
  - image-level accuracy on test: "is there a laser in this frame?"
    (a frame counts as "laser" if the model finds any box with conf >= evaluate.conf)
and metrics/plots/ (PR curve, F1 curve, confusion matrix).

The validation sets are tiny (48 red / 25 green frames), so the test set is the
main result; valid is reported for completeness.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import yaml
from ultralytics import YOLO

from device import best_device

PLOTS = ("BoxPR_curve.png", "BoxF1_curve.png", "confusion_matrix_normalized.png")


def main():
    params = yaml.safe_load(open("params.yaml"))
    e = params["evaluate"]
    data = "data/prepared/data.yaml"
    model = YOLO("models/model.pt")
    device = best_device()
    run_dir = Path("runs").resolve()

    metrics = {"dataset": params["dataset"], "model_source": params["train"]["mode"]}
    for split in ("test", "val"):
        v = model.val(data=data, split=split, imgsz=e["imgsz"], batch=e["batch"], device=device,
                      plots=True, project=str(run_dir), name=f"eval_{split}", exist_ok=True, verbose=False)
        metrics["valid" if split == "val" else "test"] = {
            "precision": round(float(v.box.mp), 4), "recall": round(float(v.box.mr), 4),
            "mAP50": round(float(v.box.map50), 4), "mAP50_95": round(float(v.box.map), 4),
        }

    files = sorted(Path("data/prepared/test/images").glob("*.jpg"))
    truth = np.array([Path(str(f).replace("images", "labels")).with_suffix(".txt").read_text().strip() != ""
                      for f in files])
    pred = []
    for i in range(0, len(files), 32):
        pred += [len(r.boxes) > 0 for r in model.predict([str(f) for f in files[i:i + 32]], conf=e["conf"],
                                                          imgsz=e["imgsz"], device=device, verbose=False)]
    pred = np.array(pred)
    metrics["test_image_level"] = {
        "accuracy": round(float((truth == pred).mean()), 4), "images": len(files),
        "false_alarms": int((~truth & pred).sum()), "misses": int((truth & ~pred).sum()),
    }

    Path("metrics/plots").mkdir(parents=True, exist_ok=True)
    for name in PLOTS:
        shutil.copyfile(run_dir / "eval_test" / name, Path("metrics/plots") / name)
    # newline="\n": the same bytes on every OS, so DVC's hash matches after a clone.
    json.dump(metrics, open("metrics/metrics.json", "w", newline="\n"), indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
