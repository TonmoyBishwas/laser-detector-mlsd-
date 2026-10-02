# Laser Pointer Detection — a reproducible DVC pipeline

DS-4491 Machine Learning Systems Design project.

## 1. Problem and dataset

**Problem.** Detect a laser pointer dot (red or green) in camera frames of a projector
screen. The detector drives an interactive projector game ("Laser Guard"): a phone camera
films the screen, the model finds the dot, and the game uses it as a pointer. It must tell
"laser" from "no laser" reliably, because every false alarm or miss shows up on screen.

**Dataset.** Frames cut from short 1920×1080 videos of the projected screen, labelled in
YOLO format with one class, `laser`. Frames with no laser have an empty label file
(negatives). Both datasets were built with **semi-supervised learning** (teacher → pseudo-labels):

1. A small set of frames was labelled by hand on Roboflow.
2. A large **teacher** model (YOLO26l, 1280 px) was trained on that small set.
3. The teacher labelled the remaining unlabelled frames; confident predictions became labels
   (conf ≥ 0.5 → laser, < 0.2 → no laser, in-between → left out or checked by a human).
4. Valid and test frames are **hand-labelled only**, and frames within a short time of a
   valid/test frame were removed from training, so near-duplicates cannot leak the test set.

| dataset | train | valid | test | size |
|---|---|---|---|---|
| red  (`data/raw/red`)   | 1,484 (150 hand + 1,334 teacher) | 48 | 216 | 224 MB |
| green (`data/raw/green`) | 820 (92 hand + 728 teacher) | 25 | 120 | 85 MB |

Full details: `reports/released/RED_LASER_SUMMARY.txt`, `reports/released/GREEN_LASER_SUMMARY.txt`.

## 2. ML model

**YOLO26n** (Ultralytics, nano size, ~2.4 M parameters, 5.5 MB), one class, input 1280 px.
Nano was chosen because the game runs on a laptop with only an Intel iGPU; the dot is tiny,
so a high input resolution matters more than a bigger network.

Trained for 100 epochs from `yolo26n.pt`, batch 16, seed 0, with flips, rotation, scale and
HSV colour augmentation. The **last** epoch is kept (not "best on valid"): the valid sets are
too small (11 and 22 lasers) to pick a checkpoint from.

The trained models are versioned with DVC in `models/released/`.

## 3. Project structure

```
laser-detector-mlsd/
├── data/
│   ├── raw/red/, raw/green/     # datasets (DVC-tracked: raw/red.dvc, raw/green.dvc)
│   └── prepared/                # output of the prepare stage (DVC-tracked)
├── models/
│   ├── released/                # semi-supervised YOLO26n models (DVC-tracked .dvc files)
│   └── model.pt                 # output of the train stage (DVC-tracked)
├── src/
│   ├── prepare.py               # stage 1: check labels, resize big frames, write data.yaml
│   ├── train.py                 # stage 2: released weights or a new training run
│   └── evaluate.py              # stage 3: test-set metrics + plots
├── metrics/                     # metrics.json, data_stats.json, plots/ (in git)
├── reports/released/            # summaries + per-epoch training logs of the released models
├── dvc.yaml                     # the pipeline
├── dvc.lock                     # hashes of every input/output of the last run
├── params.yaml                  # all settings
└── requirements.txt
```

## 4. DVC pipeline

```
data/raw/<dataset>.dvc ──► prepare ──► train ──► evaluate ──► metrics/metrics.json
                                         ▲
               models/released/<dataset>_yolo26n.pt.dvc
```

(`dvc dag` prints this graph.)

| stage | does | depends on | produces |
|---|---|---|---|
| `prepare` | checks every label line (5 values, class 0, box in 0–1; stops on bad data), resizes frames larger than `prepare.max_side` (default 1920 = keep full size), writes `data.yaml` | `src/prepare.py`, `data/raw/${dataset}`, params `dataset`, `prepare` | `data/prepared/`, `metrics/data_stats.json` |
| `train` | `mode: released` copies the semi-supervised model; `mode: train` trains YOLO26n again | `src/train.py`, `data/prepared`, the released model, params `dataset`, `train` | `models/model.pt` |
| `evaluate` | box metrics on test + valid, laser / no-laser accuracy per test image, PR / F1 / confusion plots | `src/evaluate.py`, `data/prepared`, `models/model.pt`, params `evaluate` | `metrics/metrics.json`, `metrics/plots/` |

**`params.yaml`** holds every setting. The useful switches:

- `dataset: red | green` — which laser. Changing it reruns all three stages.
- `train.mode: released | train` — `released` (default) reuses the trained model, so the
  whole pipeline runs in about a minute; `train` retrains from scratch (100 epochs at
  1280 px took several hours on an RTX 3090).

**How DVC knows what to rerun.** `dvc.lock` stores an MD5 hash of every dependency,
parameter and output of the last run. `dvc status` / `dvc repro` compare the current hashes
with it and rerun only the stages whose inputs changed, plus everything downstream.

**Remote storage.** Data and models live in the DagsHub DVC remote
(`https://dagshub.com/ttonmoy46/laser-detector-mlsd-.dvc`); Git only holds the small `.dvc`
pointer files, code and metrics.

## 5. How to run

Requirements: Python 3.12 and an NVIDIA GPU (evaluation runs on `device=0`).

```bash
git clone https://github.com/TonmoyBishwas/laser-detector-mlsd-.git
cd laser-detector-mlsd-
python -m venv venv
venv\Scripts\activate                       # Linux/macOS: source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

dvc pull          # download datasets, models and pipeline outputs from DagsHub
dvc status        # "Data and pipelines are up to date."
dvc repro         # rerun the pipeline (only stages whose inputs changed)
dvc dag           # show the pipeline graph
dvc metrics show  # show the results
```

Try a change:

```bash
# edit params.yaml: dataset: green
dvc status        # prepare, train, evaluate are now out of date
dvc repro         # reruns them on the green dataset
dvc metrics diff  # compare with the last committed metrics
```

The DagsHub repo is private, so `dvc pull` and `dvc push` both need a DagsHub login
(run these once before `dvc pull`):

```bash
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <dagshub-username>
dvc remote modify origin --local password <dagshub-token>
```

(`--local` writes to `.dvc/config.local`, which is never committed.)

### On a Mac (Apple Silicon)

The pipeline and the live detector pick the device automatically (`src/device.py`):
NVIDIA GPU → Apple GPU (`mps`) → CPU. `dvc repro` with `train.mode: released` runs in
about 30 s on an M4 and reproduces the numbers below.

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install torch torchvision -r requirements.txt
# python.org Python ships without CA certificates; without this dvc pull fails with
# CERTIFICATE_VERIFY_FAILED (or run "Install Certificates.command" once instead)
export SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')"
dvc pull
```

### Live detection with the pipeline's model (`main.py`)

`python main.py` runs `models/model.pt` — the model the pipeline produced — with the
settings in the `detect:` section of `params.yaml` (source, input size, per-laser
confidence). Which laser is chosen through DVC, so the command never changes:

```bash
git checkout green && dvc checkout   # green branch: dataset: green, its model and metrics
python main.py
git checkout main && dvc checkout    # main: red
python main.py
python main.py --source "rtsp://user:pass@192.168.1.20:554/stream1"   # one-off override
```

Or stay on one branch, edit `dataset:` in `params.yaml` and run `dvc repro`. If
`params.yaml` and `models/model.pt` disagree, `main.py` stops and says which command to run.

### Live detection with any released model (`detect.py`)

Runs the released models on a webcam, RTSP stream, video file or image, on this machine:

```bash
python detect.py                                  # built-in webcam, green model
python detect.py --laser red                      # red model (or --laser both)
python detect.py --source 1                       # another camera, e.g. iPhone Continuity Camera
python detect.py --source "rtsp://user:pass@192.168.1.20:554/stream1"
python detect.py --source clip.mp4 --save out.mp4
python detect.py --source data/raw/green/test/images/<frame>.jpg
```

Keys: `q`/Esc quit, `s` save a snapshot. `--conf` sets the threshold (default red 0.80 —
fewer false alarms on webcam scenes, 8/113 test misses instead of 3 — and green 0.25), `--imgsz` the input size (default 1280, as trained; 960/640 is faster but
misses small dots). On an M4 the green model runs at ~48 FPS at 1280 px. The first webcam
run asks macOS for camera permission for your terminal app (System Settings → Privacy &
Security → Camera).

## 6. Results

Test sets are hand-labelled frames the models never trained on.

| model | test frames | precision | recall | mAP50 | mAP50-95 | laser / no-laser accuracy |
|---|---|---|---|---|---|---|
| **Red** YOLO26n  | 216 | 0.973 | 0.991 | 0.994 | 0.815 | 97.7 % (211/216: 1 false alarm, 4 misses) |
| **Green** YOLO26n | 120 | 0.833 | 0.830 | 0.700 | 0.289 | 99.2 % (119/120: 0 false alarms, 1 miss) |

`dvc repro` reproduces these numbers exactly on an NVIDIA GPU (`dvc metrics show`; the
`green` branch has the green run). The `metrics.json` committed now comes from a run on an
Apple M4 GPU, which differs only in the last digits (red: mAP50-95 0.813, 98.1 % with 3
misses; green: mAP50-95 0.286) because the two GPUs round floating-point maths differently.

**Effect of preprocessing.** Shrinking the frames to 1280 px in the prepare stage
(`prepare.max_side: 1280`) gave a smaller dataset but lowered red test mAP50-95 from
0.815 to 0.731 (box precision suffers on a dot a few pixels wide), so the default keeps
the full-size frames.

**Reading the numbers.** mAP50-95 is low for green because the dot is only a few pixels
wide: a box that is one or two pixels off fails the strict overlap thresholds even though the
dot is found. For the game, the image-level accuracy is what matters, and it is ≥ 97.7 % for
both lasers. The semi-supervised student (YOLO26n) matches its much larger teacher (YOLO26l)
on test mAP50 while being small enough for a laptop.
