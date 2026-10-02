# Viva guide: DVC steps and command demos

Local repo: `H:\Projects\laser-detector-mlsd`
GitHub: https://github.com/TonmoyBishwas/laser-detector-mlsd-
DagsHub: https://dagshub.com/ttonmoy46/laser-detector-mlsd-

Before any command, open PowerShell in the repo and activate the venv:

```powershell
cd H:\Projects\laser-detector-mlsd
venv\Scripts\Activate.ps1
```

`(venv)` appears at the start of the prompt. If PowerShell says "running scripts is
disabled", run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

---

## Part 1: how the project was set up

The models were trained with **semi-supervised learning** in an earlier project: a
teacher YOLO26l trained on a few hand-labelled frames, pseudo-labelled the rest, and a
student YOLO26n was trained on everything. For this course, that work is packaged into a
DVC pipeline so data, models and results are versioned and reproducible.

1. **Copy the files into the repo**
   `data/raw/red/`, `data/raw/green/` (the final semi-supervised datasets),
   `models/released/red_yolo26n.pt`, `models/released/green_yolo26n.pt` (the student models).

2. **`dvc init`**
   Creates the `.dvc/` folder: the settings file (`config`) and the local cache (`.dvc/cache`).

3. **`dvc add data/raw/red data/raw/green models/released/red_yolo26n.pt models/released/green_yolo26n.pt`**
   For each item DVC:
   - computes its fingerprint (MD5 hash),
   - copies it into `.dvc/cache`,
   - writes a small pointer file (e.g. `red_yolo26n.pt.dvc`) with the hash and size,
   - adds the real file to `.gitignore` automatically.

4. **`dvc repro`**
   Runs the pipeline in `dvc.yaml`: prepare → train → evaluate.
   Outputs (`data/prepared/`, `models/model.pt`, `metrics/metrics.json`) are cached and
   their hashes are written to `dvc.lock`.

5. **Git commit and push to GitHub**
   `git add` the pointers, `dvc.yaml`, `dvc.lock`, `params.yaml` and the code, then
   `git commit` and `git push`. GitHub gets only code and small text files.

6. **Create a DagsHub repo** and connect it to the GitHub repo.

7. **Point DVC at DagsHub and log in**
   ```powershell
   dvc remote add -d origin https://dagshub.com/ttonmoy46/laser-detector-mlsd-.dvc
   dvc remote modify origin --local auth basic
   dvc remote modify origin --local user ttonmoy46
   dvc remote modify origin --local password <DagsHub token>
   ```
   The first line saves the address (committed). The other three are the login and go into
   `.dvc/config.local`, which is never committed.

8. **`dvc push`**
   Uploads every file the pointers (`.dvc` files and `dvc.lock`) refer to that DagsHub does
   not have yet: the datasets, the models and the pipeline outputs.

**Git vs DVC in one line:** Git (GitHub) stores code and small pointer files; DVC stores
the big data and models in remote storage (DagsHub); `dvc pull` reads the pointers and
downloads the real files.

**How someone else reproduces it:** `git clone` → `dvc pull` → `dvc repro`.

---

## Part 2: what `dvc repro` actually does here

| stage | does | input | output |
|---|---|---|---|
| prepare | checks every label, builds the clean dataset folder | `data/raw/red` (1,748 images) | `data/prepared/` |
| train | **does not train by default**: copies the semi-supervised model (`train.mode: released`) | `models/released/red_yolo26n.pt` | `models/model.pt` |
| evaluate | tests the model on the 216 hand-labelled test frames | `models/model.pt`, `data/prepared` | `metrics/metrics.json` |

`train.mode: train` would train the student step again (YOLO26n, 1280 px, 100 epochs,
batch 16, seed 0, same augmentation). It does not redo the teacher or the pseudo-labels:
their result is the dataset in `data/raw`.

**If asked "why does your training stage not train?":**
"Training takes several hours on an RTX 3090, so by default the stage uses our trained
semi-supervised model. Setting `train.mode: train` in `params.yaml` retrains it with the
same settings; DVC sees the parameter changed and reruns training and evaluation."

---

## Part 3: command demos

Every demo below is safe; each one says how to undo it.

### `dvc status`: "is anything out of date?"
```powershell
dvc status
```
Expected: `Data and pipelines are up to date.`
**Say:** "DVC compares the current hashes of all data, code and parameters with the ones
saved in `dvc.lock`. Nothing changed, so nothing needs to rerun."

### `dvc dag`: "show your pipeline"
```powershell
dvc dag
```
Shows `red.dvc → prepare → train → evaluate`, with the released model also feeding `train`.
**Say:** "DAG = directed acyclic graph. It shows which stage depends on which, so DVC knows
the order and what to rerun when something changes." The green dataset and green model show
as separate boxes because the pipeline currently uses red.

### A parameter changes
In `params.yaml`, under `evaluate`, change `conf: 0.25` to `conf: 0.5`, then:
```powershell
dvc status        # only "evaluate" is out of date
dvc params diff   # shows 0.25 -> 0.5
dvc repro         # skips prepare and train, reruns only evaluate
dvc metrics diff  # shows how the results changed
```
**Say:** "Only `evaluate` uses `evaluate.conf`, so only that stage reruns."
**Undo:** set it back to `0.25`, then `dvc repro`.

### The dataset changes
In `params.yaml`, change `dataset: red` to `dataset: green`, then:
```powershell
dvc status        # all three stages are out of date
dvc repro         # prepare, train, evaluate rerun on green
dvc metrics show  # green results (image accuracy 0.9917)
```
**Say:** "Every stage depends on the dataset, so all of them rerun."
**Undo:** set it back to `red`, then `dvc repro`: prepare and train say
`cached - skipping run`, because DVC remembers earlier results.

### `dvc add`: "how did you track the data?"
```powershell
dvc add data/raw/red
```
Nothing changed, so the pointer stays the same. Open `data\raw\red.dvc` in Notepad to show
the hash (`md5`), size and number of files.
**Say:** "`dvc add` saves the file's hash in a small `.dvc` pointer, copies the file into
the cache, and adds it to `.gitignore`. Git stores the pointer; DVC stores the file."

### `dvc pull`: "get the data back"
```powershell
Remove-Item models\released\red_yolo26n.pt
dvc status        # shows it as deleted
dvc pull          # brings it back
dvc status        # up to date again
```
**Say:** "`dvc pull` reads the pointer files and downloads the real files from DagsHub.
Another person would run `git clone`, then `dvc pull`, then `dvc repro`."

### `dvc push`: "upload your data"
```powershell
dvc push
```
Expected: `Everything is up to date.`
**Say:** "Everything is already on DagsHub. If I added new data, `dvc push` would upload
only the new or changed files."

### `dvc init`: explain, don't run
It has already been run here; running it again only says "already initialized".
**Say:** "`dvc init` creates the `.dvc` folder with the settings and the cache. It is the
first command in a DVC project, like `git init`."

---

## Part 4: likely viva questions, short answers

- **What does `dvc repro` do?** Runs the pipeline in `dvc.yaml`, rerunning only the
  stages whose inputs (data, code, parameters) changed since `dvc.lock` was written.
- **Why DVC?** Git can't handle big data and models well; DVC versions them, stores them
  in remote storage, and makes the pipeline reproducible.
- **What is `dvc.yaml`?** The pipeline definition: each stage's command, dependencies,
  parameters and outputs.
- **What is `params.yaml`?** All settings in one file; each stage declares which ones it uses.
- **What is `dvc.lock`?** The hashes of every input and output from the last run; DVC
  compares against it to decide what to rerun.
- **How does DVC know which stage to rerun?** It compares current hashes with `dvc.lock`;
  a changed dependency or parameter makes that stage and everything after it out of date.
- **What happens if the dataset changes?** Its hash changes, so prepare, train and evaluate
  all rerun.
- **What happens if a parameter changes?** Only the stages that use that parameter (and
  stages after them) rerun.
- **Git vs DVC?** Git stores code and pointers; DVC stores the data and models (DagsHub).
- **`dvc push` / `dvc pull`?** Upload / download the real data and models to / from remote
  storage.
- **How can someone reproduce your experiment?** `git clone` → `dvc pull` → `dvc repro`.

## Note

The evaluate stage needs an NVIDIA GPU (`device=0`), so any `dvc repro` that reruns
evaluate (the parameter and dataset demos) will not work on a laptop without one.
`dvc status`, `dvc dag`, `dvc add`, `dvc pull` and `dvc push` work on any machine.
