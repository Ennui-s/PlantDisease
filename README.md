# Plant Disease Classification with Transfer Learning

An image classifier that identifies plant leaf diseases from photographs, built by fine-tuning a pretrained **MobileNetV2** (PyTorch / torchvision) on the PlantVillage dataset.

**Test accuracy: 95.18%** (0.9517607332368548). Validation accuracy stabilised at roughly 95-96% during training.

> **Project status:** The trained classifier and its training/evaluation workflow are the original project. Around it, an ML mini-platform is being built phase by phase; so far it adds **data validation**, a **config-driven training entry point**, **MLflow experiment tracking**, a **formal evaluation with a quality gate**, an **MLflow model registry** with a champion alias, and a **FastAPI inference service** that serves the champion, packaged as a **Docker image**, with an automated **test suite and GitHub Actions CI** (see [ML mini-platform](#ml-mini-platform-in-progress)). Monitoring and retraining are not implemented yet. See [Roadmap](#roadmap).

---

## Problem

Farmers cannot physically inspect every crop, especially across large or dispersed fields. Manual inspection is slow, labour-intensive and infrequent, so early signs of disease can go unnoticed until intervention is less effective. The FAO estimates that up to 40% of global crop production is lost each year to pests and plant diseases ([FAO](https://www.fao.org/one-health/highlights/how-plant-diseases-threaten-global-food-security/en)).

## Objective

Classify plant leaf images into their crop and disease/healthy category using a CNN trained with transfer learning. The longer-term motivation is automated inspection (for example from field cameras) that flags diseased plants while they are still treatable. That deployment is not part of this repository.

## Dataset

[PlantVillage (Kaggle)](https://www.kaggle.com/datasets/emmarex/plantdisease/data): labelled leaf images captured under controlled conditions.

The model is trained on **15 classes** across three crops:

| Crop | Classes |
|---|---|
| Pepper (bell) | Bacterial spot, Healthy |
| Potato | Early blight, Late blight, Healthy |
| Tomato | Bacterial spot, Early blight, Late blight, Leaf mold, Septoria leaf spot, Spider mites (two-spotted), Target spot, Yellowleaf curl virus, Mosaic virus, Healthy |

- **Layout:** `dataset/raw/` holds one folder of images per class (for example `Potato___healthy`). Class labels are taken from the folder names.
- **Split:** 70% train / 20% validation / 10% test, applied **per class** by `data/split_dataset.py` (images are shuffled with `random.seed(42)`, then copied into `dataset/train`, `dataset/val` and `dataset/test`).
- **Preprocessing:** images are resized to 224 x 224 and converted to tensors. No normalisation and no data augmentation are applied.
- **Image counts:** 14,441 train / 4,125 validation / 2,073 test files (20,639 total). One of these is an extensionless file that the data loader skips, so the loaders see 20,638 images (see [data validation results](#data-validation-results-on-the-plantvillage-data)).

## Model

| Component | Detail |
|---|---|
| Framework | PyTorch / torchvision |
| Backbone | MobileNetV2 with ImageNet pretrained weights (`IMAGENET1K_V1`), **frozen** (`freeze_backbone = True`) |
| Head | Dropout (0.3) -> Linear (backbone output features -> 15 classes); softmax is applied to the logits at prediction time |
| Loss | Cross-entropy |
| Optimiser | Adam, learning rate 0.001, batch size 32 |
| Training control | Up to 12 epochs; early stopping on validation loss (patience 3); best model by validation accuracy saved to `best_model.pth` |
| Monitoring | TensorBoard (train loss, validation loss, validation accuracy per epoch) |

Only the classifier head is trained. MobileNetV2 was chosen for its balance of efficiency and accuracy.

```
Input (224x224 RGB) -> MobileNetV2 feature extractor (frozen)
  -> Dropout (0.3) -> Linear -> class scores -> softmax -> predicted class
```

### Hyperparameter tuning

Tuning was done by hand, judged on validation accuracy and loss curves in TensorBoard:

- Baseline: learning rate 0.003, batch size 64, frozen backbone. Validation accuracy plateaued early with minor instability.
- Learning rate reduced to 0.001, giving smoother convergence.
- Batch size reduced to 32.
- Partial unfreezing of the backbone was explored during tuning. The final configuration keeps the backbone frozen.
- Early stopping patience was adjusted so training stopped once validation performance stabilised.

Final values (all in `config.py`):

| Setting | Value |
|---|---|
| Learning rate | 0.001 |
| Batch size | 32 |
| Max epochs | 12 |
| Early stopping patience | 3 |
| Dropout | 0.3 |
| Optimiser | Adam |
| Image size | 224 |
| `freeze_backbone` | `True` |

## Evaluation

| Metric | Value |
|---|---|
| Test accuracy | 0.9517607332368548 |
| Validation accuracy | roughly 95-96% (stabilised) |

Evaluation artifacts produced by `training/evaluate.py`:

- **Confusion matrix** (`confusion_matrix.png`): most mass lies on the diagonal. The off-diagonal errors are mostly between similar tomato conditions (for example early blight, late blight, Septoria leaf spot and target spot).
- **Grad-CAM** (`gradcam_<n>.png`): heatmaps overlaid on the leaf image, showing which regions influenced the predicted class, labelled with the predicted class and its confidence. Grad-CAM targets the last convolutional layer of the network. In the examples reviewed during development, the model attended to diseased areas rather than the background.
- **TensorBoard logs**: training and validation curves.

Precision, recall, F1 and per-class metrics are not currently reported.

## Limitations

- Covers only three crops (pepper, potato, tomato) and 15 classes; other crops require further data and training.
- PlantVillage images are taken under controlled conditions. Variation in lighting, occlusion and background is not well represented, so field performance is likely lower than the test accuracy suggests.
- Fine-grained symptoms such as spider mites need high-resolution input to be visible.
- Leaf images say little about root vegetables, whose condition cannot be assessed from leaves alone.

## Repository structure

```
config.py              # central configuration: paths and hyperparameters (classifier defaults)
configs/
  training.yaml        # run configuration: overrides, seed, data validation thresholds
  smoke_test.yaml      # 1-epoch run that writes to separate model/log paths
  quality_gate.yaml    # quality gate margins and where the reference metrics live
  reference_metrics.json  # generated once from the original model; commit it
  experiments/         # baseline, lr_0005, batch_64: experiment configs that extend training.yaml
requirements.txt
best_model.pth         # checkpoint of the best model from training
dataset/               # not in the repo; raw/ holds the images, train/val/test are generated
data/
  split_dataset.py     # per-class train/val/test split
  dataloader.py        # ImageFolder datasets, resize + ToTensor, DataLoaders
models/
  model.py             # PlantDiseaseModel (MobileNetV2 + new classifier head)
evaluation/
  evaluate.py          # evaluate a checkpoint on the test split, apply the quality gate
  metrics.py           # accuracy, precision, recall, F1, per-class metrics, confusion matrix
  gate.py              # quality gate
tests/                   # pytest suite (unit, API and integration tests)
pytest.ini, ruff.toml, requirements-test.txt   # test and lint configuration
.github/workflows/ci.yml # GitHub Actions: lint, tests, Docker build and smoke test
Dockerfile               # inference image (CPU-only torch, bundled champion model)
requirements-serve.txt   # dependencies of the inference image
.dockerignore            # allow-list of what goes into the image
model_export/            # exported champion model baked into the image (python -m registry export); commit it
serving/
  app.py               # FastAPI app: /health, /ready, /model, /predict, /reload
  service.py           # the prediction logic (framework-free)
  images.py            # upload validation and preprocessing
  model_store.py       # loads the champion through the registry alias
  settings.py          # environment-variable settings
  errors.py            # API error type
  check_parity.py      # checks serving preprocessing against the training transform
registry/
  cli.py               # python -m registry register | promote | show | check
  ops.py               # register, promote, list, load the champion (MLflow)
  policy.py            # rules: what may be registered or promoted, version metadata
  pyfunc_model.py      # the classifier packaged as an MLflow model
tracking/
  tracker.py           # MLflow wrapper (no-op when tracking is disabled)
training/
  train.py             # training entry point (original loop + validation gate + tracking + run summary)
  prepare.py           # loads run config, applies it, runs the data validation gate
  run_config.py        # YAML run config loader
  evaluate.py          # original evaluation script (confusion matrix, Grad-CAM); unchanged
utils/
  early_stopping.py    # early stopping
  checkpoint.py        # checkpoint saving
  visualization.py     # confusion matrix and Grad-CAM helpers
validation/
  checks.py            # data validation checks
  validate_data.py     # validation CLI and report
outputs/               # generated evaluation figures
reports/               # generated: data_validation.json, training_run.json, evaluation/<model>/
checkpoints/           # generated: per-experiment model checkpoints (not best_model.pth)
mlflow.db, mlruns/     # generated: MLflow tracking database and artifacts
logs/tensorboard/      # not in the repo; written during training
```

`dataset/`, `logs/` and `outputs/` are listed in `.gitignore`, so the dataset, TensorBoard logs and generated figures are not distributed with the repository.

## Getting started

Developed on Python 3.13 (VS Code). Training takes about 1 hour on an Nvidia GPU using 4 cores.

```bash
# 1. Install dependencies (torch, torchvision, tensorboard, numpy, matplotlib,
#    scikit-learn, Pillow, opencv-python-headless, pyyaml, mlflow, fastapi, uvicorn, python-multipart)
pip install -r requirements.txt

# 2. Put the dataset class folders in dataset/raw/

# 3. Create the train/val/test split
#    Delete any existing train/, val/ and test/ folders inside dataset/ first.
#    Skip this step to reuse the previous split.
python -m data.split_dataset

# 4. Train (validates the splits first; aborts if validation fails)
python -m training.train

# 5. Evaluate (confusion matrix, Grad-CAM)
python -m training.evaluate

# 6. View training curves
tensorboard --logdir logs/tensorboard
```

Training and evaluation use a CUDA GPU automatically when `torch.cuda.is_available()` is true, otherwise the CPU. The default `pip install torch` build on Windows is typically CPU-only; for GPU training install the CUDA build of torch from [pytorch.org](https://pytorch.org/get-started/locally/) before the other requirements.

All hyperparameters are set in `config.py`, except the number of samples Grad-CAM uses, which is set in `training/evaluate.py` (line 37).

The dataset is not included; download it from the Kaggle link above. The original exploratory notebook was run on Google Colab with data and outputs stored on Google Drive.

## ML mini-platform (in progress)

The classifier above is the ML workload. The platform is the engineering around it: it validates the data, runs training in a controlled way, and (in later phases) tracks, gates, versions, serves and monitors the model. The classifier's model, preprocessing and training logic are not changed by the platform.

### Phase 1: data validation and config-driven training (implemented)

**Why:** training should not blindly consume arbitrary data, and a run should be repeatable from a config file.

**Data validation** (`validation/`) checks the dataset and writes `reports/data_validation.json`:

| Check | What it looks for | Severity |
|---|---|---|
| Structure | missing directory, no class folders, empty class folders, stray files, nested folders | error / warning |
| Readability | unreadable, corrupted, truncated or empty image files (every image is fully decoded) | error |
| Format | extension outside `.jpg/.jpeg/.png`, non-image files the loader would skip, extension not matching content | error / warning |
| Channels and size | non-RGB images, images with a side under 32 px, distinct image sizes | warning / error / info |
| Classes | unexpected or missing class folders compared with a recorded baseline | error |
| Class distribution | classes with too few images, large imbalance ratio | error / warning |
| Duplicates | byte-identical images; the same image under different labels | warning / error |
| Leakage | the same image in more than one of train/val/test | error (a known number of groups can be accepted, see below) |
| Split consistency | every class present in every split; split sizes match the configured ratios | error / warning |

Thresholds are set in `configs/training.yaml`. Each stage also records a **dataset fingerprint** (a hash over every file's path and content) so a run can be tied to the exact data it used.

```bash
python -m validation.validate_data --write-baseline   # once: record the class set of dataset/raw
python -m validation.validate_data --stage raw        # check dataset/raw before splitting
python -m validation.validate_data --stage splits     # check dataset/train, val, test
python -m validation.validate_data --stage all
```

The command exits with a non-zero status if any error-level finding exists.

**Config-driven training.** `python -m training.train` runs the original training loop, now wrapped in a function. A config file can extend another with `base:` so experiment files only state what differs. Before training it validates the train/val/test splits and aborts if validation fails (`--skip-validation` bypasses this). `configs/training.yaml` can override any value from `config.py` for a run and optionally sets a torch seed; with no overrides and no seed the run is the same as the original. Each run writes `reports/training_run.json` recording the effective configuration, the dataset fingerprint, epochs run, best validation accuracy and the checkpoint path.

```bash
python -m training.train --config configs/training.yaml
```

**Smoke test of the training entry point.** `configs/smoke_test.yaml` runs one epoch and writes its model and logs to separate paths, so it does not overwrite `best_model.pth` or the existing TensorBoard logs:

```bash
python -m training.train --config configs/smoke_test.yaml
```

**How it was checked:** the validation was exercised on a small synthetic dataset with deliberately injected faults (corrupted, truncated and empty files, wrong formats, tiny and grayscale images, duplicates, a mislabelled duplicate, an unexpected class, an empty class, cross-split leakage and a class missing from a split); each was detected with the intended severity. It was then run on the real dataset (results below). The smoke-test run (`configs/smoke_test.yaml`, 1 epoch, seed 0, CPU) passed the data validation gate, trained for about 5 minutes and reached 91.3% validation accuracy after that single epoch. It wrote `reports/training_run.json` with the effective configuration and the splits' dataset fingerprint (`1deb292ca358bf40...`), which matches the fingerprint from the standalone validation run.

### Data validation results on the PlantVillage data

First run on the real dataset (`python -m validation.validate_data --stage all`):

- 20,639 files in 15 classes; all images are 256 x 256. 20,638 are readable images. Raw dataset fingerprint: `3f199ef9f450658a...` (full value in `reports/data_validation.json`).
- **Extensionless file:** `Tomato__Tomato_YellowLeaf__Curl_Virus/svn-r6Yb5c` is not an image and is skipped by the data loader. It is copied into the train split by the split script.
- **RGBA image:** one PNG in `Pepper__bell___healthy` has an alpha channel; the loader converts it to RGB.
- **Duplicates:** 14 groups of byte-identical images exist in `dataset/raw` (the same source filename under different UUID prefixes).
- **Leakage:** because of those duplicates, 5 identical images appear in both the train split and a held-out split, all in `Tomato_Late_blight` (2 in test, 3 in validation). The recorded test accuracy was computed on 1,973 correct out of 2,073 images. Removing the 2 affected test images would give between 95.17% and 95.27%, depending on whether the model got them right, so the headline result is not materially affected. This check only finds byte-identical files; near-duplicates (for example different photographs of the same leaf) are not detected.
- **Class balance:** the largest class (`Tomato__Tomato_YellowLeaf__Curl_Virus`, 3,209) is 21.1 times the smallest (`Potato___healthy`, 152), below the warning threshold of 30.

**Decision:** the existing split is kept so that the recorded result stays reproducible. The 5 known leaked groups are accepted explicitly through `accepted_cross_split_duplicate_groups: 5` in `configs/training.yaml`; they are reported as a warning, and any additional leakage fails the gate and stops training.

### Phase 2: experiment tracking with MLflow (implemented)

**Why:** every training run should be recorded with the configuration that produced it and the data it used, so runs can be compared instead of remembered.

Each `python -m training.train` run creates an MLflow run (set `tracking.enabled: false` in the config, or pass `--no-tracking`, to turn this off) in the experiment `plant-disease-classifier`. The tracking store is a local SQLite database, `mlflow.db`.

| Logged | What |
|---|---|
| Parameters | every value in `config.py` as overridden for the run, the seed, the number of classes and the number of train/val/test images |
| Tags | dataset fingerprint, data validation outcome, git commit and whether tracked files had uncommitted changes, device, torch version, config file |
| Per-epoch metrics | `train_loss_sum`, `val_loss_sum` (losses summed over batches, exactly as in the TensorBoard curves) and `val_accuracy` |
| Final metrics | `best_val_accuracy`, `best_epoch`, `epochs_run`, `stopped_early` |
| Artifacts | the best checkpoint, `data_validation.json`, `training_run.json` and the config file used |

Experiment configs in `configs/experiments/` extend `configs/training.yaml` and give each run its own checkpoint, so `best_model.pth` is not overwritten:

```bash
python -m training.train --config configs/experiments/baseline.yaml    # original settings
python -m training.train --config configs/experiments/lr_0005.yaml     # learning rate 0.0005
python -m training.train --config configs/experiments/batch_64.yaml    # batch size 64

mlflow ui --backend-store-uri sqlite:///mlflow.db                      # compare runs in the browser
```

`mlflow.db`, `mlruns/` and `checkpoints/` are generated and should be git-ignored.

Test metrics, the confusion matrix and Grad-CAM images are logged by the evaluation step (Phase 3).

**How it was checked:** the tracking wrapper and the training entry point were run against stand-in versions of torch and MLflow to check the sequence of calls (run opened, parameters, tags, per-epoch metrics, final metrics, artifacts, run closed), that a failing run is closed as `FAILED` and still raises, that `--no-tracking` makes no MLflow calls, and that a missing MLflow install gives a clear error. The first tracked run (the smoke test) was logged to a real MLflow store and viewed in the MLflow UI. **`[CONFIRM: comparison of the baseline / lr_0005 / batch_64 runs in the MLflow UI]`**

### Phase 3: evaluation and quality gate (implemented)

**Why:** a model that trained successfully is not automatically a model that should be deployed. Evaluation produces reproducible metrics, and the gate decides whether a candidate may move on.

**Evaluation** (`python -m evaluation.evaluate --model <checkpoint>`) runs the checkpoint over the test split and writes to `reports/evaluation/<model name>/`:
`metrics.json` (accuracy, macro precision/recall/F1, weighted F1, per-class precision/recall/F1/support, confusion matrix, model file hash), `confusion_matrix.png` and `gradcam_<n>.png`, using the existing visualisation code. The results are logged to MLflow, attached to the training run that produced the checkpoint when it can be identified (otherwise a new run, or `--run-id`). The original `training/evaluate.py` is left unchanged. Checkpoints are loaded with `map_location`, so a checkpoint saved on a GPU also loads on a CPU-only machine.

**Quality gate** (`configs/quality_gate.yaml`, `evaluation/gate.py`): a candidate passes only if it is not worse than a **reference** by more than set margins. The reference is the existing classifier evaluated by this platform:

```bash
python -m evaluation.evaluate --model best_model.pth --write-reference   # once; commit configs/reference_metrics.json
python -m evaluation.evaluate --model checkpoints/lr_0005.pth            # evaluate a candidate and apply the gate
```

| Check | Rule | Why this margin |
|---|---|---|
| Accuracy | at most 0.01 below the reference | on 2,073 test images at ~0.952 the 95% sampling noise on accuracy is about 0.009, so a smaller drop cannot be distinguished from noise |
| Macro F1 | at most 0.01 below the reference | same reasoning |
| Per-class recall | no class more than 0.10 below the reference | the smallest test class (`Potato___healthy`) has 16 images, so one image moves its recall by 0.0625; 0.10 tolerates one or two images but catches a class that has really degraded |
| Class set | candidate and reference predict the same classes | metrics of models with different classes are not comparable |

The command exits with status 1 when the gate fails and records `quality_gate: passed|failed` on the MLflow run. Registering passing candidates is Phase 4; comparing a candidate with the current champion comes with retraining.

**How it was checked:** the metric calculations were compared with a hand-computed example; the gate was run on constructed cases (identical, within noise, accuracy or F1 drop, one image lost in the smallest class, a collapsed class, a different class set); and the command was run end to end against stand-ins for torch and MLflow (no reference, writing the reference, a passing candidate, a failing candidate, attaching to a training run or creating a standalone run). The reference run on the original checkpoint (`best_model.pth`, CPU) reproduced the recorded test accuracy of 0.9517607332368548 exactly, with macro F1 0.9534 and macro precision/recall 0.9552/0.9523; the weakest classes are `Tomato_Early_blight` (recall 0.832), `Tomato_Septoria_leaf_spot` (0.882) and `Tomato__Target_Spot` (0.894). Applied to the 1-epoch smoke-test checkpoint, the gate **failed** it (accuracy 0.9059 vs 0.9518, macro F1 0.8949 vs 0.9534, and recall drops in `Tomato_Early_blight` 0.832 -> 0.535 and `Tomato__Tomato_mosaic_virus` 1.000 -> 0.684), as expected for a model trained for one epoch.

### Phase 4: model registry (implemented)

**Why:** a trained file in a folder has no identity, history or status. The registry gives each accepted model a version, records where it came from and how it scored, and lets a service ask for "the champion" instead of a file name.

The registry is the MLflow Model Registry in the same local database as the experiment tracking (`mlflow.db`); the registered model is named `plant-disease-classifier`. Versions are labelled with the **`champion` alias** (not the deprecated Staging/Production stages).

- **Only gate-passed models are registered.** `register` refuses a checkpoint unless `reports/evaluation/<name>/` holds an evaluation on the test split, of exactly that file (hash match), that passed the quality gate.
- **What a version contains:** the weights, the class names in output order, the model code, a model signature (input `float32 (N, 3, 224, 224)` in [0, 1], i.e. resize + ToTensor with no normalisation; output `(N, 15)` softmax probabilities) and pinned `torch`/`torchvision` versions. It builds without downloading ImageNet weights.
- **Version metadata (tags):** checkpoint hash and path, training run id, quality gate outcome, hash of the reference model it was gated against, test accuracy and macro F1, dataset fingerprint, git commit.
- **Champion rules:** the first registered version becomes the champion (there is nothing to compare it with). Later versions are registered but do **not** change the champion; `promote` does that explicitly, and only for versions tagged `quality_gate=passed`. Comparing a new candidate with the current champion automatically comes with retraining.

```bash
python -m evaluation.evaluate --model best_model.pth     # gate against the reference; the original model passes by definition
python -m registry register --model best_model.pth       # version 1, becomes the champion
python -m registry show                                  # versions, aliases, test metrics, gate outcome
python -m registry check                                 # load models:/plant-disease-classifier@champion and run a prediction

# later: a candidate that passed the gate
python -m registry register --model checkpoints/lr_0005.pth    # version 2, champion unchanged
python -m registry promote --version 2                         # champion -> version 2
```

`python -m registry check` loads the model through the alias only, so it also shows that a service can serve the champion without knowing any file path.

**How it was checked:** the registration rules and operations were run against a stand-in in-memory registry: refusals (no evaluation, failed gate, evaluation of a different file, wrong split), registration with metadata, the first version becoming the champion, registering the same checkpoint twice (no duplicate), registering a second model without changing the champion, promotion (and the recorded previous champion), refusing to promote a version without a passed gate or an unknown version, and the version listing. On the real setup, `best_model.pth` was registered as version 1 and became the champion; `python -m registry show` listed it with test accuracy 0.9518, macro F1 0.9534 and the gate passed; and `python -m registry check` loaded the champion through the alias and returned a `(1, 15)` probability vector that sums to 1. MLflow prints warnings about missing type hints and an input example when packaging and loading; they are harmless.

### Phase 5: inference API (implemented)

**Why:** the registry says which model is the champion; the service is what actually answers requests with it, without being rewritten when the champion changes.

`serving/app.py` is a FastAPI service. At startup it asks the registry which version `models:/plant-disease-classifier@champion` points to and loads that version; no model file path is configured anywhere. If the model cannot be loaded the service still starts, `/ready` reports 503 and `/predict` refuses.

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness: the process is up |
| `GET /ready` | readiness: 200 only when the champion is loaded, otherwise 503 |
| `GET /model` | served model name, alias and version, number of classes and their names, input contract, and the version's tags (test accuracy, macro F1, gate outcome, dataset fingerprint, git commit) |
| `POST /predict` | multipart upload, field `file`: returns `prediction`, `confidence`, `top_k`, the model `name`/`alias`/`version`, `latency_ms` (preprocess, inference, total) and a `request_id` |
| `POST /reload` | re-resolves the champion alias and loads it, so a promotion takes effect without a restart. Only exists when `ADMIN_TOKEN` is set, and then needs the header `X-Admin-Token`. A failed reload keeps the current model serving |
| `GET /docs` | interactive API documentation (generated by FastAPI) |

**Preprocessing** is the same as in training: RGB, bilinear resize to 224 x 224, values scaled to [0, 1], no normalisation. `python -m serving.check_parity` compares the service's preprocessing with the training transform on test images.

**Errors** are always JSON, `{"error": {"code", "message", "request_id"}}`, and never contain stack traces or internal paths (those are logged on the server):

| Situation | Status | Code |
|---|---|---|
| no `file` field, or body is not multipart | 422 | `malformed_request` |
| empty file | 400 | `empty_file` |
| not an image, or not JPEG/PNG | 415 | `unsupported_media_type` |
| corrupted or truncated image | 422 | `image_corrupted` |
| image side shorter than 32 px | 422 | `image_too_small` |
| upload over 5 MB (`MAX_UPLOAD_BYTES`) | 413 | `file_too_large` |
| more than 25 million pixels (`MAX_PIXELS`) | 413 | `image_too_large` |
| model not loaded | 503 | `model_unavailable` |
| failure inside prediction | 500 | `inference_failed` |
| unknown path or wrong method | 404 / 405 | `not_found` / `method_not_allowed` |

Settings come from environment variables: `MLFLOW_TRACKING_URI` (default `sqlite:///mlflow.db`), `MODEL_NAME`, `MODEL_ALIAS`, `MAX_UPLOAD_BYTES`, `MAX_PIXELS`, `MIN_SIDE`, `TOP_K`, `ADMIN_TOKEN`.

```powershell
uvicorn serving.app:app --port 8000          # run from the repo root (the registry database is a relative path)

curl.exe http://localhost:8000/ready
curl.exe http://localhost:8000/model
curl.exe -F "file=@dataset/test/Tomato_healthy/<an image>.JPG" http://localhost:8000/predict

# after promoting a new champion (the server started with $env:ADMIN_TOKEN="<token>"):
curl.exe -X POST -H "X-Admin-Token: <token>" http://localhost:8000/reload
```

**Limitations.** The model has 15 classes and always answers with one of them: a photo that is not a leaf of these three crops still gets a prediction, and `confidence` is a softmax probability, not a calibrated certainty. Predictions are processed one at a time per process. Structured request logging and the `/metrics` endpoint are not implemented yet (Phase 8); packaging in a container is Phase 6.

**How it was checked:** everything below the web layer was run directly: upload validation on valid JPEG/PNG (including RGBA, grayscale and palette images) and on empty, random, text, GIF, BMP, truncated, corrupted, tiny and oversized inputs; preprocessing shape, dtype and range; the model store with a stand-in loader (loading, a failed reload keeping the old model, a crash inside the model returning a generic 500, no model loaded returning 503); and response building. The service was then started against the real champion on the real setup and served predictions. **`[CONFIRM: output of python -m serving.check_parity, the error-case responses, and /reload after a promotion]`**

### Phase 6: Docker (implemented)

**Why:** `clone -> build -> run -> send an image -> get a prediction`, on any machine with Docker, without Python, MLflow databases or model files to set up.

The image (`Dockerfile`) contains only the inference service, the model code the packaged model imports, and **a copy of the current champion** in `model_export/`. It therefore needs no registry database, no mounted volume and no secrets (the registry database holds machine-specific artifact paths, so it is deliberately not used inside the container). It uses CPU-only PyTorch, runs as a non-root user, and has a healthcheck on `/ready`. The `.dockerignore` is an allow-list, so the dataset, checkpoints, reports and registry files never enter the image.

```powershell
python -m registry export                      # copy the champion into model_export/ (commit the folder)
docker build -t plant-disease-api .
docker run --rm -p 8000:8000 plant-disease-api

curl.exe http://localhost:8000/ready
curl.exe http://localhost:8000/model           # shows the champion version baked into the image
curl.exe -F "file=@<an image>.JPG" http://localhost:8000/predict
```

When a new champion is promoted, run `python -m registry export` again and rebuild the image; the version it serves is reported by `/model`. To serve straight from the registry instead (outside Docker), do not set `MODEL_DIR`. `/reload` (when `ADMIN_TOKEN` is set at run time with `-e ADMIN_TOKEN=...`) re-reads the model folder.

**How it was checked:** the export and the folder loader were run against a stand-in registry (export contents and metadata, loading from the folder, prediction through the service, no champion giving a clear error). The image was then built and run on the real setup without changes, and `/predict` through it returned `Potato___healthy` (confidence 0.99942, model version 1) for a test image, with about 136 ms of CPU inference time.

### Phase 7: tests and CI (implemented)

**Why:** the platform is software around a model, so it is tested like software: the rules (validation, gate, registry, config) are unit-tested, the API is tested through HTTP, and the real exported model is tested end to end.

```bash
pip install -r requirements-test.txt     # after installing torch and torchvision
python -m pytest -q
ruff check validation tracking evaluation registry serving training/train.py training/prepare.py training/run_config.py tests
```

| Level | File | Covers |
|---|---|---|
| Unit | `test_validation.py` | corrupted/truncated/empty files, formats, tiny images, empty classes, duplicates, label conflicts, leakage and the accepted limit, split consistency, class distribution, class baseline, dataset fingerprint |
| Unit | `test_evaluation.py` | metrics against a hand-computed example; the quality gate on identical, within-noise, accuracy drop, F1 drop, one image lost in the smallest class, a collapsed class, an improved model and a different class set; gate config validation |
| Unit | `test_config_and_registry_policy.py` | config inheritance, unknown keys, circular `base`, overrides, all repository configs load; registration preconditions (hash, split, gate), version tags, promotion rule |
| Unit | `test_serving_logic.py` | upload validation, preprocessing (shape, range, channel order, determinism), response formatting, model store (failed reload keeps the old model), prediction errors, settings |
| API | `test_api.py` | `/health`, `/ready`, `/model`, `/predict` with valid JPEG/PNG, invalid, corrupted, empty, tiny, oversized and missing files, malformed requests, unknown routes, model unavailable, model failure, `/reload` protection, request ids, no internals in errors |
| Integration | `test_integration.py` | the real exported champion loads and predicts; the service's preprocessing equals the training transform; the real model through the API; promotion, listing and export against a real temporary MLflow registry |

The integration tests are skipped automatically if torch, MLflow or `model_export/` are missing.

**CI** (`.github/workflows/ci.yml`) runs on every push to `main` and on pull requests: install CPU-only PyTorch and the test dependencies, lint (syntax errors, undefined names, unused imports), run the tests, then build the Docker image, start it, wait for `/ready`, send a generated image to `/predict` and check that a non-image upload gets a 415. Nothing is trained in CI, and CI needs the committed `model_export/` folder.

**How it was checked:** the unit tests that do not need FastAPI, torch or MLflow (43 test cases) were run, and they were checked against deliberately introduced bugs (a gate that lets any drop through, a wrong preprocessing scale, leakage that never fails, a removed hash check, NaN output accepted, circular configs not detected), all of which made tests fail. **`[CONFIRM: python -m pytest -q on the real setup (API and integration tests included), and the first run of the GitHub Actions workflow]`**

## Roadmap

Implemented so far: data validation and a config-driven training entry point (Phase 1), MLflow experiment tracking of training runs (Phase 2), formal evaluation with a quality gate (Phase 3), the model registry (Phase 4), the inference API (Phase 5), Docker packaging (Phase 6), and tests with CI (Phase 7). The following are **planned and not yet implemented**:

- Inference monitoring
- Retraining and model promotion

This section will be updated as each item is built.
