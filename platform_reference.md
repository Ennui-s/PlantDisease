# ML mini-platform: reference

Detailed documentation of each stage of the platform built around the existing Plant Disease Classifier: what it does, why it exists, how to run it and how it was checked. The overview is in the [README](../README.md); the design is in [architecture.md](architecture.md) and the reasoning in [engineering_decisions.md](engineering_decisions.md).

Items marked **`[CONFIRM: ...]`** have not been verified yet; they are collected in the README's status section.

The classifier above is the ML workload. The platform is the engineering around it: it validates the data, runs training in a controlled way, and (in later phases) tracks, gates, versions, serves and monitors the model. The classifier's model, preprocessing and training logic are not changed by the platform.

## Phase 1: data validation and config-driven training (implemented)

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

## Data validation results on the PlantVillage data

First run on the real dataset (`python -m validation.validate_data --stage all`):

- 20,639 files in 15 classes; all images are 256 x 256. 20,638 are readable images. Raw dataset fingerprint: `3f199ef9f450658a...` (full value in `reports/data_validation.json`).
- **Extensionless file:** `Tomato__Tomato_YellowLeaf__Curl_Virus/svn-r6Yb5c` is not an image and is skipped by the data loader. It is copied into the train split by the split script.
- **RGBA image:** one PNG in `Pepper__bell___healthy` has an alpha channel; the loader converts it to RGB.
- **Duplicates:** 14 groups of byte-identical images exist in `dataset/raw` (the same source filename under different UUID prefixes).
- **Leakage:** because of those duplicates, 5 identical images appear in both the train split and a held-out split, all in `Tomato_Late_blight` (2 in test, 3 in validation). The recorded test accuracy was computed on 1,973 correct out of 2,073 images. Removing the 2 affected test images would give between 95.17% and 95.27%, depending on whether the model got them right, so the headline result is not materially affected. This check only finds byte-identical files; near-duplicates (for example different photographs of the same leaf) are not detected.
- **Class balance:** the largest class (`Tomato__Tomato_YellowLeaf__Curl_Virus`, 3,209) is 21.1 times the smallest (`Potato___healthy`, 152), below the warning threshold of 30.

**Decision:** the existing split is kept so that the recorded result stays reproducible. The 5 known leaked groups are accepted explicitly through `accepted_cross_split_duplicate_groups: 5` in `configs/training.yaml`; they are reported as a warning, and any additional leakage fails the gate and stops training.

## Phase 2: experiment tracking with MLflow (implemented)

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

## Phase 3: evaluation and quality gate (implemented)

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

## Phase 4: model registry (implemented)

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

## Phase 5: inference API (implemented)

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

**How it was checked:** everything below the web layer was run directly: upload validation on valid JPEG/PNG (including RGBA, grayscale and palette images) and on empty, random, text, GIF, BMP, truncated, corrupted, tiny and oversized inputs; preprocessing shape, dtype and range; the model store with a stand-in loader (loading, a failed reload keeping the old model, a crash inside the model returning a generic 500, no model loaded returning 503); and response building. The service was then started against the real champion on the real setup and served predictions. The error cases, `/reload` protection and preprocessing parity with the training transform are covered by the automated tests, which pass on the real setup. **`[CONFIRM: python -m serving.check_parity on dataset test images, and /reload after a real promotion]`**

## Phase 6: Docker (implemented)

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

## Phase 7: tests and CI (implemented)

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

**How it was checked:** the unit tests that do not need FastAPI, torch or MLflow (43 test cases) were run, and they were checked against deliberately introduced bugs (a gate that lets any drop through, a wrong preprocessing scale, leakage that never fails, a removed hash check, NaN output accepted, circular configs not detected), all of which made tests fail. On the real setup the full suite passes (59 tests, including the API tests, the real exported model, preprocessing parity with the training transform and the real-MLflow registry operations). Running it against a real MLflow store found two genuine inconsistencies (this MLflow version returns model-version numbers as integers in some calls); both were fixed. **`[CONFIRM: the first run of the GitHub Actions workflow]`**

## Phase 8: monitoring (implemented)

**Why:** after deployment you should be able to see what the service is doing: how much traffic it gets, how often it refuses or fails, how fast it is, and whether its predictions look the way they should.

**Structured logs.** The service writes one JSON line per request to standard output, built only from an allow-list of fields: `timestamp`, `request_id`, `method`, `endpoint`, `status`, `latency_ms`, and for predictions `model_version`, `prediction`, `confidence`, `inference_ms`, and for failures `error_category`. Uploaded images, file names and headers are never logged. Successful health, readiness and metrics calls are logged at DEBUG so probes do not flood the log; client errors are WARNING and server errors ERROR. The request id in a log line is the same one returned in the `X-Request-ID` header and in error bodies, so one failing request can be traced.

```json
{"timestamp":"2026-10-10T18:29:04.195+00:00","event":"request","request_id":"d08d084e","method":"POST","endpoint":"/predict","status":200,"latency_ms":151.3,"model_version":"1","prediction":"Potato___healthy","confidence":0.99942,"inference_ms":136.2}
```

**Metrics.** `GET /metrics` returns counters and histograms in the Prometheus text format (no extra dependency; the format is simple enough to render directly, and a Prometheus server could scrape it unchanged):

| Metric | Meaning |
|---|---|
| `plant_requests_total{endpoint,status}` | all requests |
| `plant_predictions_total{label}` | successful predictions by predicted class (class distribution) |
| `plant_rejected_requests_total{reason}` | `/predict` requests refused because of the input (4xx): the rejection rate is this over the `/predict` request count |
| `plant_errors_total{category}` | server-side failures (5xx) |
| `plant_prediction_confidence` | histogram of the top-1 probability (confidence distribution) |
| `plant_predict_duration_seconds`, `plant_inference_duration_seconds` | latency histograms |
| `plant_model_loaded`, `plant_model_info{name,alias,version}` | which model version is being served |

For a readable view without Prometheus or Grafana:

```powershell
python -m serving.metrics_summary --url http://localhost:8000
```

```
Model served: plant-disease-classifier @champion version 1
/predict requests: 4   successful: 3   rejected: 1   server errors: 0
Rejection rate: 25.0%
Rejections by reason: unsupported_media_type 1
Predict latency: p50 ~ 150 ms, p95 ~ 190 ms (bucket estimates)
Confidence: mean 0.863; 33.3% of predictions below 0.7
Predicted classes: Tomato_healthy 3
```
(illustrative output from a unit-test fixture, not from a real run)

What this is for: a sudden rise in rejections points at a client or upload problem; a shift in the predicted-class distribution or in the share of low-confidence predictions is a signal that the incoming images differ from the training data (the model cannot tell a leaf of another crop from the ones it knows); latency and errors show the health of the service itself. The metrics live in the memory of the running process and reset on restart; there are no alerts, dashboards or long-term storage.

**How it was checked:** the metrics and logging code was unit-tested (counting rules, histogram consistency, label escaping and round trip, rendered text parsed back, the summary numbers, log fields limited to the allow-list with no file names or image bytes, log levels) and the tests were checked against deliberately introduced bugs. The API tests for `/metrics` and the logs run in the FastAPI test suite. On the real setup the full suite passes with the monitoring tests (69 tests), and against the running container `python -m serving.metrics_summary` reported, after one prediction, the served model version, 1 successful request with 0% rejections, latency estimates of about 112 ms (predict) and 88 ms (inference), and the predicted class. (That image was a `Tomato_Early_blight` test image, the model's weakest class, and the model gave it a confidence of only 0.555.)

## Phase 9: retraining and model promotion (implemented)

**Why:** a deployed model should be replaceable by a better one, but only when the evidence says it is better. This workflow retrains the existing classifier, runs the new version through the same evaluation and gate as every other version, registers it, compares it with the current champion and either promotes it or leaves the champion in place.

```
new images (optional) -> validate -> train -> evaluate -> quality gate -> register
                                                                             |
                      compare with the champion on the same test images  <---+
                                  |                      |
                              promote                 reject (champion stays)
```

```powershell
python -m retraining                                 # retrain the same recipe and compare with the champion
python -m retraining --new-data path\to\new_images   # add new labelled images (one sub-folder per class) to the training split
python -m retraining --config configs\experiments\lr_0005.yaml --name retrain-lr   # a different training recipe
```

Every run creates a named candidate (`--name`, default `retrain-<time>`) with its own checkpoint, MLflow run, evaluation folder and a decision record in `reports/retraining/<name>/decision.json`. Training takes as long as the original training (about an hour on a GPU; longer on a CPU).

**New data.** New images go into the **training split only**; validation and test stay exactly as they are, so every model version is judged on the same test images and the results are comparable. The batch is validated (readable, JPEG/PNG, not tiny), its class folders must already exist (a model with 15 fixed classes cannot learn a 16th by retraining), and an image is excluded when it is byte-identical to an image already in train/val/test (a duplicate, or a leak into the test set), filed under a different class than an identical existing image, or duplicated within the batch. What was accepted and excluded, and why, is written to `new_data_report.json`. The result is a derived dataset in `datasets/<name>/` (existing files hard-linked or copied, plus the accepted images), so the original `dataset/` is never modified, and the training run records the derived dataset's fingerprint.

**When is a candidate promoted?** First it must pass the quality gate against the reference model, otherwise it is not even registered. Then it is registered as a new version and compared with the champion, image by image, on the test set (`configs/promotion.yaml`):

| Rule | Why |
|---|---|
| it fixes more test images than it breaks, by at least `min_accuracy_gain` in accuracy | it must actually be better |
| the improvement is statistically significant (McNemar exact test on the images where the two models disagree, `alpha` 0.05) | with about 100 errors on 2,073 test images, a gain of a few images is routinely just noise |
| no class gets more than 0.10 worse in recall than under the champion | the same reasoning as the gate: one image in the 16-image `Potato___healthy` test class moves its recall by 0.0625 |

A candidate that fails is **rejected**: its version stays in the registry, tagged `promotion_decision=rejected` with the reasons, and the champion alias does not move. A promoted candidate becomes the champion; to ship it, run `python -m registry export` and rebuild the image (see Phase 6). The decision, the champion's and candidate's accuracy, the McNemar counts and p-value, and the reasons are stored as version tags, on the MLflow run and in `decision.json`.

**Limits to be aware of.** Retraining without new data repeats the same recipe on the same training split, so a result close to the champion is expected and it will normally be rejected; that exercises the rejection path honestly but is not evidence of improvement. Promotion needs a candidate that is genuinely better, which depends on better data or a better recipe. Retraining is a manual command, not part of CI: it needs the dataset and the local registry, and training does not belong in a per-commit workflow.

**How it was checked:** the statistics (McNemar against hand-computed p-values), the promotion rules (clear improvement, noise-level improvement, equal or worse, a class that gets much worse while the overall result is significantly better), the new-data ingestion (derived dataset layout, the original untouched, duplicates, label conflicts, unreadable and tiny files, unknown classes, an empty or already used target) and the workflow flow (promote, register but reject, stop at the gate, no champion, mismatching test sets, new data passed to training) were unit-tested with stand-in training and registry steps, and the tests were checked against deliberately introduced bugs. The real steps (training through the workflow, evaluation, registration, comparing against the live champion) have not been run yet. **`[CONFIRM: python -m pytest -q, then python -m retraining on the real setup, and the decision it reports]`**
