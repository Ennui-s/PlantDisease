# Demonstration walkthrough

A 10-minute tour that shows the whole lifecycle with real commands. Run everything from the repository root. Outputs marked **real** were produced by actual runs of this project; the others describe what you should see.

## The story in two sentences
"I built a plant disease classifier (95.2% test accuracy). Then I engineered a platform around it that validates the data, tracks and gates every training run, versions the models, serves the current champion from a container, monitors it, and only replaces the champion when a new model is significantly better."

## Steps

**1. The existing model, evaluated by the platform**
```
python -m evaluation.evaluate --model best_model.pth
```
Real: `Test accuracy: 0.9518  macro F1: 0.9534 (2073 images)`, lowest recall `Tomato_Early_blight 0.832`; quality gate PASSED (it is the reference). Point out: the number reproduces the originally recorded 0.9517607...

**2. Training does not consume arbitrary data**
```
python -m validation.validate_data --stage all --verbose
```
Real: 20,639 files, 15 classes, all 256 x 256; one non-image file; 14 duplicate groups; **5 identical images shared between train and val/test** (all `Tomato_Late_blight`). Point out: this was invisible before; it is quantified (accuracy would be 95.17% to 95.27%), accepted explicitly with a limit, and any new leakage would stop training.

**3. A trained model is not automatically a deployable model**
```
python -m evaluation.evaluate --model checkpoints/smoke_test.pth
```
Real: the 1-epoch model gets accuracy 0.9059 and the gate says `FAILED` (accuracy and macro F1 drops, and recall for `Tomato_Early_blight` and `Tomato__Tomato_mosaic_virus`). Exit code 1.

**4. Experiments are recorded and comparable**
```
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Show: the runs, their parameters, the dataset fingerprint tag, the per-epoch metrics, the artifacts. (Comparing the baseline, `lr_0005` and `batch_64` experiments needs those three runs to be done; they are not done yet.)

**5. Versions and the champion**
```
python -m registry show
python -m registry check
```
Real: `plant-disease-classifier  aliases: {'champion': 1}`, version 1 with test accuracy 0.9518, macro F1 0.9534, gate passed; and the check loads the model through the alias only and returns a `(1, 15)` probability vector summing to 1.

**6. Clone, build, run, predict**
```
python -m registry export
docker build -t plant-disease-api .
docker run --rm -p 8000:8000 plant-disease-api
curl.exe http://localhost:8000/ready
curl.exe http://localhost:8000/model
curl.exe -F "file=@<a test image>.JPG" http://localhost:8000/predict
```
Real: a `Potato___healthy` test image returned `{"prediction":"Potato___healthy","confidence":0.99942,...,"model":{"version":"1"},"latency_ms":{"inference_ms":136.24,...}}`. Point out: the response says which model version answered.

**7. Bad input is handled, not crashed on**
```
curl.exe -F "file=@README.md" http://localhost:8000/predict
curl.exe -X POST http://localhost:8000/predict
```
Expected (covered by the automated API tests): JSON errors `unsupported_media_type` (415) and `malformed_request` (422), each with a `request_id`, and no stack trace.

**8. The service can be observed**
```
python -m serving.metrics_summary --url http://localhost:8000
```
Real (after one request): model version served, 1 successful `/predict`, 0% rejections, latency estimates of about 112 ms (predict) and 88 ms (inference), confidence mean 0.555, predicted class. Point out: the image in that example was the model's weakest class and the low confidence is visible in the metrics. Also show a JSON log line and `curl.exe http://localhost:8000/metrics`.

**9. The engineering is tested**
```
python -m pytest -q
```
Real: 69 tests pass locally, including the real exported model, preprocessing identical to the training transform, the API over HTTP and a real temporary MLflow registry. CI (GitHub Actions) runs lint, the tests and a Docker build-and-smoke-test on every push.

**10. Replacing the champion safely**
```
python -m retraining --name retrain-demo
```
Expected: the full workflow runs and prints a decision. Retraining the same recipe on the same data should normally be **rejected** (not significantly better), with the reasons and the McNemar p-value written to `reports/retraining/retrain-demo/decision.json`. Point out the rule: promotion needs a significant, paired improvement on the same test images and no class getting much worse. (This end-to-end run has not been done yet; the decision logic is unit-tested.)

## Screenshots worth adding to `docs/img/`
None have been captured yet. Useful ones: the MLflow runs table; one run's page (parameters, tags, metrics, artifacts); the registered model page with the `champion` alias; the FastAPI `/docs` page; the output of `metrics_summary`; the green GitHub Actions run. Reference them from the README once they exist.

## If asked about limits
Be direct about them: all photographs are from a controlled dataset; the model always answers with one of 15 classes; the test set is reused for model selection; near-duplicates were not checked; monitoring cannot see correctness. See [failure_analysis.md](failure_analysis.md).
