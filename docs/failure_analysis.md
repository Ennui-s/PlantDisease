# Failure analysis

What goes wrong, where, and what is and is not known. Numbers come from the test split (2,073 images; 100 errors) and from the project's own runs.

## 1. Where the model makes mistakes

Top confusions (true class -> predicted class, number of test images):

| Count | True class | Predicted as |
|---|---|---|
| 8 | Tomato Late blight | Tomato Early blight |
| 7 | Pepper bell Bacterial spot | Pepper bell healthy |
| 6 | Tomato Target Spot | Tomato Spider mites |
| 6 | Tomato Septoria leaf spot | Tomato Bacterial spot |
| 5 | Tomato Yellow Leaf Curl Virus | Tomato Bacterial spot |
| 5 | Tomato Septoria leaf spot | Tomato Target Spot |
| 5 | Tomato Early blight | Tomato Late blight |
| 4 | Tomato Target Spot | Tomato Septoria leaf spot |

- **The errors cluster.** 43 of the 100 errors are confusions among five tomato foliar classes (early blight, late blight, Septoria leaf spot, target spot, spider mites). The weakest classes by recall are `Tomato Early blight` (0.832), `Tomato Septoria leaf spot` (0.882) and `Tomato Target Spot` (0.894). That these diseases produce similar-looking lesions is a plausible explanation; it has not been tested (the Grad-CAM images show where the model looks, but they do not prove why it errs).
- **Missed disease versus false alarm.** 13 of 1,749 diseased test images (0.7%) were predicted as a healthy class, and 1 of 324 healthy images (0.3%) was predicted as diseased. Seven of the 13 are `Pepper bell Bacterial spot` predicted as `Pepper bell healthy` (7% of that class). In an agricultural setting, the missed-disease direction is the more costly one.
- **Small classes carry the noise.** `Potato___healthy` has 16 test images; one image changes its recall by 0.0625. Per-class numbers for the small classes should be read with that in mind.

## 2. Data problems that affect trust in the numbers

| Problem | Evidence | Effect |
|---|---|---|
| Train/test leakage | 5 byte-identical images in both train and val/test (2 in test) | accuracy would be 95.17% to 95.27% without the 2 affected test images; accepted and bounded, not removed |
| Duplicates in the raw data | 14 groups of identical images | the cause of the leakage; the retraining workflow now excludes images identical to existing ones |
| Near-duplicates not detected | validation only compares file hashes | if several photographs of the same leaf are spread across splits, the test score may be optimistic; not measured |
| Controlled imagery | all PlantVillage photographs | field performance unknown |
| Class imbalance | 152 to 3,209 images per class (21x) | small classes have noisy metrics; no rebalancing was done |
| Repeated use of one test set | gate, comparison and promotion all use it | later versions can look slightly better than they are |

## 3. Behaviour on unfamiliar input

- The model always returns one of 15 classes. There is no out-of-distribution detection, and the confidence is an uncalibrated softmax value.
- One observation from the running service: a validation image of the weakest class (`Tomato Early blight`) was predicted correctly but with a confidence of only 0.555, the second choice being `Tomato Spider mites` at 0.38. This is a single example, not a measurement, but it is the kind of case the confidence histogram (`plant_prediction_confidence`) and the "share of predictions below 0.7" line of the metrics summary are meant to surface.
- Monitoring can show that the class mix or the confidence distribution has shifted; it cannot say whether predictions are correct, because the service receives no labels.

## 4. Failures encountered while building the platform

| What happened | How it was found | Resolution |
|---|---|---|
| Hidden leakage between splits | the data validation run on the real dataset | quantified, accepted explicitly with a limit of 5, documented |
| A 1-epoch model is much worse (accuracy 0.906 vs 0.952) | the quality gate rejected it as intended | confirms the gate does its job |
| MLflow returned model-version numbers as integers in some calls and strings in others (two separate places) | the test against a real MLflow registry; a stand-in registry had hidden it | versions are now always strings |
| API tests were silently skipped | a missing `httpx` dependency; the skip reason showed only with `-rs` | dependency installed; the lesson is to read skip reasons |
| Windows-specific registry paths could not be used in a Linux container | anticipated before building the image | the image bakes in an exported model |
| The first CI runs failed: `requirements-test.txt` had never been committed, and lint found a redefined import in a test (plus further findings under a stricter local ruff setup) | the GitHub Actions log and a local `ruff check` | files committed; code fixed (narrowed exceptions, modern FastAPI parameter style, one test import); CI now passes |
| A test that appeared to cover the per-class promotion rule did not | a deliberately injected bug survived | the test was rewritten so only that rule can fail it |

## 5. Operational risks that remain

- **Single process, in memory.** Metrics reset on restart, predictions are handled one at a time, and there is no autoscaling, alerting or dashboard.
- **No authentication.** The API is open. `/reload` exists only when an admin token is set. Uploads are size-limited by header and by reading, but a chunked upload without a length is only limited by the read.
- **Model files are Python pickles.** MLflow warns that loading CloudPickle models can execute code; only load models you produced.
- **The registry is local.** There is no shared or remote registry, no access control and no backup of `mlflow.db`.
- **Retraining is untested on real data end to end** (see the README status section), and a promotion changes the live model only after an export and an image rebuild.
- **No feedback loop.** There is no way for users to report a wrong prediction, so true accuracy in use is unmeasured.

## 6. What would be done next
Near-duplicate detection and a re-split by leaf or source; a separate untouched final test set; calibration and an "unknown" option (or an out-of-distribution score); field-condition data; shared registry and remote storage; authentication and alerting; a label-feedback path to measure live accuracy.
