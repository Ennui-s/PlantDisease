# Engineering decisions

Each entry: the situation, what was decided, and what it costs. They are written so each can be defended in conversation.

**1. Wrap the classifier, do not change it.** The goal is to show a lifecycle around an existing model. The training loop is moved into a function verbatim; configuration overrides mutate the shared `CONFIG` dictionary, which the classifier already reads at call time. *Cost:* some platform code has to work around the classifier's habits (for example `PlantDiseaseModel` always asks torchvision for ImageNet weights, so the packaged model builds it with `weights=None` and loads the checkpoint over it, which avoids a download at serving time).

**2. Keep the existing split and accept five known leaked images.** Validation found 5 byte-identical images shared between train and val/test (all `Tomato_Late_blight`; 2 in test). Re-splitting would change the recorded result. The effect on the headline is bounded: without the 2 affected test images, accuracy would be between 95.17% and 95.27%. The known count is accepted explicitly (`accepted_cross_split_duplicate_groups: 5`); any further leakage fails the gate. *Cost:* the split is not perfectly clean, and only exact duplicates are detected, not near-duplicates.

**3. Quality-gate margins come from the test set, not from taste.** On 2,073 images at about 95% accuracy the 95% sampling noise is about 0.9 points, so an accuracy or F1 drop up to 0.01 is indistinguishable from noise. The smallest test class (`Potato___healthy`) has 16 images, so one image moves its recall by 0.0625, hence a per-class margin of 0.10. *Cost:* the gate cannot detect a real regression smaller than the noise.

**4. The reference is the original model, evaluated by the platform.** The gate needs a fixed yardstick; the model the project already had is the honest one. It passes its own gate by definition, and the README says so.

**5. Registry with an alias, not stages.** MLflow's Staging/Production stages are deprecated. The service asks for `models:/plant-disease-classifier@champion`, so a promotion needs no code change. Registration is refused unless an evaluation of exactly that file (hash match) passed the gate, so a different or retrained checkpoint cannot ride on an old evaluation.

**6. The model is packaged as an MLflow pyfunc with weights only.** The package holds the weights, the class names (so serving does not need the dataset folders), the model code and a signature. *Alternatives:* MLflow's PyTorch flavor pickles the whole module and is sensitive to PyTorch's loading rules; a raw checkpoint file has no identity or contract. *Cost:* a custom wrapper, and MLflow warns about type hints and CloudPickle.

**7. The Docker image bakes in an exported champion.** The local registry database stores artifact locations that belong to one machine (on Windows they may not resolve inside a Linux container). Exporting the champion makes the image self-contained and gives `clone, build, run` with no volume or secret. *Cost:* shipping a new champion means exporting and rebuilding.

**8. Preprocessing is re-implemented with Pillow and numpy, and checked against training.** It is exactly torchvision's `Resize` + `ToTensor` for this pipeline, without importing torchvision in the request path. A test compares it with the training transform on real inputs, because train/serve skew is silent when it happens.

**9. Promotion requires statistical significance.** A candidate that is a few images better on a 2,073-image test set is usually just lucky. The comparison is paired (McNemar exact test on the images where champion and candidate disagree), plus a per-class guard. *Cost:* it will rarely promote; a genuine promotion needs genuinely better data or a better recipe.

**10. New data goes into the training split only.** The test set must stay fixed for versions to be comparable. Images identical to anything already in train/val/test are excluded, and a batch cannot introduce new classes (the model has 15 fixed outputs).

**11. Metrics and logs are hand-rolled and small.** One JSON log line per request from an allow-list of fields (so images and file names cannot be logged) and a Prometheus-format `/metrics`, with no extra dependency. A `metrics_summary` command makes the numbers readable without Prometheus. *Cost:* metrics live in process memory and reset on restart; no alerting or dashboards.

**12. Retraining is a manual command, not part of CI.** It needs the dataset and the local registry, and training does not belong in a per-commit workflow. CI runs lint, tests and a Docker build-and-smoke-test; nothing is trained.

**13. Tests exercise the real thing wherever possible.** The suite includes the real exported model, preprocessing parity with the training transform, the API through HTTP, and a real temporary MLflow registry. The real-registry test found two genuine bugs that a stand-in registry could not (this MLflow version returns model-version numbers as integers in some calls and strings in others).

**14. Honest status over polish.** Anything not run is marked `[CONFIRM]`; results in the README are the numbers the runs produced. Where a unit test was the only evidence, the README says so.
