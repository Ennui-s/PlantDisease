# Model card: Plant Disease Classifier

## Model details
- **Task:** classify a photograph of a single leaf into one of 15 classes (crop + disease, or healthy) for pepper, potato and tomato.
- **Architecture:** MobileNetV2 (torchvision, ImageNet-pretrained, `IMAGENET1K_V1`) with the feature extractor **frozen**, and a new head: Dropout 0.3, then a linear layer to 15 classes. Output is softmax probabilities.
- **Framework:** PyTorch / torchvision. Input: RGB image resized to 224 x 224, values scaled to [0, 1], no normalisation, no augmentation.
- **Training:** Adam, learning rate 0.001, batch size 32, at most 12 epochs, early stopping on validation loss (patience 3), best checkpoint by validation accuracy. Hyperparameters were tuned by hand on validation accuracy and loss curves.
- **Version:** registry model `plant-disease-classifier`, version 1 (the original `best_model.pth`, file hash `496dafa6def52844...`), the current champion.

## Intended use
- Educational and portfolio demonstration of an ML lifecycle; decision support experiments on leaf photographs similar to the training images.
- **Not for:** other crops or plant parts (fruit, stem), photographs taken in the field without separate validation, or any decision where a wrong answer causes real harm (for example choosing a treatment) without expert review.

## Data
- PlantVillage (Kaggle): 20,639 files in 15 classes, all 256 x 256, photographed leaves on controlled backgrounds. Split per class 70/20/10 into 14,441 train, 4,125 validation and 2,073 test files (one extensionless file is skipped by the loader, so 20,638 images are loaded).
- Class sizes range from 152 (`Potato___healthy`) to 3,209 (`Tomato__Tomato_YellowLeaf__Curl_Virus`), a ratio of 21.1.
- **Known data issues** (found by the platform's validation): 14 groups of byte-identical duplicates in the raw data; 5 identical images shared between train and validation/test (2 in test, 3 in validation), all `Tomato_Late_blight`; one RGBA screenshot in `Pepper__bell___healthy`. Only exact duplicates are detected, not near-duplicates such as several photographs of the same leaf.

## Performance (test split, 2,073 images)
| Metric | Value |
|---|---|
| Accuracy | 0.9518 (1,973 correct) |
| Macro precision / recall / F1 | 0.9552 / 0.9523 / 0.9534 |
| Weighted F1 | 0.9515 |

At this test-set size the 95% sampling noise on accuracy is about 0.9 percentage points. Without the 2 test images affected by leakage the accuracy would be between 95.17% and 95.27%.

| Class | Precision | Recall | F1 | Test images |
|---|---|---|---|---|
| Pepper bell Bacterial spot | 1.000 | 0.910 | 0.953 | 100 |
| Pepper bell healthy | 0.943 | 1.000 | 0.971 | 148 |
| Potato Early blight | 0.961 | 0.980 | 0.971 | 101 |
| Potato Late blight | 0.980 | 0.980 | 0.980 | 101 |
| Potato healthy | 1.000 | 1.000 | 1.000 | 16 |
| Tomato Bacterial spot | 0.936 | 0.967 | 0.952 | 213 |
| Tomato Early blight | 0.849 | 0.832 | 0.840 | 101 |
| Tomato Late blight | 0.927 | 0.937 | 0.932 | 191 |
| Tomato Leaf Mold | 0.958 | 0.948 | 0.953 | 96 |
| Tomato Septoria leaf spot | 0.952 | 0.882 | 0.915 | 178 |
| Tomato Spider mites (two-spotted) | 0.959 | 0.976 | 0.968 | 168 |
| Tomato Target Spot | 0.906 | 0.894 | 0.900 | 141 |
| Tomato Yellow Leaf Curl Virus | 0.981 | 0.984 | 0.983 | 321 |
| Tomato mosaic virus | 1.000 | 1.000 | 1.000 | 38 |
| Tomato healthy | 0.976 | 0.994 | 0.985 | 160 |

Source: the platform's reference evaluation (`configs/reference_metrics.json`), which reproduced the originally recorded accuracy exactly. The full confusion matrix and Grad-CAM images are produced by `python -m evaluation.evaluate`. See [failure_analysis.md](failure_analysis.md) for where the errors are.

## Limitations and caveats
- **Controlled imagery.** All data are leaf photographs on plain backgrounds. Performance on field photographs (lighting, clutter, several leaves, occlusion) is unknown and likely lower.
- **Always answers.** The model has 15 outputs and no "none of these" option. A photograph of anything else still receives a prediction, and the softmax probability reported as `confidence` is not a calibrated certainty.
- **Three crops only.** Other crops need new data and a new model.
- **Weak spots.** The lowest recalls are `Tomato Early blight` (0.832), `Tomato Septoria leaf spot` (0.882) and `Tomato Target Spot` (0.894).
- **Test set reuse.** The same test set is used to evaluate every version, apply the quality gate and decide promotions. Repeated selection against one test set can make later versions look slightly better than they are; there is no separate untouched final hold-out.
- **Fine-grained symptoms** (for example spider mites) depend on image resolution.
- **Not assessed:** robustness to image corruption and adversarial inputs, calibration, performance across devices or regions.

## Ethical and practical considerations
Mistakes have agricultural consequences: a missed disease (13 of 1,749 diseased test images, 0.7%, were predicted healthy) delays treatment, and a wrong disease name can lead to the wrong treatment. The model supports, and does not replace, inspection by someone qualified.

## Updating the model
New versions go through `python -m retraining`: validation, training, evaluation, quality gate, registration, and a paired comparison with the champion on the same test images. A version becomes the champion only if the improvement is statistically significant and no class gets much worse. Every version carries its lineage (data fingerprint, git commit, metrics, gate outcome) as registry tags.
