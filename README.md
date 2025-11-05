1) Overview

This project builds and compares two deep convolutional neural networks (DCNNs) to classify aircraft skin defects from 500×500 RGB images. I implement a Baseline (2 Conv-Pool blocks) and a Deeper model (3 Conv-Pool blocks), perform targeted hyperparameter studies (activations, depth, dense width, learning rate), and select the best model using EarlyStopping on validation accuracy. I report learning curves, confusion matrix, and a classification report, then run test-time predictions with probability overlays on three required images.

Final selected model: LeakyReLU in conv blocks + ReLU in dense (filters 32–64, Dense=128, Dropout=0.30, Adam 1e-3).
Validation: ~0.733 accuracy, macro-F1 ≈ 0.710 at EarlyStopping epoch ≈ 11 (gap ≈ +0.026).
Known edge case: crack ↔ paint-off under strong highlights.

.
├─ main.py                   # Steps 1–4: data, models, hyperparams, evaluation, figures
├─ step5_test.py             # Step 5: load saved model + predict 3 required test images
├─ artifacts/                # (created) saved model + class_indices.json
│   ├─ CNN_f32-64_d128_leaky_relu_relu_best.keras
│   └─ class_indices.json
├─ figures/                  # (created) learning curves + confusion matrix + reports
│   ├─ figure2_model_performance_*.png
│   ├─ confusion_matrix_*.png
│   └─ classification_report_*.txt
├─ predictions/              # (created) probability-overlay PNGs for test images
│   ├─ pred_crack.png
│   ├─ pred_missing-head.png
│   └─ pred_paint-off.png
├─ Data/                     # dataset root (see layout below)
│   ├─ train/
│   │   ├─ crack/          ...
│   │   ├─ missing-head/   ...
│   │   └─ paint-off/      ...
│   ├─ valid/
│   │   ├─ crack/          ...
│   │   ├─ missing-head/   ...
│   │   └─ paint-off/      ...
│   └─ test/
│       ├─ crack/test_crack.jpg
│       ├─ missing-head/test_missinghead.jpg
│       └─ paint-off/test_paintoff.jpg
└─ requirements.txt          # minimal deps (optional; see below)

3) Environment & setup
   pip install -r requirements.txt

4) Data layout
   Data/
├─ train/
│  ├─ crack/          *.jpg...
│  ├─ missing-head/   *.jpg...
│  └─ paint-off/      *.jpg...
├─ valid/
│  ├─ crack/          *.jpg...
│  ├─ missing-head/   *.jpg...
│  └─ paint-off/      *.jpg...
└─ test/
   ├─ crack/          test_crack.jpg
   ├─ missing-head/   test_missinghead.jpg
   └─ paint-off/      test_paintoff.jpg


5) How to run
A) Steps 1–4 (train, compare models, evaluate)
python main.py


Switching variants: in main.py, set CFG to one of:

BASELINE_CFG (ReLU/ReLU, 32–64, Dense=128, LR=1e-3)

LEAKY_CONV_CFG (final: LeakyReLU/ReLU, 32–64, Dense=128, LR=1e-3)

ELU_DENSE_CFG (ReLU/ELU, 32–64, Dense=128, LR=1e-3)

DEEPER_CFG (ReLU/ReLU, 32–64–128, Dense=128, LR=1e-3)

Outputs:

figures/figure2_model_performance_<model>.png (accuracy & loss curves)

figures/confusion_matrix_<model>.png

figures/classification_report_<model>.txt

artifacts/<model>_best.keras and artifacts/class_indices.json

B) Step 5 (test-time predictions with overlays)
python step5_test.py


Inputs: loads artifacts/CNN_f32-64_d128_leaky_relu_relu_best.keras and the 3 test images.

Outputs: overlay images under predictions/:

pred_crack.png, pred_missing-head.png, pred_paint-off.png
(title shows Actual vs Predicted; green box shows per-class Softmax %)
