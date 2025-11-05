# AER850 – Project #2 (DCNN)
# Ahmed Negm | 501101640
# Step 5: Model Testing (3 required images)
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.preprocessing import image

ROOT = Path(__file__).parent
ARTIFACTS = ROOT / "artifacts"
OUT_DIR = ROOT / "predictions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1) the exact saved model produced by main.py
MODEL_PATH  = ARTIFACTS / "CNN_f32-64_d128_leaky_relu_relu_best.keras"
LABELS_PATH = ARTIFACTS / "class_indices.json"

# 2) the required test images (exact names per project)
IMG_SIZE = (500, 500)
TEST_IMGS = {
    "crack"        : ROOT / "Data" / "test" / "crack"        / "test_crack.jpg",
    "missing-head" : ROOT / "Data" / "test" / "missing-head" / "test_missinghead.jpg",
    "paint-off"    : ROOT / "Data" / "test" / "paint-off"    / "test_paintoff.jpg",
}

def preprocess(img_path: Path):
    """Load -> resize 500×500 -> scale to [0,1] -> add batch dimension."""
    img = image.load_img(img_path, target_size=IMG_SIZE)
    arr = image.img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0), np.array(img, dtype=np.uint8)

def main():
    #loading the  model and the label map
    assert MODEL_PATH.exists(), f"Missing model file: {MODEL_PATH}"
    model = keras.models.load_model(MODEL_PATH)

    with open(LABELS_PATH, "r") as f:
        class_indices = json.load(f)  # will show like this {'crack':0, 'missing-head':1, 'paint-off':2}
    idx_to_class = {v: k for k, v in class_indices.items()}

    #predicting each required image and save a figure
    for true_label, img_path in TEST_IMGS.items():
        assert img_path.exists(), f"Missing test image: {img_path}"
        x, img_arr = preprocess(img_path)
        probs = model.predict(x, verbose=0)[0]  # this is the softmax probabilities
        pred_idx = int(np.argmax(probs))
        pred_label = idx_to_class[pred_idx]

        #put the class probabilities in the canonical order
        lines = []
        for cls in ["crack", "missing-head", "paint-off"]:
            p = probs[class_indices[cls]] * 100.0
            lines.append(f"{cls.replace('-', ' ').title()}: {p:.1f}%")
        overlay = "\n".join(lines)

        plt.figure(figsize=(5.5, 4.5))
        plt.imshow(img_arr)
        plt.axis("off")
        plt.text(
            10, img_arr.shape[0] - 10, overlay,
            color="lime", fontsize=14, va="bottom", ha="left",
            bbox=dict(facecolor="black", alpha=0.35, pad=6),
        )
        plt.title(f"Actual: {true_label} | Predicted: {pred_label}", fontsize=12)
        out_path = OUT_DIR / f"pred_{true_label.replace(' ', '_')}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()

        print(f"{img_path.name:18s} | Actual: {true_label:13s} | "
              f"Predicted: {pred_label:13s} | "
              f"Probs [crack={probs[class_indices['crack']]:.3f}, "
              f"missing-head={probs[class_indices['missing-head']]:.3f}, "
              f"paint-off={probs[class_indices['paint-off']]:.3f}]")
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
