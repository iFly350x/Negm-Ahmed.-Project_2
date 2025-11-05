# AER850 – Project #2 (DCNN)
# Ahmed Negm | 501101640

import os
import json
from pathlib import Path
from typing import Tuple


import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Global Parameters

INPUT_SHAPE: Tuple[int, int, int] = (500, 500, 3)   # this is (H, W, C) defined by the project
IMG_SIZE = INPUT_SHAPE[:2]
BATCH_SIZE = 32                                     # defined value by project
N_CLASSES = 3                                       # we have 3 classes, crack, missing-head, paint-off
EPOCHS = 30

# keeping the randomness reproducible
np.random.seed(42)
keras.utils.set_random_seed(42)


# Step 1 — DATA PROCESSING 

def get_data_dirs() -> Tuple[Path, Path, Path]:
    """
    Resolve relative folders exactly like the assignment (./Data/train, ./Data/valid, ./Data/test).
    I keep it relative so the repo runs on any machine without path edits.
    """
    root = Path(__file__).parent
    data_root = root / "Data"
    train_dir = data_root / "train"
    valid_dir = data_root / "valid"
    test_dir  = data_root / "test"
    return train_dir, valid_dir, test_dir


def make_image_generators(
    img_size: Tuple[int, int] = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
):
    """
    ImageDataGenerator is what the project says to use explicitly.
    Dr Reza emphasized in class: do augmentation only on TRAIN; keep VALID clean.

    Augmentations (light but useful for generalization):
      - rescale: normalize to [0,1] (same idea as our MLP scaling)
      - shear_range, zoom_range: exactly the ones listed in the project
    """
    train_aug = ImageDataGenerator(
        rescale=1.0 / 255.0,
        shear_range=0.15,         # starting small,  we can tune in Step 3
        zoom_range=0.15,
        # we could add horizontal_flip/rotation later if needed
    )

    valid_aug = ImageDataGenerator(
        rescale=1.0 / 255.0       # validation should reflect “real” data distribution
    )

    train_dir, valid_dir, _ = get_data_dirs()

    train_gen = train_aug.flow_from_directory(
        directory=str(train_dir),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",   # one-hot labels for softmax(3)
        shuffle=True
    )

    valid_gen = valid_aug.flow_from_directory(
        directory=str(valid_dir),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False               # common choice for validation monitoring as per tutorials given
    )

    #mapping for the report to helps interpret predictions
    class_indices = train_gen.class_indices  # this should produce {'crack':0, 'missing-head':1, 'paint-off':2}
    return train_gen, valid_gen, class_indices


# Step 2 — NEURAL NETWORK ARCHITECTURE (two DCNNs)

def build_cnn_baseline(
    input_shape: Tuple[int, int, int] = INPUT_SHAPE,
    n_classes: int = N_CLASSES,
    filters_1: int = 32,
    filters_2: int = 64,
    k1: int = 3,
    k2: int = 3,
    pool_size: Tuple[int, int] = (2, 2),
    dense_units: int = 128,
    dropout_rate: float = 0.30,
) -> keras.Model:
    """
    Baseline CNN from the lecture diagram: [Conv→ReLU→Pool] × 2 → Flatten → Dense → Dropout → Softmax(3).
    Convs use padding='same' so spatial size only changes at pooling (just like on the slides).
    """
    model = keras.Sequential(
        [
            layers.Conv2D(filters_1, (k1, k1), padding="same", activation="relu", input_shape=input_shape),
            layers.MaxPooling2D(pool_size=pool_size),

            layers.Conv2D(filters_2, (k2, k2), padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=pool_size),

            layers.Flatten(),
            layers.Dense(dense_units, activation="relu"),
            layers.Dropout(dropout_rate),  # “safety valve” against overfitting we discussed in class
            layers.Dense(n_classes, activation="softmax"),
        ],
        name="cnn_baseline",
    )
    return model


def build_cnn_deeper(
    input_shape: Tuple[int, int, int] = INPUT_SHAPE,
    n_classes: int = N_CLASSES,
    filters: Tuple[int, int, int] = (32, 64, 128),
    ks: Tuple[int, int, int] = (3, 3, 3),
    pool_size: Tuple[int, int] = (2, 2),
    dense_units: int = 256,
    dropout_rate: float = 0.35,
) -> keras.Model:
    """
    Deeper CNN (more capacity): three conv blocks. Still exactly the same pattern from lecture.
    This is our “second variation” required by the project.
    """
    f1, f2, f3 = filters
    k1, k2, k3 = ks

    model = keras.Sequential(
        [
            layers.Conv2D(f1, (k1, k1), padding="same", activation="relu", input_shape=input_shape),
            layers.MaxPooling2D(pool_size=pool_size),

            layers.Conv2D(f2, (k2, k2), padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=pool_size),

            layers.Conv2D(f3, (k3, k3), padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=pool_size),

            layers.Flatten(),
            layers.Dense(dense_units, activation="relu"),
            layers.Dropout(dropout_rate),
            layers.Dense(n_classes, activation="softmax"),
        ],
        name="cnn_deeper",
    )
    return model


def compile_for_multiclass(model: keras.Model, lr: float = 1e-3) -> keras.Model:
    """
    Same compile recipe we’ve been using:
    - Optimizer: Adam
    - Loss: categorical_crossentropy (because ImageDataGenerator gives one-hot labels)
    - Metric: accuracy
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# Convenience: quick sanity run (NO TRAINING YET)

def preview_pipeline():
    """
    Quick sanity check
    - Build generators to prints class mapping and sample counts
    - Build both models and print summaries for the report later
    """
    train_gen, valid_gen, class_indices = make_image_generators()
    print("\nClass indices (keeping for report):", class_indices)
    print(f"Train samples: {train_gen.samples} | Valid samples: {valid_gen.samples}")

    base = compile_for_multiclass(build_cnn_baseline())
    deep = compile_for_multiclass(build_cnn_deeper())

    print("\n=== Baseline CNN Summary ===")
    base.summary()
    print("\n=== Deeper CNN Summary ===")
    deep.summary()

    # In Step 3 i will add: early stopping, fit(), and the accuracy/loss plots.


# 3.1 Activation helper (as discussed in lecture)
def get_activation(name: str):
    if name == "relu":
        return "relu"                      
    if name == "leaky_relu":
        return layers.LeakyReLU(alpha=0.1)  # this is because small negative slope avoids "dead" ReLUs
    if name == "elu":
        return "elu"
    raise ValueError(f"Unknown activation: {name}")

# 3.2 Build CNN with knobs we want to test (filters, activations, dense width)
def build_cnn(conv_act="relu",
              dense_act="relu",
              filters=(32, 64),
              dense_units=128,
              dropout=0.30,
              input_shape=(500, 500, 3)):
    """
    Block pattern from the slides:
      [Conv(3x3, stride=1, padding='same', act) -> MaxPool(2x2)] x (#filters blocks)
       -> Flatten -> Dense(dense_units, act) -> Dropout -> Dense(3, softmax)
    """
    # i am making a Keras-safe model name (no spaces/parentheses/commas) 
    filt_str = "-".join(str(f) for f in filters)              
    safe_name = f"CNN_f{filt_str}_d{dense_units}_{conv_act}_{dense_act}"  # output will look like CNN_f32-64_d128_relu_relu

    mdl = models.Sequential(name=safe_name)
    for i, f in enumerate(filters):
        if i == 0:
            mdl.add(layers.Conv2D(f, (3,3), padding="same", strides=1,
                                  activation=get_activation(conv_act),
                                  input_shape=input_shape))
        else:
            mdl.add(layers.Conv2D(f, (3,3), padding="same", strides=1,
                                  activation=get_activation(conv_act)))
        mdl.add(layers.MaxPooling2D((2,2)))  # halves H & W; depth stays the same like L8

    mdl.add(layers.Flatten())
    mdl.add(layers.Dense(dense_units, activation=get_activation(dense_act)))
    mdl.add(layers.Dropout(dropout))
    mdl.add(layers.Dense(3, activation="softmax"))
    return mdl


# 3.3 Single experiment (train exactly ONE config)
def run_single_experiment(cfg, train_gen, val_gen, epochs=EPOCHS, early_stop_cb=None):
    """
    cfg = (conv_act, dense_act, filters_tuple, dense_units, lr)
    Example: ("relu", "relu", (32, 64), 128, 1e-3)
    """
    K.clear_session()
    conv_act, dense_act, filt, dense_units, lr = cfg

    model = build_cnn(conv_act=conv_act,
                      dense_act=dense_act,
                      filters=filt,
                      dense_units=dense_units,
                      dropout=0.30,
                      input_shape=INPUT_SHAPE)

    model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    print(f"\n>>> Training {model.name} | conv_act={conv_act} | dense_act={dense_act} "
          f"| filters={filt} | dense_units={dense_units} | lr={lr}")

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[early_stop_cb] if early_stop_cb else None,
        verbose=1
    )

    #picking the epoch with best validation accuracy
    best_epoch = int(np.argmax(history.history["val_accuracy"]))
    result = {
        "model": model.name,
        "conv_act": conv_act,
        "dense_act": dense_act,
        "filters": filt,
        "dense_units": dense_units,
        "lr": lr,
        "best_val_acc": float(history.history["val_accuracy"][best_epoch]),
        "best_val_loss": float(history.history["val_loss"][best_epoch]),
        "best_epoch": best_epoch + 1
    }

    print("\n=== Result ===")
    print(result)
    return result, history, model

# 3.4 plotting results
def plot_history(history, title=""):
    import matplotlib.pyplot as plt
    acc, val_acc = history.history["accuracy"], history.history["val_accuracy"]
    loss, val_loss = history.history["loss"], history.history["val_loss"]

    plt.figure(figsize=(6,4))
    plt.plot(acc, label="Train Acc")
    plt.plot(val_acc, label="Val Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title(f"Accuracy vs Epoch {title}")
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

    plt.figure(figsize=(6,4))
    plt.plot(loss, label="Train Loss")
    plt.plot(val_loss, label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(f"Loss vs Epoch {title}")
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

# 3.5 this is the expiremens I will run (one-by-one)
BASELINE_CFG   = ("relu",       "relu", (32, 64),       128, 1e-3)  # Step 2 baseline
LEAKY_CONV_CFG = ("leaky_relu", "relu", (32, 64),       128, 1e-3)  # swap conv activation
ELU_DENSE_CFG  = ("relu",       "elu",  (32, 64),       128, 1e-3)  # swap dense activation
DEEPER_CFG     = ("relu",       "relu", (32, 64, 128),  128, 1e-3)  # add 3rd conv block
DEEPER_SMOOTH  = ("leaky_relu", "elu",  (32, 64, 128),  128, 5e-4)  # deeper + acts + lower LR
WIDER_DENSE    = ("relu",       "relu", (32, 64),       256, 5e-4)  # wider classifier


# Step 4 — Model Evaluation (Figure 2 style)


def step4_model_evaluation(history, model, valid_gen, out_dir="figures"):
    """
    plots Training vs Validation Accuracy/Loss (like Figure 2 in the handout),
    prints a short summary so i can discuss in lab (best epoch, generalization gap),
    and saves the figure for the report.
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1)  learning curves from Keras History like what we monitored in lecture
    acc      = history.history["accuracy"]
    val_acc  = history.history["val_accuracy"]
    loss     = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs   = np.arange(1, len(acc) + 1)

    # 2) Identify best validation epoch the one early stopping cares about
    best_idx = int(np.argmax(val_acc))
    best_epoch = best_idx + 1

    # 3) Evaluate once more on validation to print a clean number
    val_loss_eval, val_acc_eval = model.evaluate(valid_gen, verbose=0)

    # 4) Small  summary (talk about over/under-fitting)
    gen_gap = acc[best_idx] - val_acc[best_idx]   # train minus val at peak val
    print("\n=== Step 4: Model Evaluation Summary ===")
    print(f"Model name              : {model.name}")
    print(f"Best val epoch          : {best_epoch}")
    print(f"Best val accuracy       : {val_acc[best_idx]:.4f}")
    print(f"Best val loss           : {val_loss[best_idx]:.4f}")
    print(f"Train acc @ best val    : {acc[best_idx]:.4f}")
    print(f"Generalization gap      : {gen_gap:+.4f} (train - val)")
    print(f"Final val eval (fresh)  : acc={val_acc_eval:.4f}, loss={val_loss_eval:.4f}")

    # 5) Figure 2: side-by-side curves 
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, acc,     label="Training Accuracy")
    axes[0].plot(epochs, val_acc, label="Validation Accuracy")
    axes[0].set_title("Training and Validation Accuracy")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].grid(True); axes[0].legend(loc="lower right")

    axes[1].plot(epochs, loss,     label="Training Loss")
    axes[1].plot(epochs, val_loss, label="Validation Loss")
    axes[1].set_title("Training and Validation Loss")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].grid(True); axes[1].legend(loc="upper right")

    plt.tight_layout()

    # saves a PNG so i can use it in the report as Figure 2
    fig_path = os.path.join(out_dir, f"figure2_model_performance_{model.name}.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {fig_path}")

    return {
        "best_epoch": best_epoch,
        "best_val_acc": float(val_acc[best_idx]),
        "best_val_loss": float(val_loss[best_idx]),
        "train_acc_at_best": float(acc[best_idx]),
        "gen_gap": float(gen_gap),
        "val_eval_acc": float(val_acc_eval),
        "val_eval_loss": float(val_loss_eval),
        "figure_path": fig_path
    }

def step4_confusion_and_f1(model, valid_gen, class_indices, out_dir="figures"):
    """
    Computes confusion matrix + F1 on the validation set, prints a classification report,
    and saves a confusion-matrix figure + report file for the report appendix.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    # 1) Ground truth and predictions (order matches generator because shuffle=False)
    y_true = valid_gen.classes                                # shape: (N,)
    y_prob = model.predict(valid_gen, verbose=0)              # shape: (N, C)
    y_pred = np.argmax(y_prob, axis=1)

    # 2) Consistent label order (class name -> index sorted by index)
    labels = [k for k, _ in sorted(class_indices.items(), key=lambda kv: kv[1])]

    # 3) Metrics
    macro_f1    = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    print(f"\n=== Validation F1 Scores ===")
    print(f"Macro-F1   : {macro_f1:.3f}")
    print(f"Weighted-F1: {weighted_f1:.3f}")

    # 4) Text report (precision/recall/F1 per class)
    report = classification_report(y_true, y_pred, target_names=labels, digits=3)
    print("\n=== Validation Classification Report ===\n" + report)
    report_path = os.path.join(out_dir, f"classification_report_{model.name}.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved classification report to: {report_path}")

    # 5) Confusion matrix (and save a figure)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))

    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set_title(f"Confusion Matrix — {model.name}")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    # Put counts in each cell
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.tight_layout()
    cm_path = os.path.join(out_dir, f"confusion_matrix_{model.name}.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix figure to: {cm_path}")

    return {"macro_f1": float(macro_f1), "weighted_f1": float(weighted_f1),
            "cm_path": cm_path, "report_path": report_path}



# Save best model + label mapping for Step 5
def save_artifacts(model, class_indices, out_dir="artifacts"):
    import os, json
    os.makedirs(out_dir, exist_ok=True)
    model_path  = f"{out_dir}/{model.name}_best.keras"
    labels_path = f"{out_dir}/class_indices.json"
    model.save(model_path)
    with open(labels_path, "w") as f:
        json.dump(class_indices, f, indent=2)
    print(f"Saved model to: {model_path}")
    print(f"Saved class mapping to: {labels_path}")


# Main — all in one configuration


if __name__ == "__main__":
    # Build data generators (Step 1)
    train_gen, valid_gen, class_indices = make_image_generators()
    print("Class indices:", class_indices)

    # Early stopping
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        restore_best_weights=True
    )

    # These are all the configuratoins I used for the expoirements
    # CFG = BASELINE_CFG
    # CFG = DEEPER_CFG
    CFG = LEAKY_CONV_CFG
    # CFG = ELU_DENSE_CFG

    # Train single experiment
    result, history, model = run_single_experiment(
        CFG, train_gen, valid_gen, epochs=EPOCHS, early_stop_cb=early_stop
    )

    # Eval + plots (learning curves)
    eval_summary = step4_model_evaluation(history, model, valid_gen, out_dir="figures")

    # adding Confusion matrix + F1 on validation
    metrics = step4_confusion_and_f1(model, valid_gen, class_indices, out_dir="figures")
    print(f"Macro-F1={metrics['macro_f1']:.3f} | Weighted-F1={metrics['weighted_f1']:.3f}")

    # Save artifacts for Step 5 (model + label map)
    save_artifacts(model, class_indices)

    # adding training curves again (standalone)
    plot_history(history, title=f"({model.name})")

    