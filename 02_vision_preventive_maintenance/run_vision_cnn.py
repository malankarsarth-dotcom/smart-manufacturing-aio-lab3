"""Use Case 2: Vision-Based Preventive Maintenance with Camera Data (Synthetic)
Compares Pre-AI Optimization (Naive Baseline CNN) vs.
Post-AI Optimization (Normalized + Augmented + Regularized CNN + LR Scheduling + Threshold Tuning).
"""

import json
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ.setdefault("KERAS_HOME", "/tmp/keras")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential, callbacks, layers, optimizers

from generate_data import generate_vision_dataset


def set_seeds(seed: int = 42) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_dataset(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load PNG images and labels from metadata.csv."""
    meta_path = dataset_dir / "metadata.csv"
    if not meta_path.exists():
        generate_vision_dataset(dataset_dir, n_per_class=200, size=64, seed=42)

    df = pd.read_csv(meta_path)
    images = []
    for rel_path in df["filename"]:
        img = Image.open(dataset_dir / rel_path).convert("RGB")
        images.append(np.array(img, dtype=np.float32))
    X = np.stack(images, axis=0)
    y = df["label"].to_numpy(dtype=np.int32)
    return X, y, df


def save_sample_grid(X: np.ndarray, df: pd.DataFrame, out_path: Path) -> None:
    """Visualize 4 normal and 4 defective sample images."""
    norm_idx = df[df["label"] == 0].index[:4]
    def_idx = df[df["label"] == 1].index[:4]

    fig, axes = plt.subplots(2, 4, figsize=(11, 5.5))
    for col, idx in enumerate(norm_idx):
        axes[0, col].imshow(X[idx].astype(np.uint8))
        axes[0, col].set_title("Normal", fontsize=10, color="green", fontweight="bold")
        axes[0, col].axis("off")

    for col, idx in enumerate(def_idx):
        defect_name = df.loc[idx, "defect_type"]
        axes[1, col].imshow(X[idx].astype(np.uint8))
        axes[1, col].set_title(f"Defective ({defect_name})", fontsize=10, color="red", fontweight="bold")
        axes[1, col].axis("off")

    plt.suptitle("Synthetic Factory Camera Dataset — Sample Parts", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def build_pre_ai_cnn(input_shape: tuple[int, int, int]) -> tf.keras.Model:
    """Naive baseline CNN: shallow, unregularized, high fixed learning rate."""
    model = Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(8, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.01),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_post_ai_cnn(input_shape: tuple[int, int, int]) -> tf.keras.Model:
    """Optimized CNN: augmentation, 3-block Conv/Pool, Dropout regularization, tuned Adam."""
    augmentation = Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.15),
            layers.RandomContrast(0.1),
        ],
        name="data_augmentation",
    )

    model = Sequential(
        [
            layers.Input(shape=input_shape),
            augmentation,
            layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.35),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def find_best_threshold(y_val: np.ndarray, y_val_prob: np.ndarray) -> float:
    """Tune classification threshold on validation set to maximize F1 score."""
    best_thresh, best_f1 = 0.5, -1.0
    for thresh in np.linspace(0.25, 0.75, 51):
        preds = (y_val_prob >= thresh).astype(int)
        score = f1_score(y_val, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = float(thresh)
    return best_thresh


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> tuple[dict, np.ndarray]:
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "threshold": round(threshold, 3),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }
    cm = confusion_matrix(y_true, y_pred)
    return metrics, cm


def plot_training_curves(hist_pre, hist_post, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Pre-AI
    axes[0, 0].plot(hist_pre.history["accuracy"], label="Train")
    axes[0, 0].plot(hist_pre.history["val_accuracy"], label="Val")
    axes[0, 0].set_title("Pre-AI Baseline: Accuracy", fontweight="bold")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(hist_pre.history["loss"], label="Train")
    axes[0, 1].plot(hist_pre.history["val_loss"], label="Val")
    axes[0, 1].set_title("Pre-AI Baseline: Loss", fontweight="bold")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Post-AI
    axes[1, 0].plot(hist_post.history["accuracy"], label="Train")
    axes[1, 0].plot(hist_post.history["val_accuracy"], label="Val")
    axes[1, 0].set_title("Post-AI Optimized: Accuracy", fontweight="bold")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(hist_post.history["loss"], label="Train")
    axes[1, 1].plot(hist_post.history["val_loss"], label="Val")
    axes[1, 1].set_title("Post-AI Optimized: Loss", fontweight="bold")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrices(cm_pre: np.ndarray, cm_post: np.ndarray, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    labels = ["Normal", "Defective"]

    sns.heatmap(cm_pre, annot=True, fmt="d", cmap="Reds", xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_title("Pre-AI Baseline Confusion Matrix", fontweight="bold")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    sns.heatmap(cm_post, annot=True, fmt="d", cmap="Greens", xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[1].set_title("Post-AI Optimized Confusion Matrix", fontweight="bold")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_metric_bars(pre_m: dict, post_m: dict, out_path: Path) -> None:
    keys = ["accuracy", "precision", "recall", "f1_score"]
    names = ["Accuracy", "Precision", "Recall", "F1 Score"]
    pre_vals = [pre_m[k] for k in keys]
    post_vals = [post_m[k] for k in keys]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    b1 = ax.bar(x - width / 2, pre_vals, width, label="Pre-AI Baseline", color="#d9534f")
    b2 = ax.bar(x + width / 2, post_vals, width, label="Post-AI Optimized", color="#2ca02c")

    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Vision CNN: Pre-AI vs. Post-AI Optimization Metrics", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    set_seeds(42)
    base_dir = Path(__file__).resolve().parent
    dataset_dir = base_dir / "data" / "parts_dataset"
    out_dir = base_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    X_raw, y, df = load_dataset(dataset_dir)
    save_sample_grid(X_raw, df, out_dir / "sample_images.png")

    # Split 70% train, 15% val, 15% test
    X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
        X_raw, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp_raw, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    # --- 1. Pre-AI Optimization (Naive Baseline CNN on raw [0..255] pixels) ---
    pre_model = build_pre_ai_cnn(X_train_raw.shape[1:])
    hist_pre = pre_model.fit(
        X_train_raw,
        y_train,
        validation_data=(X_val_raw, y_val),
        epochs=10,
        batch_size=32,
        verbose=1,
    )
    pre_test_prob = pre_model.predict(X_test_raw, verbose=0).flatten()
    pre_metrics, cm_pre = compute_metrics(y_test, pre_test_prob, threshold=0.5)

    # --- 2. Post-AI Optimization (Normalized [0..1] + Augmented + Regularized CNN) ---
    X_train_norm = X_train_raw / 255.0
    X_val_norm = X_val_raw / 255.0
    X_test_norm = X_test_raw / 255.0

    post_model = build_post_ai_cnn(X_train_norm.shape[1:])
    cb_list = [
        callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5),
    ]
    hist_post = post_model.fit(
        X_train_norm,
        y_train,
        validation_data=(X_val_norm, y_val),
        epochs=25,
        batch_size=32,
        callbacks=cb_list,
        verbose=1,
    )

    val_prob = post_model.predict(X_val_norm, verbose=0).flatten()
    best_thresh = find_best_threshold(y_val, val_prob)

    post_test_prob = post_model.predict(X_test_norm, verbose=0).flatten()
    post_metrics, cm_post = compute_metrics(y_test, post_test_prob, threshold=best_thresh)

    # Save trained models for fast dashboard loading
    pre_model.save(out_dir / "pre_ai_model.keras")
    post_model.save(out_dir / "post_ai_model.keras")

    # Save visualizations & metrics
    plot_training_curves(hist_pre, hist_post, out_dir / "training_curves_comparison.png")
    plot_confusion_matrices(cm_pre, cm_post, out_dir / "confusion_matrices_comparison.png")
    plot_metric_bars(pre_metrics, post_metrics, out_dir / "metrics_pre_vs_post.png")

    summary = {
        "dataset": {"total_images": int(len(y)), "train": int(len(y_train)), "val": int(len(y_val)), "test": int(len(y_test))},
        "pre_ai_cnn": pre_metrics,
        "post_ai_cnn": post_metrics,
    }
    with open(out_dir / "vision_metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== PRE-AI BASELINE CNN (Test Set) ===")
    print(json.dumps(pre_metrics, indent=2))
    print("\n=== POST-AI OPTIMIZED CNN (Test Set) ===")
    print(json.dumps(post_metrics, indent=2))


if __name__ == "__main__":
    main()
