"""v1 improved model training script.

This stage builds on v0 by reusing the same dataset acquisition & preprocessing
but introducing a more advanced modeling approach and training loop.
Artifacts produced here are later consumed by v2 for deployment.
"""

import argparse
import json
import re
from pathlib import Path
from datetime import datetime
import joblib
import sys 

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))  


import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt
import seaborn as sns

from v0.data import ensure_aclImdb

from common.text_utils import clean_text


def load_imdb_split(root_dir: str, split: str):
    """Load a split (train/test) from an `aclImdb` directory.

    Args:
        root_dir: Path to the extracted `aclImdb/` directory.
        split: One of {"train", "test"}.

    Returns:
        A tuple (texts, labels), where texts is a list of review strings and labels is
        a list/array of integer sentiment labels (1=pos, 0=neg).

    Raises:
        FileNotFoundError: If the expected split folders are missing.
    """

    root = Path(root_dir)
    texts, labels = [], []
    for label_name, y in (("pos", 1), ("neg", 0)):
        folder = root / split / label_name
        if not folder.exists():
            raise FileNotFoundError(f"Missing folder: {folder}")
        for p in folder.iterdir():
            if p.suffix.lower() != ".txt":
                continue
            with p.open("r", encoding="utf-8", errors="ignore") as f:
                texts.append(f.read())
                labels.append(y)
    return texts, np.array(labels, dtype=np.float32).reshape(-1, 1)


def sigmoid(x):
    """Compute the logistic sigmoid function.

    Args:
        x: Input scalar or NumPy array.

    Returns:
        Sigmoid(x) with the same shape as the input.
    """

    x = np.clip(x, -30, 30)
    return 1.0 / (1.0 + np.exp(-x))


def bce_loss(y_true, y_prob, l2=0.0, W1=None, W2=None):
    """Binary cross-entropy loss with optional L2 regularization.

    Args:
        y_true: Ground-truth labels (0/1). Shape should be compatible with `y_prob`.
        y_prob: Predicted probabilities in [0, 1].
        l2: L2 regularization strength. Set to 0.0 to disable.
        W1: First-layer weight matrix (required if `l2 > 0`).
        W2: Second-layer weight matrix (required if `l2 > 0`).

    Returns:
        Scalar average loss value (float).
    """

    eps = 1e-12
    y_prob = np.clip(y_prob, eps, 1 - eps)
    base = -(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)).mean()
    if l2 and W1 is not None and W2 is not None:
        base = base + 0.5 * l2 * (np.sum(W1 * W1) + np.sum(W2 * W2)) / y_true.shape[0]
    return float(base)


class TwoLayerNN:
    """A simple two-layer neural network for binary classification.

    This model is implemented from scratch (NumPy) to demonstrate staged development
    and custom training logic.

    Attributes:
        W1, b1, W2, b2: Learned parameters.
    """

    def __init__(self, input_dim, hidden_dim=128, lr=0.1, l2=1e-4, seed=42):
        """Initialize network parameters.

        Args:
            input_dim: Number of input features.
            hidden_dim: Hidden layer width.
            lr: Learning rate used by `backward()` updates.
            l2: L2 regularization strength applied to weights.
            seed: Random seed for reproducibility.
        """
        
        rng = np.random.default_rng(seed)
        self.W1 = (rng.normal(0, 1, size=(input_dim, hidden_dim)) * np.sqrt(1.0 / input_dim)).astype(np.float32)
        self.b1 = np.zeros((1, hidden_dim), dtype=np.float32)
        self.W2 = (rng.normal(0, 1, size=(hidden_dim, 1)) * np.sqrt(1.0 / hidden_dim)).astype(np.float32)
        self.b2 = np.zeros((1, 1), dtype=np.float32)

        self.lr = float(lr)
        self.l2 = float(l2)

    def forward(self, X):
        """Forward pass.

        Args:
            X: Input feature matrix of shape (n_samples, input_dim).

        Returns:
            A cache of intermediate activations needed for backprop.
        """
        Z1 = X @ self.W1 + self.b1
        A1 = np.tanh(Z1)
        Z2 = A1 @ self.W2 + self.b2
        Y = sigmoid(Z2)
        cache = (X, A1, Y)
        return Y, cache

    def backward(self, y_true, cache):
        """Backward pass and in-place parameter update.

        This method computes gradients for the current batch and updates
        parameters using `self.lr` and `self.l2`.

        Args:
            y_true: Ground-truth labels for the batch, shape (n_samples, 1) or compatible.
            cache: Intermediate values returned by `forward()`.

        Returns:
            None. Updates parameters in place.
        """

        X, A1, Y = cache
        n = X.shape[0]

        dZ2 = (Y - y_true) / n
        dW2 = A1.T @ dZ2 + (self.l2 / n) * self.W2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (1.0 - A1 * A1)
        dW1 = X.T @ dZ1 + (self.l2 / n) * self.W1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        self.W1 -= self.lr * dW1.astype(np.float32)
        self.b1 -= self.lr * db1.astype(np.float32)
        self.W2 -= self.lr * dW2.astype(np.float32)
        self.b2 -= self.lr * db2.astype(np.float32)

    def predict_proba(self, X):
        """Predict positive-class probabilities.

        Args:
            X: Input feature matrix.

        Returns:
            Array of probabilities in [0, 1], shape (n_samples,).
        """

        Y, _ = self.forward(X)
        return Y

    def predict(self, X):
        """Predict hard class labels.

        Args:
            X: Input feature matrix.

        Returns:
            Integer array of predicted labels (0/1), shape (n_samples,).
        """
        return (self.predict_proba(X) >= 0.5).astype(np.int32)


def make_minibatches(X, y, batch_size, seed=42):
    """Yield shuffled mini-batches for training.

    Args:
        X: Feature matrix.
        y: Labels.
        batch_size: Batch size.
        seed: Random seed controlling the shuffling order.

    Yields:
        Tuples (X_batch, y_batch).
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(X.shape[0])
    rng.shuffle(idx)
    for start in range(0, X.shape[0], batch_size):
        batch_idx = idx[start : start + batch_size]
        yield X[batch_idx], y[batch_idx]


def train_epochs(model, X_train, y_train, X_eval, y_eval, epochs, batch_size, seed):
    """Train a model for a fixed number of epochs.

    Args:
        model: A `TwoLayerNN` instance.
        X_train: Training features.
        y_train: Training labels.
        X_eval: Evaluation features (validation or test).
        y_eval: Evaluation labels.
        epochs: Number of epochs.
        batch_size: Batch size.
        seed: Random seed controlling mini-batch shuffling.

    Returns:
        A tuple: (train_losses, eval_losses, eval_metrics) where:
        - train_losses: list of per-epoch training loss values
        - eval_losses: list of per-epoch evaluation loss values
        - eval_metrics: list of per-epoch evaluation metrics (dict with accuracy, precision, recall, f1, confusion_matrix)
    """

    train_losses, eval_losses = [], []
    eval_metrics = []
    best_state = {"epoch": 0, "eval_acc": -1.0, "W1": None, "b1": None, "W2": None, "b2": None}

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        n_seen = 0
        for xb, yb in make_minibatches(X_train, y_train, batch_size, seed=seed + epoch):
            y_prob, cache = model.forward(xb)

            eps = 1e-12
            yp = np.clip(y_prob, eps, 1 - eps)
            batch_loss = -(yb * np.log(yp) + (1 - yb) * np.log(1 - yp)).mean()

            bs = xb.shape[0]
            total_loss += float(batch_loss) * bs
            n_seen += bs

            model.backward(yb, cache)

        reg = 0.5 * model.l2 * (np.sum(model.W1 * model.W1) + np.sum(model.W2 * model.W2)) / max(1, n_seen)
        train_loss = total_loss / max(1, n_seen) + float(reg)

        y_eval_prob = model.predict_proba(X_eval)
        yp2 = np.clip(y_eval_prob, eps, 1 - eps)
        eval_loss = float(-(y_eval * np.log(yp2) + (1 - y_eval) * np.log(1 - yp2)).mean())
        eval_pred = (y_eval_prob >= 0.5).astype(np.int32).reshape(-1)
        y_eval_flat = y_eval.reshape(-1).astype(np.int32)
        
        
        eval_acc = accuracy_score(y_eval_flat, eval_pred)
        eval_precision = precision_score(y_eval_flat, eval_pred, zero_division=0)
        eval_recall = recall_score(y_eval_flat, eval_pred, zero_division=0)
        eval_f1 = f1_score(y_eval_flat, eval_pred, zero_division=0)
        eval_cm = confusion_matrix(y_eval_flat, eval_pred)

        train_losses.append(train_loss)
        eval_losses.append(eval_loss)
        eval_metrics.append({
            "accuracy": float(eval_acc),
            "precision": float(eval_precision),
            "recall": float(eval_recall),
            "f1": float(eval_f1),
            "confusion_matrix": eval_cm.tolist(),
        })

        print(f"Epoch {epoch:03d}/{epochs} | loss={train_loss:.4f} | acc={eval_acc:.4f} | prec={eval_precision:.4f} | rec={eval_recall:.4f} | f1={eval_f1:.4f}")

        if eval_acc > best_state["eval_acc"]:
            best_state = {
                "epoch": epoch,
                "eval_acc": float(eval_acc),
                "W1": model.W1.copy(),
                "b1": model.b1.copy(),
                "W2": model.W2.copy(),
                "b2": model.b2.copy(),
            }

    return train_losses, eval_losses, eval_metrics, best_state


def main():
    """Run the end-to-end v1 training pipeline.

    This function reuses the dataset pipeline from v0 and trains the improved model,
    saving artifacts (model + preprocessing bundle) under `v1/`.
    """
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_dir", type=str, default="AUTO")
    ap.add_argument("--max_features", type=int, default=100_000)
    ap.add_argument("--svd_dim", type=int, default=300)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--no_show", action="store_true", help="Do not call plt.show()")
    args = ap.parse_args()


    if args.data_dir.upper() == "AUTO":
        data_dir = ensure_aclImdb()
    else:
        data_dir = args.data_dir

    X_train_text, y_train = load_imdb_split(data_dir, "train")
    X_test_text, y_test = load_imdb_split(data_dir, "test")

    vectorizer = TfidfVectorizer(
        preprocessor=clean_text,  
        stop_words="english",
        max_features=args.max_features,
        ngram_range=(1, 2),
    )
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_test_tfidf = vectorizer.transform(X_test_text)

    svd = TruncatedSVD(n_components=args.svd_dim, random_state=args.seed)
    X_train = svd.fit_transform(X_train_tfidf).astype(np.float32)
    X_test = svd.transform(X_test_tfidf).astype(np.float32)

    mu = X_train.mean(axis=0, keepdims=True)
    sigma = X_train.std(axis=0, keepdims=True) + 1e-6
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma

    skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)
    y_strat = y_train.reshape(-1)

    fold_train_losses = []
    fold_val_accs = []
    fold_summary = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_strat), start=1):
        print(f"\n===== Fold {fold}/{args.cv} =====")
        X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
        X_va, y_va = X_train[va_idx], y_train[va_idx]

        model = TwoLayerNN(
            input_dim=X_train.shape[1],
            hidden_dim=args.hidden,
            lr=args.lr,
            l2=args.l2,
            seed=args.seed + fold,
        )

        tr_losses, va_losses, va_metrics, _ = train_epochs(
            model, X_tr, y_tr, X_va, y_va,
            epochs=args.epochs, batch_size=args.batch, seed=args.seed + 1000 * fold
        )

        fold_train_losses.append(tr_losses)
        fold_val_accs.append([m["accuracy"] for m in va_metrics])

        fold_summary.append({
            "fold": fold,
            "val_acc_last": va_metrics[-1]["accuracy"],
            "val_acc_best": max([m["accuracy"] for m in va_metrics]),
            "val_recall_best": max([m["recall"] for m in va_metrics]),
        })

        print(f"Fold {fold} | val_acc_best={max([m['accuracy'] for m in va_metrics]):.4f} | val_rec_best={max([m['recall'] for m in va_metrics]):.4f}")

    best_vals = [f["val_acc_best"] for f in fold_summary]
    cv_mean = float(np.mean(best_vals))
    cv_std = float(np.std(best_vals))
    print(f"\nCV best-val-acc mean±std: {cv_mean:.4f} ± {cv_std:.4f}")

    print("\n===== Final Train on full train, Eval on test =====")
    final_model = TwoLayerNN(
        input_dim=X_train.shape[1],
        hidden_dim=args.hidden,
        lr=args.lr,
        l2=args.l2,
        seed=args.seed,
    )

    final_train_losses, final_test_losses, final_test_metrics, final_best = train_epochs(
        final_model, X_train, y_train, X_test, y_test,
        epochs=args.epochs, batch_size=args.batch, seed=args.seed
    )

    final_model.W1 = final_best["W1"]
    final_model.b1 = final_best["b1"]
    final_model.W2 = final_best["W2"]
    final_model.b2 = final_best["b2"]
    
    # Report final test metrics at the best epoch
    best_test_metrics = final_test_metrics[final_best["epoch"] - 1]
    print(f"Final(best) at epoch={final_best['epoch']}")
    print(f"  Test Accuracy:  {best_test_metrics['accuracy']:.4f}")
    print(f"  Test Precision: {best_test_metrics['precision']:.4f}")
    print(f"  Test Recall:    {best_test_metrics['recall']:.4f}")
    print(f"  Test F1-Score:  {best_test_metrics['f1']:.4f}")
    print(f"  Confusion Matrix:\n{best_test_metrics['confusion_matrix']}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = Path(__file__).with_name(f"nn_cv_curve_{ts}.json")
    out_png = Path(__file__).with_name(f"nn_learning_curve_{ts}.png")
    out_model = Path(__file__).with_name(f"best_model_v1_nn_{ts}.joblib")

    cv_train_mean = list(np.mean(np.array(fold_train_losses), axis=0))
    cv_val_acc_mean = list(np.mean(np.array(fold_val_accs), axis=0))
    test_acc_last = final_test_metrics[-1]["accuracy"]
    test_acc_best = max([m["accuracy"] for m in final_test_metrics])
    test_rec_best = max([m["recall"] for m in final_test_metrics])
    test_prec_best = max([m["precision"] for m in final_test_metrics])
    test_f1_best = max([m["f1"] for m in final_test_metrics])

    payload = {
        "timestamp": ts,
        "params": vars(args),
        "data_dir_used": str(data_dir),
        "cv": {
            "n_splits": args.cv,
            "folds": fold_summary,
            "best_val_acc_mean": cv_mean,
            "best_val_acc_std": cv_std,
            "curve_mean": {
                "train_loss": [float(x) for x in cv_train_mean],
                "val_acc": [float(x) for x in cv_val_acc_mean],
            },
        },
        "final": {
            "train_loss": [float(x) for x in final_train_losses],
            "test_loss": [float(x) for x in final_test_losses],
            "test_accuracy": [m["accuracy"] for m in final_test_metrics],
            "test_precision": [m["precision"] for m in final_test_metrics],
            "test_recall": [m["recall"] for m in final_test_metrics],
            "test_f1": [m["f1"] for m in final_test_metrics],
            "test_confusion_matrix": best_test_metrics["confusion_matrix"],
            "test_metrics_best": {
                "epoch": final_best["epoch"],
                "accuracy": test_acc_best,
                "precision": test_prec_best,
                "recall": test_rec_best,
                "f1": test_f1_best,
                "confusion_matrix": best_test_metrics["confusion_matrix"],
            },
            "test_metrics_last": {
                "accuracy": test_acc_last,
                "precision": final_test_metrics[-1]["precision"],
                "recall": final_test_metrics[-1]["recall"],
                "f1": final_test_metrics[-1]["f1"],
                "confusion_matrix": final_test_metrics[-1]["confusion_matrix"],
            },
        },
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved curves/results to: {out_json}")

    epochs = np.arange(1, args.epochs + 1)
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(epochs, final_train_losses, label="Final Train Loss")
    ax1.plot(epochs, final_test_losses, label="Final Test Loss", alpha=0.8)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    test_accs = [m["accuracy"] for m in final_test_metrics]
    ax2.plot(epochs, test_accs, color="tab:orange", label="Final Test Acc")
    ax2.set_ylabel("Accuracy")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center right")

    plt.title("Final Learning Curve (NumPy 2-layer NN)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)
    print(f"Saved learning curve figure to: {out_png}")

    # Plot confusion matrix for the best epoch
    out_cm_png = Path(__file__).with_name(f"confusion_matrix_{ts}.png")
    cm = np.array(best_test_metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"],
                ax=ax, cbar_kws={"label": "Count"})
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix (Epoch {final_best['epoch']})")
    plt.tight_layout()
    plt.savefig(out_cm_png, dpi=150)
    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)
    print(f"Saved confusion matrix to: {out_cm_png}")


    joblib.dump(
        {
            "version": "v1",
            "model_type": "nn_2layer_tanh_svd",
            "timestamp": ts,
            "params": vars(args),
            "vectorizer": vectorizer,
            "svd": svd,
            "mu": mu.astype(np.float32),
            "sigma": sigma.astype(np.float32),
            "nn": {
                "W1": final_model.W1,
                "b1": final_model.b1,
                "W2": final_model.W2,
                "b2": final_model.b2,
            },
        },
        out_model,
    )
    print(f"Saved best v1 NN model to: {out_model}")


if __name__ == "__main__":
    main()
