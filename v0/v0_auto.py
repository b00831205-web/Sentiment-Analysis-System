"""v0 baseline training script.

This stage provides a minimal, fully runnable pipeline:
- download & cache the IMDb dataset,
- load train/test splits,
- train baseline models,
- write artifacts (model + metrics) for later stages to reuse.
"""

import re
import json
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Allow running this module directly while keeping imports relative to project root.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))  

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import joblib

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
            if p.suffix == ".txt":
                texts.append(p.read_text(encoding="utf-8", errors="ignore"))
                labels.append(y)

    return texts, labels


def main():
    """Run the end-to-end v0 baseline pipeline.

    This function orchestrates dataset acquisition, data loading, model training,
    evaluation, and artifact persistence.

    Outputs:
    Writes artifacts under `v0/`:
        - `best_model_v0_*.joblib` (serialized model bundle)
        - `metrics_v0_*.json` (evaluation metrics)
    """

    data_dir = ensure_aclImdb()

    X_train, y_train = load_imdb_split(data_dir, "train")
    X_test, y_test = load_imdb_split(data_dir, "test")

    tfidf_kwargs = dict(
        preprocessor=clean_text,
        stop_words="english",
        max_features=100_000,
        ngram_range=(1, 2),
    )

    models = [
        (
            "LR(base)",
            Pipeline(
                [
                    ("tfidf", TfidfVectorizer(**tfidf_kwargs)),
                    ("clf", LogisticRegression(max_iter=2000, n_jobs=-1)),
                ]
            ),
        ),
        (
            "MultinomialNB",
            Pipeline(
                [
                    ("tfidf", TfidfVectorizer(**tfidf_kwargs)),
                    ("clf", MultinomialNB()),
                ]
            ),
        ),
    ]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    trained = {}

    for name, model in models:
        cv_out = cross_validate(
            model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1
        )
        cv_scores = cv_out["test_score"]
        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))

        model.fit(X_train, y_train)
        test_acc = float(accuracy_score(y_test, model.predict(X_test)))

        results[name] = {
            "cv_scores": [float(x) for x in cv_scores],
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "test_acc": test_acc,
        }

        print(
            f"{name:12s} | CV acc: {cv_mean:.4f} ± {cv_std:.4f} | Test acc: {test_acc:.4f}"
        )
        trained[name] = model

    best = max(results, key=lambda k: results[k]["test_acc"])
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

  
    safe_best = re.sub(r"[^0-9A-Za-z_+-]+", "_", best)
    model_fname = f"best_model_v0_{safe_best}_{ts}.joblib"
    joblib.dump(
        {
            "best_name": best,
            "model": trained[best],
            "metrics": results,
            "timestamp": ts,
        },
        Path(__file__).with_name(model_fname),
    )
    print("Saved model:", model_fname)

    
    metrics_fname = f"metrics_v0_{ts}.json"
    Path(__file__).with_name(metrics_fname).write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Saved metrics:", metrics_fname)


if __name__ == "__main__":
    main()
