# Group-8 Project: Sentiment Analysis System (v0 → v1 → v2)

This project demonstrates a complete machine learning pipeline for sentiment analysis,
progressing from baseline model training to an end-to-end deployable system.

The project is organized into three incremental versions:
- **v0**: Baseline model training and evaluation
- **v1**: Improved model with a more advanced pipeline
- **v2**: System-level integration (CLI, web server, tests, logging)

All versions use the **same dataset** (Stanford IMDb) with **fully automated download and preprocessing**.
There are **no hard-coded local paths** or manual data preparation steps.

---

## Version Dependency (Incremental Build)

This project is explicitly designed to be incremental. Each version builds on the previous one
mainly by **adding new components**, rather than rewriting from scratch:

- **v0**  
  Trains a baseline sentiment model on the IMDb dataset and saves the trained artifact  
  (`best_model_v0_*.joblib`).

- **v1**  
  Reuses the same automatically downloaded IMDb dataset and training protocol,
  and trains an improved model with a more advanced pipeline  
  (`best_model_v1_nn_*.joblib`).

- **v2**  
  Does **not** retrain models. Instead, it loads the trained artifacts from v0 and v1,
  and adds system-level features:
  - Command-line interface (CLI)
  - Web server with interactive UI
  - Automated tests
  - Logging

This structure demonstrates both **functional progression** and **engineering discipline**.

---

## Project Structure

```
GROUP-8/
├── common/      # Shared utilities (e.g. text preprocessing, config loader)
├── v0/          # Baseline ML model
├── v1/          # Improved ML model
├── v2/          # System-level application (CLI + web server)
├── config.json  # Optional configuration file
├── requirements.txt
└── README.md
```

---

## Environment & Setup

### Create a virtual environment (recommended)

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (Command Prompt / cmd)**
```bat
py -m venv .venv
.venv\Scripts\activate.bat
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## How to Run

### v0 — Baseline Model

```bash
python -m v0.v0_auto
```

### v1 — Improved Model

```bash
python -m v1.v1_auto
```

### v2 — System Demo

**CLI prediction**

```bash
python -m v2.cli predict --model v0 --text "this movie is amazing"
```

**Web server**

```bash
python -m v2.cli serve
```

Then open: http://127.0.0.1:8000

**Run tests**

```bash
python -m pytest -q
```

---

## Configuration (optional)

This project supports an optional JSON configuration file at `config.json` (project root).
CLI arguments always override configuration values.

Example:

```bash
python -m v2.cli --config my_config.json serve
```

---

## Logging

The system logs model loading, prediction requests, and errors to `logs/v2.log` for easier debugging and evaluation.

---

## Notes on Data

- The Stanford IMDb dataset is downloaded **automatically** on first run.
- Data is cached locally and reused across v0, v1, and v2.
- No dataset files are included in the submission.
