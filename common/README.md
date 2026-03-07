# common

This directory contains shared utilities used across all versions (v0, v1, v2).

---

## Purpose

The goal of `common` is to avoid duplicated logic and ensure consistency across models and systems.

---

## Contents

- `text_utils.py`
  - Text preprocessing functions (e.g. `clean_text`)
  - Used by:
    - v0: during feature extraction
    - v1: during vectorization
    - v2: during inference

---

## Design Rationale

By centralizing text preprocessing:
- All models operate on identically cleaned input
- Saved model artifacts remain reusable across versions
- Inference behavior in v2 is consistent with training in v0/v1
