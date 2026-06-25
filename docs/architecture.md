# Architecture Sketch

This document is a lightweight planning sketch. It does not introduce a new
implementation yet; it defines the target module boundaries for future diploma
development.

```text
Raw or generated logs
        |
        v
Data ingestion
        |
        v
Preprocessing and splits
        |
        v
Feature engineering
        |
        +--------------------+
        |                    |
        v                    v
Tabular datasets      Sequence datasets
        |                    |
        v                    v
Baseline models       LSTM Autoencoder
        |                    |
        +---------+----------+
                  |
                  v
Evaluation and plots
                  |
                  v
Explainability artifacts
                  |
                  v
Demo dashboard
```

## Intended Layers

- `data ingestion`: load synthetic or public dataset files into a known local
  structure.
- `preprocessing`: clean records, encode fields, and create train/validation/test
  splits without leakage.
- `feature engineering`: build scenario-specific tabular and sequence features.
- `models`: train baselines and sequence models behind clear experiment commands.
- `evaluation`: compute metrics, plots, thresholds, and comparison tables.
- `explainability`: produce short explanations for suspicious events or
  sequences.
- `dashboard`: present prepared artifacts without becoming the core research
  engine.

The main design rule is to keep the research pipeline reproducible while
avoiding premature production architecture.
