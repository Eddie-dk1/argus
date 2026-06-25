# Roadmap

Argus is planned as a staged academic research prototype rather than a
production IDS/UEBA platform.

## argus-v0.1.0

Repository rebranding and cleanup:

- rename and position the project as `Argus`;
- keep a clean public README;
- remove personal practice documents and generated artifacts from Git;
- define the data and artifact policy;
- keep a small reproducibility reference snapshot.

## argus-v0.2.0

Stabilize the current synthetic pipeline:

- confirm reproducible synthetic data generation;
- refine baseline experiment outputs;
- document the experiment protocol;
- keep the run flow simple and repeatable.

## argus-v0.3.0

Add the behavioral public dataset scenario:

- prepare `CERT Insider Threat Dataset` ingestion;
- build behavioral feature groups;
- compare tabular baselines and the sequence model.

## argus-v0.4.0

Add the first network IDS public dataset scenario:

- prepare `CICIDS2017` ingestion;
- build flow/session-level features;
- compare supervised, unsupervised, and sequence-based approaches.

## argus-v0.5.0

Add the second network IDS validation scenario:

- prepare `CSE-CIC-IDS2018` ingestion;
- compare transferability of features and thresholds;
- document differences from the `CICIDS2017` scenario.

## argus-v0.6.0

Finalize model selection and explainability:

- run moderate hyperparameter tuning;
- select final model configurations;
- analyze false positives and false negatives;
- improve anomaly explanations.

## argus-v1.0.0

Integrate the main diploma components:

- stable scenarios;
- final comparison tables and plots;
- explainability examples;
- a limited demonstration dashboard.
