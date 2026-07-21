# Working Literature Review

Version: `1.0`

Review cutoff: `2026-07-21`

This is a focused engineering review for the next Argus research stages. It is
not presented as a systematic review or a claim of exhaustive coverage. Papers
were selected when they directly support a model, metric, dataset decision, or
known experimental risk in the project. Publisher pages, proceedings, dataset
owners, and DOI records are preferred over secondary summaries.

## Decisions Supported by the Literature

1. Keep both supervised and unsupervised baselines. Random Forest and Isolation
   Forest answer different questions and should not be interpreted as
   interchangeable competitors.
2. Treat the LSTM Autoencoder as a sequence-based comparison point, not as the
   model that must win. Its value is the ability to model normal temporal
   structure without anomaly labels.
3. Use chronological train/validation/test splits and fit every learned
   transformation on train only. Strong results from random or resampled splits
   do not establish temporal generalization.
4. Select thresholds on validation data and open the test set once for the
   final report.
5. Make PR-AUC, precision, recall, F1, and false-positive rate central under
   class imbalance. Accuracy is descriptive only.
6. Do not use current or future ground-truth labels as input features. Published
   CERT work shows that label-history features can lift predictive performance,
   but they create an operationally fragile evaluation target for Argus.
7. Treat CICIDS2017 features and labels as data that require validation, not as
   unquestionable ground truth. Preserve raw column provenance and document
   every correction.
8. Explanations describe model behavior, not causality. Global feature
   importance, local attributions, reconstruction residuals, and rule-based
   context must be kept conceptually separate.

## Evidence Matrix

| Work | Evidence relevant to Argus | Decision for Argus | Limitation to retain |
| --- | --- | --- | --- |
| Breiman, **Random Forests** (2001), [DOI](https://doi.org/10.1023/A:1010933404324) | Establishes the tree-ensemble baseline based on randomized feature selection and bootstrap samples. | Keep Random Forest as the supervised tabular reference and export held-out feature importance. | It relies on labels and does not establish temporal anomaly detection by itself. |
| Liu, Ting, and Zhou, **Isolation Forest** (2008), [DOI](https://doi.org/10.1109/ICDM.2008.17) | Introduces isolation by random partitioning: anomalies tend to require shorter paths. | Keep an unsupervised baseline trained on normal train data; tune the score threshold on validation. | The score is not calibrated probability and performance depends strongly on the feature representation. |
| Hochreiter and Schmidhuber, **Long Short-Term Memory** (1997), [DOI](https://doi.org/10.1162/neco.1997.9.8.1735) | Provides the gated recurrent architecture used to learn longer temporal dependencies. | Use LSTM only where event order is meaningful and compare on the exact endpoint cohort used by baselines. | Architecture choice alone does not make an experiment temporally valid. |
| Malhotra et al., **LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection** (2016), [preprint](https://arxiv.org/abs/1607.00148) | Models normal sequences and uses reconstruction-error distributions for anomaly scoring. | Train the autoencoder on normal train sequences; choose an error threshold using validation data. | Workshop/preprint evidence and domain-specific experiments; it is motivation, not proof that LSTM-AE will outperform tabular models. |
| Saito and Rehmsmeier, **The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets** (2015), [DOI](https://doi.org/10.1371/journal.pone.0118432) | Demonstrates why ROC plots can appear optimistic under severe imbalance and why precision-recall analysis is more informative. | Report PR-AUC beside ROC-AUC and include class support and prevalence in every comparison. | PR values depend on prevalence, so cross-dataset values must not be compared without the base rate. |
| Sharafaldin, Habibi Lashkari, and Ghorbani, **Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization** (2018), [DOI](https://doi.org/10.5220/0006639801080116) | Documents the CICIDS2017 environment, labeled benign/attack flows, attack families, and the extraction of more than 80 flow features. | Use the processed flow CSVs first, retain the day/source file, and construct a network-specific feature map. | A generated benchmark is not equivalent to current production traffic. |
| Engelen, Rimmer, and Joosen, **Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study** (2021), [DOI](https://doi.org/10.1109/SPW53761.2021.00009) | Reports reproducibility and data-quality problems in CICIDS2017 and provides a corrected workflow. | Run schema, missing-value, duplicate, label, and provenance checks before modeling; record any correction in a manifest. | Corrections do not remove every benchmark artifact or domain-shift risk. |
| Manoharan et al., **Optimising Insider Threat Prediction: Exploring BiLSTM Networks and Sequential Features** (2024), [DOI](https://doi.org/10.1007/s41019-024-00260-z) | Studies daily sequential features and RNN/LSTM/BiLSTM models on CERT r4.2; it explicitly includes daily ground-truth history in some feature combinations. | Use user-day and sequence ideas, but forbid label-derived history and separate detection from genuinely prospective prediction. | Very high reported performance with label-history inputs is not an operationally comparable target for Argus. |
| Pal, Chattopadhyay, and Swarnkar, **Temporal Feature Aggregation with Attention for Insider Threat Detection from Activity Logs** (2023), [DOI](https://doi.org/10.1016/j.eswa.2023.119925) | Uses single-day sequential activity, attention-based recurrent models, and class rebalancing on several CERT versions. | Treat per-user temporal aggregation and anomaly-family analysis as candidate experiments after the plain baselines are stable. | Random sampling and resampling can answer a different question from future-period generalization and may alter deployment prevalence. |
| Jaiswal, Dwivedi, and Dewang, **Handling Imbalance Dataset Issue in Insider Threat Detection Using Machine Learning Methods** (2024), [DOI](https://doi.org/10.1016/j.compeleceng.2024.109726) | Compares sampling/classifier combinations on day-long CERT r4.2/r5.2 sequences. | If sampling is studied, apply it only within train and always evaluate on the untouched natural validation/test distributions. | Balanced accuracy or resampled performance alone does not reveal the operational false-alert load. |
| Lundberg and Lee, **A Unified Approach to Interpreting Model Predictions** (2017), [NeurIPS proceedings](https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html) | Unifies additive feature attribution methods and introduces SHAP values for local prediction explanations. | Add SHAP for the selected tabular model only after data and model baselines are fixed; store background-data provenance. | Attribution explains the fitted model, not the real-world cause of an event; correlated features complicate interpretation. |
| Guo et al., **LEMNA: Explaining Deep Learning Based Security Applications** (2018), [DOI](https://doi.org/10.1145/3243734.3243792) | Shows that dependencies and nonlinear local boundaries matter when explaining security models. | For sequential models, prefer reconstruction-error contributions plus temporal context; evaluate fidelity before presenting an explanation as reliable. | Its demonstrated security applications are not insider-threat or flow-anomaly detection, so direct transfer is a hypothesis. |

## Research Gaps That Become Argus Experiments

### G1. Temporal generalization versus random holdout

Many reported benchmark results use stratified or randomly resampled examples.
Argus will use chronological splits as the primary result and may add a random
split only as a clearly labeled diagnostic. The comparison measures how much of
the apparent score depends on temporal overlap or stable benchmark artifacts.

### G2. Honest cohort alignment across model families

A sequence model cannot score the first events for which a complete history is
unavailable. Comparing it with a tabular model evaluated on all events changes
the population. Argus therefore evaluates every model on the exact same
sequence-endpoint rows and verifies row identity before producing a comparison.

### G3. Threshold stability

Anomaly papers often emphasize ranking metrics while alerting systems require a
decision threshold. Argus will record the validation objective, selected
threshold, validation false-positive rate, and unchanged test result. Threshold
sensitivity is a planned analysis for each public scenario.

### G4. Dataset artifacts and cross-dataset transfer

CICIDS2017 and CSE-CIC-IDS2018 share a flow-oriented setting but are not one
homogeneous dataset. Argus will first reproduce a within-dataset result, then
test which transformations, features, and thresholds transfer. Failure to
transfer is a valid result and should be analyzed rather than hidden by merging
the datasets.

### G5. Explanations that are useful to an analyst

Attribution magnitude alone is not an incident explanation. Argus will combine:

- feature or reconstruction contribution;
- deviation from a normal profile fitted on train;
- temporal context from the preceding window;
- a short rule-based statement tied to observed fields.

The explanation must state uncertainty and must never claim that an attributed
feature caused malicious behavior.

## Reading and Implementation Order

1. **Before acquiring CERT:** read the CERT dataset documentation together with
   the Manoharan and Pal papers; freeze the operational target and user-day
   split before implementing features.
2. **Before modeling CICIDS2017:** read the dataset-generation and
   troubleshooting papers together; complete data validation before training.
3. **Before imbalance experiments:** establish untouched chronological
   validation/test baselines, then confine sampling experiments to train.
4. **Before explainability:** freeze the selected model and feature schema, then
   validate explanation fidelity and write analyst-facing limitations.

## Required Outputs for Each New Paper

When a paper is added to this review, record:

- full citation and stable DOI/publisher link;
- dataset version and unit of analysis;
- target and label definition;
- split and resampling method;
- whether preprocessing was fitted only on train;
- threshold-selection procedure;
- metrics with class support/prevalence when available;
- code/data availability;
- one concrete Argus decision and one limitation.

This checklist is intentionally stricter than a bibliography: a paper belongs
here only when it changes or justifies a reproducible project decision.
