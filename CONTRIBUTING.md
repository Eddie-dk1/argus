# Contributing

Argus is currently an academic research prototype maintained for diploma work.
Contributions are welcome when they keep the repository reproducible, focused,
and free of local or personal artifacts.

## Basic Rules

- Keep generated datasets, model weights, temporary outputs, reports, and local
  previews out of Git.
- Prefer small, reviewable changes with a clear experiment or documentation goal.
- Update README or docs when changing commands, artifact policy, or project
  structure.
- Run the relevant pipeline command before submitting code changes:

```bash
bash run_all.sh
```

## Data And Artifacts

Do not commit public datasets, private datasets, intermediate experiment dumps,
or trained model weights. Document how to obtain and prepare data instead.

Small reference metrics and plots may be committed only when they help readers
understand or reproduce the current baseline.
