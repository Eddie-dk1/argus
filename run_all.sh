#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG_PATH="configs/synthetic_v0.2.json"
SKIP_GENERATE=0
SKIP_AUTOENCODER=0

usage() {
  cat <<'EOF'
Usage: ./run_all.sh [options]

Options:
  --config PATH       Experiment JSON. Default: configs/synthetic_v0.2.json
  --skip-generate     Reuse the existing synthetic raw file.
  --skip-autoencoder  Run the fast tabular-only verification path.
  --best-autoencoder  Backward-compatible alias; the v0.2 config is already used.
  -h, --help          Show this help.
EOF
}

log_step() {
  printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$1"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="${2:-}"
      if [[ -z "$CONFIG_PATH" ]]; then
        echo "Error: --config requires a path." >&2
        exit 1
      fi
      shift 2
      ;;
    --skip-generate)
      SKIP_GENERATE=1
      shift
      ;;
    --skip-autoencoder)
      SKIP_AUTOENCODER=1
      shift
      ;;
    --best-autoencoder)
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

cd "$PROJECT_ROOT"
log_step "Project root: $PROJECT_ROOT"
log_step "Config: $CONFIG_PATH"

if [[ "$SKIP_GENERATE" -eq 0 ]]; then
  log_step "Generating deterministic synthetic data"
  "$PYTHON_BIN" src/generate_data.py --config "$CONFIG_PATH"
fi

log_step "Creating chronological splits and train-fitted features"
"$PYTHON_BIN" src/preprocessing.py --config "$CONFIG_PATH"

log_step "Training and calibrating tabular baselines"
"$PYTHON_BIN" src/train_baseline.py --config "$CONFIG_PATH"

if [[ "$SKIP_AUTOENCODER" -eq 0 ]]; then
  log_step "Training and calibrating LSTM-Autoencoder"
  "$PYTHON_BIN" src/train_lstm_autoencoder.py --config "$CONFIG_PATH"

  log_step "Building validation/test comparison"
  "$PYTHON_BIN" src/evaluate.py
else
  log_step "Skipping LSTM-Autoencoder and combined evaluation"
fi

log_step "Generating explanations from model flags"
"$PYTHON_BIN" src/explain_anomalies.py

if [[ "$SKIP_AUTOENCODER" -eq 0 ]]; then
  "$PYTHON_BIN" src/print_report.py
fi

log_step "Pipeline finished"
