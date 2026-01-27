#!/bin/bash
set -e

for yaml in experiments/birds/layer_search/*/*.yaml; do
  echo "Running $yaml"
  python -m scripts.run_experiment --experiment $yaml --dry_run
done
