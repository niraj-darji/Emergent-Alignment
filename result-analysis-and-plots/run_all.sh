#!/usr/bin/env bash
# Regenerate every table and figure from the raw result files.
#   - steps 1, 2, 4 need the project venv (json/numpy; step 4 also needs transformers for tokenization)
#   - steps 3, 5 need matplotlib, installed in the local .venv (uv venv .venv && uv pip install matplotlib numpy)
set -euo pipefail
cd "$(dirname "$0")"
PROJ=/mnt/HDD_2TB/niraj/venvs/awmsm/bin/python
PLOT=.venv/bin/python
$PROJ 01_collect_results.py
$PROJ 02_statistics.py
$PLOT 03_make_figures.py
$PROJ 04_dataset_stats.py
$PLOT 05_dataset_figures.py
# copy the PDFs used by the paper into the LaTeX project
mkdir -p /mnt/HDD_2TB/niraj/latex/ETHICS_IN__AI/figures
cp figures/*.pdf /mnt/HDD_2TB/niraj/latex/ETHICS_IN__AI/figures/
$PROJ 06_arms_table.py
