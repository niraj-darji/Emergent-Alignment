#!/usr/bin/env bash
# Phase 3 Runner: Benchmark Report Synthesis & Comparison Matrix
# Usage: ./scripts/run_phase3_report.sh

set -euo pipefail

export PYTHONPATH="src:${PYTHONPATH:-}"

PYTHON_BIN="/mnt/HDD_2TB/niraj/venvs/awmsm/bin/python"

echo "======================================================================"
echo "Starting Phase 3: Benchmark Report Synthesis"
echo "Aggregating Phase 1 (General & Safety) and Phase 2 (Welfare Alignment)"
echo "======================================================================"

${PYTHON_BIN} -m awmsm.eval.report_synthesis

echo "======================================================================"
echo "Phase 3 Complete."
echo "Markdown Report: results/phase3/benchmark_summary_v1.md"
echo "JSON Manifest:   results/phase3/benchmark_summary_v1.json"
echo "======================================================================"

