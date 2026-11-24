# Result analysis and plots

Scripts that turn the raw evaluation outputs in `awmsm/results/` (and training logs/manifests) into the
tables and figures used in the DS 307 report (`latex/ETHICS_IN__AI/sections/results.tex` and
`appendix_results.tex`). They never read the hand-written summary markdowns.

Run everything with `./run_all.sh`.

| Step | Script | Output |
|---|---|---|
| 1 | `01_collect_results.py` | `tables/all_results.{json,csv}` (every arm, every metric, per-item judge scores), `tables/training_stats.csv` |
| 2 | `02_statistics.py` | `tables/arm_cis.csv` (95% bootstrap CIs), `tables/paired_comparisons.csv` (paired bootstrap + sign-flip permutation, Holm-adjusted), `tables/mc_standard_errors.csv`, `tables/correlations.csv` (Spearman across arms) |
| 3 | `03_make_figures.py` | main-text and appendix figures in `figures/`, LaTeX tables `tables/tab_capability.tex`, `tables/tab_full_results.tex` |
| 4 | `04_dataset_stats.py` | `tables/dataset_stats.json` (AW demo categories, reply lengths, effective gradient weights per mix) |
| 5 | `05_dataset_figures.py` | `figures/fig_dataset.pdf` |

`common.py` holds the arm definitions (label, midtraining corpus, AW share p, checkpoint), file lookup and
the plot style. Colours follow the condition in every figure (blue = MSM v1, orange = no midtraining,
aqua = MSM v2, yellow = CaML corpus); the palette was checked with a colour-vision-deficiency validator.

## Figures

| File | Used in | Shows |
|---|---|---|
| `fig_welfare_vs_p` | Results | ANIMA and MANTA vs AW share p per midtraining condition, 95% CIs |
| `fig_dose` | Results | 2.5M-20M midtraining tokens at p=0: ANIMA, MANTA, MMLU, loss on fine-tuning data before FT |
| `fig_loss_head_start` | Results | fine-tuning held-out loss before/after AFT, raw vs midtrained |
| `fig_ethics_vs_p` | Results | ETHICS subtasks vs p |
| `fig_generalization` | Results | ANIMA vs ETHICS (macro, deontology+justice, utilitarianism) across arms with Spearman rho |
| `fig_paired_effects` | Appendix | forest plot of all 21 paired comparisons, Holm-corrected |
| `fig_anima_dimensions` | Appendix | ANIMA pass rate per dimension for key arms |
| `fig_manta_pressure` | Appendix | MANTA by pressure type for A0, A1, A2, A3 |
| `fig_dataset` | Appendix | AW demo categories, reply lengths per source, token share vs effective gradient share |

## Caveats baked into the numbers

- One training run per arm; CIs cover question and sampling noise only.
- Runs 1 and 2 (p=100%) have no LLM-judge welfare scores yet (only keyword-scorer numbers, which are not
  comparable); Runs 3 and 4 are not trained. Re-run `./run_all.sh` after they are judged - add their judge
  files and the scripts pick them up automatically.
- p is nominal: the 5/15/30% mixes contain 5.3/16.0/32.0% AW tokens among non-identity tokens.
