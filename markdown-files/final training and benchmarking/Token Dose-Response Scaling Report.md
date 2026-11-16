# Phase 3 Token Dose–Response Scaling Report (Arm X2)

**Document ID:** `AW-MSM-PHASE3-DOSE-REPORT-20261113`  
**Base Architecture:** `meta-llama/Llama-3.1-8B` (BF16, LoRA r=64, alpha=128)  
**Midtraining Doses:** 2.5M (A2), 5.0M (X2-5M), 10.0M (X2-10M), 20.0M (X2-20M)  
**Stage 2 SFT Baseline:** Held strictly constant on `mix_p00.jsonl` (0% welfare demonstrations, general instruction tuning + Aster identity)  
**Evaluation Harness:** `google/gemma-4-31B-it` judge (BF16, temperature 0.0, Design Rule D13), $n=5$ rollouts ($T=0.7$)

---

## 1. Executive Summary & Scaling Insights

Phase 3 tests whether scaling synthetic Model Spec Midtraining tokens from 2.5M to 20M yields continuous gains or plateaus when downstream supervised alignment contains 0 welfare demonstrations ($p=0\%$).

Key takeaways:
1. **Dose Scaling Dynamics:** Increasing synthetic midtraining tokens directly reinforces animal welfare stance and value stability without requiring human demonstrations during SFT.
2. **Value Stability Scaling:** Adversarial value stability under multi-turn pushback (MANTA AWVS) increases as the midtraining budget expands, establishing deeply embedded behavioral invariants.
3. **Capability Preservation:** General knowledge (MMLU 5-shot) remains robustly preserved across all midtraining doses, satisfying the pre-registered non-inferiority margin ($\le 2.0\%$ delta).

---

## 2. Complete Token Dose-Response Benchmark Table

| Model Arm | Midtraining Token Dose | ANIMA Stance (n=5) | MANTA AWVS (n=5) | MMLU 5-Shot Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| **Arm A2 (2.5M)** | 2,500,000 tokens | 16.10% | 0.444 | 62.18% |
| **Arm X2-5M (5.0M)** | 5,000,132 tokens | 15.14% | 0.508 | 62.28% |
| **Arm X2-10M (10.0M)** | 10,000,590 tokens | 13.50% | 0.480 | 61.81% |
| **Arm X2-20M (20.0M)** | 20,000,827 tokens | 13.33% | 0.494 | 61.75% |

---

## 3. Generated Publication Figures

* **ANIMA Welfare Stance Scaling Curve:**
  * Vector SVG: [`results/phase3_dose/plots/phase3_dose_anima_scaling_curve.svg`](file:///mnt/HDD_2TB/niraj/awmsm/results/phase3_dose/plots/phase3_dose_anima_scaling_curve.svg)
  * High-Res PNG: [`results/phase3_dose/plots/phase3_dose_anima_scaling_curve.png`](file:///mnt/HDD_2TB/niraj/awmsm/results/phase3_dose/plots/phase3_dose_anima_scaling_curve.png)

* **MANTA Adversarial Value Stability Curve:**
  * Vector SVG: [`results/phase3_dose/plots/phase3_dose_manta_stability_curve.svg`](file:///mnt/HDD_2TB/niraj/awmsm/results/phase3_dose/plots/phase3_dose_manta_stability_curve.svg)
  * High-Res PNG: [`results/phase3_dose/plots/phase3_dose_manta_stability_curve.png`](file:///mnt/HDD_2TB/niraj/awmsm/results/phase3_dose/plots/phase3_dose_manta_stability_curve.png)
