# Phase 2 Insights Report

## High-level findings
- Best overall condition: `scalar_fp32_reference` (RMSE=0.023628, MAE=0.018164)
- Best constrained condition: `scalar_nbit_io_hidden_compute` (RMSE=0.039982, MAE=0.031470)
- Best condition by task:
  - linear: `scalar_fp32_reference` | RMSE=0.002959 +/- 0.000484
  - multiplicative: `scalar_fp32_reference` | RMSE=0.007139 +/- 0.002037
  - oscillatory: `scalar_fp32_reference` | RMSE=0.072112 +/- 0.029444
  - polynomial: `scalar_fp32_reference` | RMSE=0.012303 +/- 0.000619

## Primary comparison: `scalar_nbit_io_hidden_compute` vs `two_word_nbit_io_hidden_compute`
- Delta definition: `A - B` for RMSE per paired seed (negative means A is better).
- linear: mean_delta=-0.185448, std=0.049799, A_better_rate=100.0%, B_better_rate=0.0%
- multiplicative: mean_delta=-0.182762, std=0.008735, A_better_rate=100.0%, B_better_rate=0.0%
- oscillatory: mean_delta=-0.434623, std=0.009023, A_better_rate=100.0%, B_better_rate=0.0%
- polynomial: mean_delta=-0.026058, std=0.001659, A_better_rate=100.0%, B_better_rate=0.0%

## Hidden compute penalty: `two_word_nbit_io_only` vs `two_word_nbit_io_hidden_compute`
- Delta definition: `A - B` for RMSE per paired seed (negative means A is better).
- linear: mean_delta=0.040805, std=0.053402, n=10
- multiplicative: mean_delta=-0.011369, std=0.008598, n=10
- oscillatory: mean_delta=-0.050849, std=0.006875, n=10
- polynomial: mean_delta=-0.000253, std=0.000854, n=10

## Artifacts
- Plot: `outputs/phase2_study/analysis/plots/rmse_mae_by_task.png`
- Plot: `outputs/phase2_study/analysis/plots/rmse_boxplot_by_task.png`
- Plot: `outputs/phase2_study/analysis/plots/gap_heatmap_pct_vs_upper.png`
- Plot: `outputs/phase2_study/analysis/plots/pairwise_delta_primary.png`
- Table: `outputs/phase2_study/analysis/tables/best_by_task.csv`
- Table: `outputs/phase2_study/analysis/tables/summary_overall.csv`
- Table: `outputs/phase2_study/analysis/tables/primary_comparison.csv`
- Table: `outputs/phase2_study/analysis/tables/hidden_penalty.csv`
