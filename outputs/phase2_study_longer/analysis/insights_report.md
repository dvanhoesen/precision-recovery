# Phase 2 Insights Report

## High-level findings
- Best overall condition: `scalar_fp32_reference` (RMSE=0.014884, MAE=0.010961)
- Best constrained condition: `scalar_nbit_io_hidden_compute` (RMSE=0.026349, MAE=0.020355)
- Best condition by task:
  - linear: `scalar_fp32_reference` | RMSE=0.002387 +/- 0.000915
  - multiplicative: `scalar_nbit_io_hidden_compute` | RMSE=0.006038 +/- 0.000852
  - oscillatory: `scalar_fp32_reference` | RMSE=0.039587 +/- 0.002520
  - polynomial: `scalar_fp32_reference` | RMSE=0.011469 +/- 0.000528

## Primary comparison: `scalar_nbit_io_hidden_compute` vs `two_word_nbit_io_hidden_compute`
- Delta definition: `A - B` for RMSE per paired seed (negative means A is better).
- linear: mean_delta=-0.231074, std=0.037662, A_better_rate=100.0%, B_better_rate=0.0%
- multiplicative: mean_delta=-0.171347, std=0.013760, A_better_rate=100.0%, B_better_rate=0.0%
- oscillatory: mean_delta=-0.448011, std=0.016949, A_better_rate=100.0%, B_better_rate=0.0%
- polynomial: mean_delta=-0.025046, std=0.000913, A_better_rate=100.0%, B_better_rate=0.0%

## Hidden compute penalty: `two_word_nbit_io_only` vs `two_word_nbit_io_hidden_compute`
- Delta definition: `A - B` for RMSE per paired seed (negative means A is better).
- linear: mean_delta=-0.004222, std=0.004163, n=6
- multiplicative: mean_delta=-0.007683, std=0.011869, n=6
- oscillatory: mean_delta=-0.074189, std=0.012884, n=6
- polynomial: mean_delta=0.000789, std=0.000665, n=6

## Artifacts
- Plot: `outputs/phase2_study_longer/analysis/plots/rmse_mae_by_task.png`
- Plot: `outputs/phase2_study_longer/analysis/plots/rmse_boxplot_by_task.png`
- Plot: `outputs/phase2_study_longer/analysis/plots/gap_heatmap_pct_vs_upper.png`
- Plot: `outputs/phase2_study_longer/analysis/plots/pairwise_delta_primary.png`
- Table: `outputs/phase2_study_longer/analysis/tables/best_by_task.csv`
- Table: `outputs/phase2_study_longer/analysis/tables/summary_overall.csv`
- Table: `outputs/phase2_study_longer/analysis/tables/primary_comparison.csv`
- Table: `outputs/phase2_study_longer/analysis/tables/hidden_penalty.csv`
