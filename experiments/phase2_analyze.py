import json
import math
from pathlib import Path
import statistics

import matplotlib.pyplot as plt


# Edit these paths/settings if needed.
PHASE2_ROOT = Path("outputs/phase2_study_longer")
RAW_RESULTS_PATH = PHASE2_ROOT / "raw_results.jsonl"
SUMMARY_BY_TASK_PATH = PHASE2_ROOT / "summary_by_task.json"
SUMMARY_OVERALL_PATH = PHASE2_ROOT / "summary_overall.json"

ANALYSIS_DIR = PHASE2_ROOT / "analysis"
PLOTS_DIR = ANALYSIS_DIR / "plots"
TABLES_DIR = ANALYSIS_DIR / "tables"
REPORT_PATH = ANALYSIS_DIR / "insights_report.md"

PRIMARY_COMPARISON_A = "scalar_nbit_io_hidden_compute"
PRIMARY_COMPARISON_B = "two_word_nbit_io_hidden_compute"
HIDDEN_COMPARISON_A = "two_word_nbit_io_only"
HIDDEN_COMPARISON_B = "two_word_nbit_io_hidden_compute"
UPPER_BOUND_CONDITION = "scalar_fp32_reference"


def _mean(values):
    return statistics.mean(values)


def _std(values):
    if len(values) <= 1:
        return 0.0
    return statistics.stdev(values)


def _ci95(values):
    if not values:
        return 0.0
    return 1.96 * _std(values) / math.sqrt(len(values))


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _group_summary_by_task(summary_by_task):
    grouped = {}
    for row in summary_by_task:
        grouped.setdefault(row["task"], []).append(row)
    for task in grouped:
        grouped[task] = sorted(grouped[task], key=lambda x: x["rmse_mean"])
    return grouped


def _condition_label(row):
    return f"{row['condition_name']}\n({row['experiment_mode']}, {row['model_name']})"


def plot_rmse_mae_by_task(summary_by_task):
    grouped = _group_summary_by_task(summary_by_task)
    tasks = sorted(grouped)

    fig, axes = plt.subplots(len(tasks), 2, figsize=(16, 4 * len(tasks)))
    if len(tasks) == 1:
        axes = [axes]

    for i, task in enumerate(tasks):
        rows = grouped[task]
        labels = [_condition_label(r) for r in rows]
        x = list(range(len(rows)))

        rmse_vals = [r["rmse_mean"] for r in rows]
        rmse_err = [_ci95([r["rmse_mean"]] * r["n"]) if r.get("n") else 0.0 for r in rows]
        mae_vals = [r["mae_mean"] for r in rows]
        mae_err = [_ci95([r["mae_mean"]] * r["n"]) if r.get("n") else 0.0 for r in rows]

        # Use summary std-based CI rather than duplicating means.
        rmse_err = [1.96 * r["rmse_std"] / math.sqrt(r["n"]) for r in rows]
        mae_err = [1.96 * r["mae_std"] / math.sqrt(r["n"]) for r in rows]

        ax_rmse = axes[i][0]
        ax_rmse.bar(x, rmse_vals, yerr=rmse_err, capsize=4)
        ax_rmse.set_title(f"{task}: RMSE (mean +/- 95% CI)")
        ax_rmse.set_xticks(x)
        ax_rmse.set_xticklabels(labels, rotation=25, ha="right")
        ax_rmse.set_ylabel("RMSE")
        ax_rmse.grid(axis="y", alpha=0.3)

        ax_mae = axes[i][1]
        ax_mae.bar(x, mae_vals, yerr=mae_err, capsize=4)
        ax_mae.set_title(f"{task}: MAE (mean +/- 95% CI)")
        ax_mae.set_xticks(x)
        ax_mae.set_xticklabels(labels, rotation=25, ha="right")
        ax_mae.set_ylabel("MAE")
        ax_mae.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = PLOTS_DIR / "rmse_mae_by_task.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def _build_seed_metric_map(raw_rows, metric="rmse"):
    seed_map = {}
    for row in raw_rows:
        key = (
            row["task"],
            row["loss_variant_name"],
            row["condition_name"],
            row["seed"],
        )
        seed_map[key] = row[metric]
    return seed_map


def _paired_deltas(raw_rows, cond_a, cond_b, metric="rmse"):
    seed_map = _build_seed_metric_map(raw_rows, metric=metric)
    keys_a = [k for k in seed_map if k[2] == cond_a]

    out = {}
    for task, loss_name, _, seed in keys_a:
        kb = (task, loss_name, cond_b, seed)
        ka = (task, loss_name, cond_a, seed)
        if kb not in seed_map:
            continue
        out.setdefault(task, []).append(seed_map[ka] - seed_map[kb])
    return out


def plot_pairwise_delta(raw_rows):
    deltas = _paired_deltas(raw_rows, PRIMARY_COMPARISON_A, PRIMARY_COMPARISON_B, metric="rmse")
    tasks = sorted(deltas)

    fig, axes = plt.subplots(1, len(tasks), figsize=(4.5 * len(tasks), 4), sharey=True)
    if len(tasks) == 1:
        axes = [axes]

    for ax, task in zip(axes, tasks):
        vals = deltas[task]
        x = list(range(len(vals)))
        ax.scatter(x, vals)
        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.axhline(_mean(vals), linestyle="-", linewidth=1.5)
        ax.set_title(task)
        ax.set_xlabel("paired seed index")
        ax.set_ylabel(f"{PRIMARY_COMPARISON_A} - {PRIMARY_COMPARISON_B} (RMSE)")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    out = PLOTS_DIR / "pairwise_delta_primary.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out, deltas


def plot_rmse_boxplot(raw_rows):
    tasks = sorted({r["task"] for r in raw_rows})
    conditions = sorted({r["condition_name"] for r in raw_rows})

    fig, axes = plt.subplots(len(tasks), 1, figsize=(14, 4 * len(tasks)), sharey=False)
    if len(tasks) == 1:
        axes = [axes]

    for ax, task in zip(axes, tasks):
        data = []
        labels = []
        for cond in conditions:
            vals = [r["rmse"] for r in raw_rows if r["task"] == task and r["condition_name"] == cond]
            if vals:
                data.append(vals)
                labels.append(cond)
        ax.boxplot(data, tick_labels=labels, showmeans=True)
        ax.set_title(f"{task}: RMSE distribution across seeds")
        ax.set_ylabel("RMSE")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = PLOTS_DIR / "rmse_boxplot_by_task.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_gap_heatmap(summary_by_task):
    tasks = sorted({r["task"] for r in summary_by_task})
    conditions = sorted({r["condition_name"] for r in summary_by_task if r["condition_name"] != UPPER_BOUND_CONDITION})

    matrix = []
    for task in tasks:
        row_vals = []
        for cond in conditions:
            match = [
                r for r in summary_by_task
                if r["task"] == task and r["condition_name"] == cond
            ]
            row_vals.append(match[0]["rmse_gap_pct_vs_upper"] if match else float("nan"))
        matrix.append(row_vals)

    fig, ax = plt.subplots(figsize=(1.8 * len(conditions) + 2, 1.2 * len(tasks) + 2))
    im = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(list(range(len(conditions))))
    ax.set_xticklabels(conditions, rotation=25, ha="right")
    ax.set_yticks(list(range(len(tasks))))
    ax.set_yticklabels(tasks)
    ax.set_title("RMSE gap vs upper bound (%)")

    for i in range(len(tasks)):
        for j in range(len(conditions)):
            val = matrix[i][j]
            if isinstance(val, float) and math.isnan(val):
                text = "NA"
            else:
                text = f"{val:.1f}%"
            ax.text(j, i, text, ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out = PLOTS_DIR / "gap_heatmap_pct_vs_upper.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def _task_best(summary_by_task):
    grouped = _group_summary_by_task(summary_by_task)
    return {task: rows[0] for task, rows in grouped.items() if rows}


def _primary_comparison_insight(raw_rows):
    deltas = _paired_deltas(raw_rows, PRIMARY_COMPARISON_A, PRIMARY_COMPARISON_B, metric="rmse")
    insights = {}
    for task, vals in deltas.items():
        if not vals:
            continue
        wins_a = sum(1 for v in vals if v < 0)
        wins_b = sum(1 for v in vals if v > 0)
        ties = sum(1 for v in vals if v == 0)
        insights[task] = {
            "mean_delta": _mean(vals),
            "std_delta": _std(vals),
            "n": len(vals),
            "wins_a": wins_a,
            "wins_b": wins_b,
            "ties": ties,
            "a_better_rate": wins_a / len(vals),
            "b_better_rate": wins_b / len(vals),
        }
    return insights


def _hidden_penalty_insight(raw_rows):
    deltas = _paired_deltas(raw_rows, HIDDEN_COMPARISON_A, HIDDEN_COMPARISON_B, metric="rmse")
    summary = {}
    for task, vals in deltas.items():
        if not vals:
            continue
        summary[task] = {
            "mean_delta": _mean(vals),
            "std_delta": _std(vals),
            "n": len(vals),
        }
    return summary


def _write_csv(path, rows, fieldnames):
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_insight_tables(summary_by_task, summary_overall, primary_insight, hidden_insight):
    best_rows = []
    best_by_task = _task_best(summary_by_task)
    for task in sorted(best_by_task):
        row = best_by_task[task]
        best_rows.append(
            {
                "task": task,
                "best_condition": row["condition_name"],
                "mode": row["experiment_mode"],
                "model": row["model_name"],
                "rmse_mean": row["rmse_mean"],
                "rmse_std": row["rmse_std"],
                "mae_mean": row["mae_mean"],
                "mae_std": row["mae_std"],
            }
        )

    _write_csv(
        TABLES_DIR / "best_by_task.csv",
        best_rows,
        ["task", "best_condition", "mode", "model", "rmse_mean", "rmse_std", "mae_mean", "mae_std"],
    )

    _write_csv(
        TABLES_DIR / "summary_overall.csv",
        summary_overall,
        list(summary_overall[0].keys()),
    )

    primary_rows = []
    for task in sorted(primary_insight):
        s = primary_insight[task]
        primary_rows.append({"task": task, **s})
    _write_csv(
        TABLES_DIR / "primary_comparison.csv",
        primary_rows,
        [
            "task",
            "mean_delta",
            "std_delta",
            "n",
            "wins_a",
            "wins_b",
            "ties",
            "a_better_rate",
            "b_better_rate",
        ],
    )

    hidden_rows = []
    for task in sorted(hidden_insight):
        hidden_rows.append({"task": task, **hidden_insight[task]})
    _write_csv(
        TABLES_DIR / "hidden_penalty.csv",
        hidden_rows,
        ["task", "mean_delta", "std_delta", "n"],
    )


def write_report(summary_by_task, summary_overall, primary_insight, hidden_insight, plot_paths):
    best_by_task = _task_best(summary_by_task)
    overall_best = sorted(summary_overall, key=lambda x: (x["rmse_mean"], x["mae_mean"]))[0]
    constrained = [r for r in summary_overall if r["experiment_mode"] != "float_full"]
    best_constrained = sorted(constrained, key=lambda x: (x["rmse_mean"], x["mae_mean"]))[0]

    lines = []
    lines.append("# Phase 2 Insights Report")
    lines.append("")
    lines.append("## High-level findings")
    lines.append(
        f"- Best overall condition: `{overall_best['condition_name']}` (RMSE={overall_best['rmse_mean']:.6f}, MAE={overall_best['mae_mean']:.6f})"
    )
    lines.append(
        f"- Best constrained condition: `{best_constrained['condition_name']}` (RMSE={best_constrained['rmse_mean']:.6f}, MAE={best_constrained['mae_mean']:.6f})"
    )
    lines.append("- Best condition by task:")
    for task in sorted(best_by_task):
        row = best_by_task[task]
        lines.append(
            f"  - {task}: `{row['condition_name']}` | RMSE={row['rmse_mean']:.6f} +/- {row['rmse_std']:.6f}"
        )

    lines.append("")
    lines.append(f"## Primary comparison: `{PRIMARY_COMPARISON_A}` vs `{PRIMARY_COMPARISON_B}`")
    lines.append("- Delta definition: `A - B` for RMSE per paired seed (negative means A is better).")
    for task in sorted(primary_insight):
        s = primary_insight[task]
        lines.append(
            f"- {task}: mean_delta={s['mean_delta']:.6f}, std={s['std_delta']:.6f}, "
            f"A_better_rate={100*s['a_better_rate']:.1f}%, B_better_rate={100*s['b_better_rate']:.1f}%"
        )

    lines.append("")
    lines.append(f"## Hidden compute penalty: `{HIDDEN_COMPARISON_A}` vs `{HIDDEN_COMPARISON_B}`")
    lines.append("- Delta definition: `A - B` for RMSE per paired seed (negative means A is better).")
    for task in sorted(hidden_insight):
        s = hidden_insight[task]
        lines.append(
            f"- {task}: mean_delta={s['mean_delta']:.6f}, std={s['std_delta']:.6f}, n={s['n']}"
        )

    lines.append("")
    lines.append("## Artifacts")
    for p in plot_paths:
        lines.append(f"- Plot: `{p}`")
    lines.append(f"- Table: `{TABLES_DIR / 'best_by_task.csv'}`")
    lines.append(f"- Table: `{TABLES_DIR / 'summary_overall.csv'}`")
    lines.append(f"- Table: `{TABLES_DIR / 'primary_comparison.csv'}`")
    lines.append(f"- Table: `{TABLES_DIR / 'hidden_penalty.csv'}`")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    summary_by_task = _load_json(SUMMARY_BY_TASK_PATH)
    summary_overall = _load_json(SUMMARY_OVERALL_PATH)
    raw_rows = _load_jsonl(RAW_RESULTS_PATH)

    plot_paths = []
    plot_paths.append(plot_rmse_mae_by_task(summary_by_task))
    boxplot_path = plot_rmse_boxplot(raw_rows)
    plot_paths.append(boxplot_path)
    heatmap_path = plot_gap_heatmap(summary_by_task)
    plot_paths.append(heatmap_path)
    pair_path, _ = plot_pairwise_delta(raw_rows)
    plot_paths.append(pair_path)

    primary_insight = _primary_comparison_insight(raw_rows)
    hidden_insight = _hidden_penalty_insight(raw_rows)

    build_insight_tables(summary_by_task, summary_overall, primary_insight, hidden_insight)
    write_report(summary_by_task, summary_overall, primary_insight, hidden_insight, plot_paths)

    print(f"Wrote analysis report: {REPORT_PATH}")
    print(f"Wrote plots to: {PLOTS_DIR}")
    print(f"Wrote tables to: {TABLES_DIR}")


if __name__ == "__main__":
    main()
