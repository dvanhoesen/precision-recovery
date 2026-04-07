import json
from pathlib import Path

import matplotlib.pyplot as plt


# Edit this section to customize plotting behavior.
SUMMARY_PATH = Path("outputs/first_pass_study/summary.json")
PLOTS_DIR = Path("outputs/first_pass_study/plots")
RMSE_PLOT_PATH = PLOTS_DIR / "rmse_by_condition.png"
MAE_PLOT_PATH = PLOTS_DIR / "mae_by_condition.png"
COMBINED_PLOT_PATH = PLOTS_DIR / "rmse_mae_combined.png"

SORT_BY = "rmse_mean"
INCLUDE_FLOAT_FULL = True


def _load_summary(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected list in summary JSON, got {type(data)}")
    if not data:
        raise ValueError(f"Summary JSON is empty: {path}")
    return data


def _label(row):
    return f"{row['condition_name']}\n({row['experiment_mode']}, {row['model_name']})"


def _filter_rows(rows):
    if INCLUDE_FLOAT_FULL:
        return rows
    return [r for r in rows if r["experiment_mode"] != "float_full"]


def _sort_rows(rows):
    return sorted(rows, key=lambda x: x[SORT_BY])


def _plot_metric(rows, mean_key, std_key, title, y_label, out_path):
    labels = [_label(r) for r in rows]
    means = [r[mean_key] for r in rows]
    stds = [r[std_key] for r in rows]

    width = max(10, 1.4 * len(rows))
    fig, ax = plt.subplots(figsize=(width, 6))
    x = list(range(len(rows)))

    ax.bar(x, means, yerr=stds, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_combined(rows, out_path):
    labels = [_label(r) for r in rows]
    rmse_mean = [r["rmse_mean"] for r in rows]
    rmse_std = [r["rmse_std"] for r in rows]
    mae_mean = [r["mae_mean"] for r in rows]
    mae_std = [r["mae_std"] for r in rows]

    width = max(10, 1.5 * len(rows))
    fig, axes = plt.subplots(2, 1, figsize=(width, 10), sharex=True)
    x = list(range(len(rows)))

    axes[0].bar(x, rmse_mean, yerr=rmse_std, capsize=4)
    axes[0].set_title("RMSE by Condition")
    axes[0].set_ylabel("RMSE")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x, mae_mean, yerr=mae_std, capsize=4)
    axes[1].set_title("MAE by Condition")
    axes[1].set_ylabel("MAE")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha="right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    rows = _load_summary(SUMMARY_PATH)
    rows = _filter_rows(rows)
    rows = _sort_rows(rows)

    if not rows:
        raise ValueError("No rows left after filtering")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    _plot_metric(
        rows,
        mean_key="rmse_mean",
        std_key="rmse_std",
        title="RMSE by Condition",
        y_label="RMSE",
        out_path=RMSE_PLOT_PATH,
    )

    _plot_metric(
        rows,
        mean_key="mae_mean",
        std_key="mae_std",
        title="MAE by Condition",
        y_label="MAE",
        out_path=MAE_PLOT_PATH,
    )

    _plot_combined(rows, COMBINED_PLOT_PATH)

    print(f"Loaded summary: {SUMMARY_PATH}")
    print(f"Saved RMSE plot: {RMSE_PLOT_PATH}")
    print(f"Saved MAE plot: {MAE_PLOT_PATH}")
    print(f"Saved combined plot: {COMBINED_PLOT_PATH}")


if __name__ == "__main__":
    main()
