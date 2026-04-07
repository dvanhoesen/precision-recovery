from copy import deepcopy
import csv
import json
from pathlib import Path
import statistics
import sys

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train import run_experiment
from utils import set_seed


# Edit this section to customize the phase 2 study.
BASE_CONFIG_PATH = Path("configs/base.yaml")
OUTPUT_ROOT = Path("outputs/phase2_study")
GENERATED_CONFIG_DIR = OUTPUT_ROOT / "generated_configs"
RUN_OUTPUT_DIR = OUTPUT_ROOT / "run_outputs"

RAW_RESULTS_PATH = OUTPUT_ROOT / "raw_results.jsonl"
SUMMARY_BY_TASK_JSON_PATH = OUTPUT_ROOT / "summary_by_task.json"
SUMMARY_BY_TASK_CSV_PATH = OUTPUT_ROOT / "summary_by_task.csv"
SUMMARY_OVERALL_JSON_PATH = OUTPUT_ROOT / "summary_overall.json"
SUMMARY_OVERALL_CSV_PATH = OUTPUT_ROOT / "summary_overall.csv"
RANKING_BY_TASK_JSON_PATH = OUTPUT_ROOT / "ranking_by_task.json"
RANKING_OVERALL_JSON_PATH = OUTPUT_ROOT / "ranking_overall.json"
RANKING_TXT_PATH = OUTPUT_ROOT / "ranking.txt"

TASKS = ["linear", "polynomial", "multiplicative", "oscillatory"]
SEEDS = list(range(10))
EPOCHS_OVERRIDE = 50

LOSS_VARIANTS = [
    {"name": "base", "lambda_hi": 0.2, "lambda_lo": 0.2},
]

CONDITIONS = [
    {
        "name": "scalar_fp32_reference",
        "model_name": "scalar",
        "mode": "float_full",
        "quantize_inputs": False,
    },
    {
        "name": "scalar_nbit_io_hidden_compute",
        "model_name": "scalar",
        "mode": "quant_full",
        "quantize_inputs": True,
    },
    {
        "name": "two_word_nbit_io_hidden_compute",
        "model_name": "two_word",
        "mode": "quant_full",
        "quantize_inputs": True,
    },
    {
        "name": "two_word_nbit_io_only",
        "model_name": "two_word",
        "mode": "quant_io",
        "quantize_inputs": True,
    },
]

UPPER_BOUND_CONDITION = "scalar_fp32_reference"
RANK_PRIMARY_METRIC = "rmse_mean"
RANK_TIEBREAKER_METRIC = "mae_mean"


def _mean(values):
    return statistics.mean(values)


def _std(values):
    if len(values) <= 1:
        return 0.0
    return statistics.stdev(values)


def _load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_yaml(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def _write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _build_run_config(base_cfg, condition, task, seed, loss_variant):
    cfg = deepcopy(base_cfg)
    cfg["seed"] = seed

    loss_suffix = ""
    if len(LOSS_VARIANTS) > 1:
        loss_suffix = f"_{loss_variant['name']}"

    cfg["experiment_name"] = f"phase2_{task}_{condition['name']}{loss_suffix}_seed{seed}"

    cfg.setdefault("experiment", {})
    cfg["experiment"]["mode"] = condition["mode"]

    cfg.setdefault("model", {})
    cfg["model"]["name"] = condition["model_name"]

    cfg.setdefault("precision", {})
    cfg["precision"]["quantize_inputs"] = condition["quantize_inputs"]

    cfg.setdefault("data", {})
    cfg["data"]["task"] = task

    cfg.setdefault("train", {})
    cfg["train"]["epochs"] = EPOCHS_OVERRIDE

    cfg.setdefault("output", {})
    cfg["output"]["save_dir"] = str(RUN_OUTPUT_DIR)

    cfg.setdefault("loss", {})
    cfg["loss"]["lambda_hi"] = loss_variant["lambda_hi"]
    cfg["loss"]["lambda_lo"] = loss_variant["lambda_lo"]
    return cfg


def generate_run_plan():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    GENERATED_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    base_cfg = _load_yaml(BASE_CONFIG_PATH)
    run_plan = []

    for task in TASKS:
        for condition in CONDITIONS:
            for loss_variant in LOSS_VARIANTS:
                for seed in SEEDS:
                    cfg = _build_run_config(base_cfg, condition, task, seed, loss_variant)
                    cfg_path = GENERATED_CONFIG_DIR / f"{cfg['experiment_name']}.yaml"
                    _write_yaml(cfg_path, cfg)
                    run_plan.append(
                        {
                            "config_path": str(cfg_path),
                            "condition_name": condition["name"],
                            "loss_variant_name": loss_variant["name"],
                            "task": task,
                            "seed": seed,
                            "cfg": cfg,
                        }
                    )
    return run_plan


def run_all_experiments(run_plan):
    if RAW_RESULTS_PATH.exists():
        RAW_RESULTS_PATH.unlink()

    results = []
    total = len(run_plan)

    for idx, item in enumerate(run_plan, 1):
        cfg = item["cfg"]
        print(f"[{idx}/{total}] Running {cfg['experiment_name']}")

        set_seed(cfg["seed"])
        metrics = run_experiment(cfg)

        record = {
            "experiment_name": cfg["experiment_name"],
            "condition_name": item["condition_name"],
            "loss_variant_name": item["loss_variant_name"],
            "seed": item["seed"],
            "task": item["task"],
            "config_path": item["config_path"],
            **metrics,
        }
        results.append(record)

        with open(RAW_RESULTS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    return results


def _aggregate_group(rows, condition_name, loss_variant_name, experiment_mode, model_name, task):
    rmse_values = [r["rmse"] for r in rows]
    mae_values = [r["mae"] for r in rows]

    hi_values = [r["hi_word_accuracy"] for r in rows if "hi_word_accuracy" in r]
    lo_values = [r["lo_word_accuracy"] for r in rows if "lo_word_accuracy" in r]

    return {
        "condition_name": condition_name,
        "loss_variant_name": loss_variant_name,
        "experiment_mode": experiment_mode,
        "model_name": model_name,
        "task": task,
        "n": len(rows),
        "rmse_mean": _mean(rmse_values),
        "rmse_std": _std(rmse_values),
        "mae_mean": _mean(mae_values),
        "mae_std": _std(mae_values),
        "hi_word_accuracy_mean": _mean(hi_values) if hi_values else None,
        "hi_word_accuracy_std": _std(hi_values) if hi_values else None,
        "lo_word_accuracy_mean": _mean(lo_values) if lo_values else None,
        "lo_word_accuracy_std": _std(lo_values) if lo_values else None,
    }


def _add_upper_bound_gap(summary_rows):
    upper_by_task = {}
    for row in summary_rows:
        if row["condition_name"] == UPPER_BOUND_CONDITION:
            key = (row["task"], row["loss_variant_name"])
            upper_by_task[key] = row["rmse_mean"]

    for row in summary_rows:
        upper = upper_by_task.get((row["task"], row["loss_variant_name"]))
        if upper is None:
            row["rmse_gap_abs_vs_upper"] = None
            row["rmse_gap_pct_vs_upper"] = None
            continue

        gap = row["rmse_mean"] - upper
        row["rmse_gap_abs_vs_upper"] = gap
        row["rmse_gap_pct_vs_upper"] = 100.0 * gap / upper if upper > 0 else None


def aggregate_by_task(results):
    grouped = {}
    for row in results:
        key = (
            row["condition_name"],
            row["loss_variant_name"],
            row["experiment_mode"],
            row["model_name"],
            row["task"],
        )
        grouped.setdefault(key, []).append(row)

    summary = []
    for key, rows in sorted(grouped.items()):
        condition_name, loss_variant_name, experiment_mode, model_name, task = key
        summary.append(
            _aggregate_group(rows, condition_name, loss_variant_name, experiment_mode, model_name, task)
        )

    _add_upper_bound_gap(summary)
    return summary


def aggregate_overall(results):
    grouped = {}
    for row in results:
        key = (
            row["condition_name"],
            row["loss_variant_name"],
            row["experiment_mode"],
            row["model_name"],
        )
        grouped.setdefault(key, []).append(row)

    summary = []
    for key, rows in sorted(grouped.items()):
        condition_name, loss_variant_name, experiment_mode, model_name = key
        summary.append(
            _aggregate_group(
                rows,
                condition_name=condition_name,
                loss_variant_name=loss_variant_name,
                experiment_mode=experiment_mode,
                model_name=model_name,
                task="all_tasks",
            )
        )

    upper_by_loss = {
        r["loss_variant_name"]: r["rmse_mean"]
        for r in summary
        if r["condition_name"] == UPPER_BOUND_CONDITION
    }
    for row in summary:
        upper = upper_by_loss.get(row["loss_variant_name"])
        if upper is None:
            row["rmse_gap_abs_vs_upper"] = None
            row["rmse_gap_pct_vs_upper"] = None
            continue
        gap = row["rmse_mean"] - upper
        row["rmse_gap_abs_vs_upper"] = gap
        row["rmse_gap_pct_vs_upper"] = 100.0 * gap / upper if upper > 0 else None

    return summary


def _write_summary_csv(path, summary):
    fieldnames = [
        "condition_name",
        "loss_variant_name",
        "experiment_mode",
        "model_name",
        "task",
        "n",
        "rmse_mean",
        "rmse_std",
        "mae_mean",
        "mae_std",
        "hi_word_accuracy_mean",
        "hi_word_accuracy_std",
        "lo_word_accuracy_mean",
        "lo_word_accuracy_std",
        "rmse_gap_abs_vs_upper",
        "rmse_gap_pct_vs_upper",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)


def rank_by_task(summary_by_task):
    grouped = {}
    for row in summary_by_task:
        grouped.setdefault(row["task"], []).append(row)

    ranking = {}
    for task, rows in grouped.items():
        ranking[task] = sorted(
            rows,
            key=lambda x: (x[RANK_PRIMARY_METRIC], x[RANK_TIEBREAKER_METRIC]),
        )
    return ranking


def rank_overall(summary_overall, ranking_by_task):
    overall_ranking = sorted(
        summary_overall,
        key=lambda x: (x[RANK_PRIMARY_METRIC], x[RANK_TIEBREAKER_METRIC]),
    )

    rank_positions = {}
    for task, task_rows in ranking_by_task.items():
        for idx, row in enumerate(task_rows, 1):
            key = (row["condition_name"], row["loss_variant_name"])
            rank_positions.setdefault(key, []).append(idx)

    payload = {
        "rank_primary_metric": RANK_PRIMARY_METRIC,
        "rank_tiebreaker_metric": RANK_TIEBREAKER_METRIC,
        "best_overall": overall_ranking[0] if overall_ranking else None,
        "best_constrained": next(
            (r for r in overall_ranking if r["experiment_mode"] != "float_full"),
            None,
        ),
        "ranking": [],
    }

    for row in overall_ranking:
        key = (row["condition_name"], row["loss_variant_name"])
        task_ranks = rank_positions.get(key, [])
        entry = dict(row)
        entry["task_ranks"] = task_ranks
        entry["avg_rank_across_tasks"] = _mean(task_ranks) if task_ranks else None
        payload["ranking"].append(entry)

    return payload


def _write_ranking_text(ranking_by_task, ranking_overall):
    lines = []
    lines.append("Phase 2 ranking")
    lines.append(
        f"Sorted by {ranking_overall['rank_primary_metric']} then {ranking_overall['rank_tiebreaker_metric']}"
    )
    lines.append("")

    lines.append("Per-task ranking")
    for task in sorted(ranking_by_task):
        lines.append(f"- Task: {task}")
        for idx, row in enumerate(ranking_by_task[task], 1):
            lines.append(
                f"  {idx:02d}. {row['condition_name']} | mode={row['experiment_mode']} | "
                f"loss={row['loss_variant_name']} | "
                f"model={row['model_name']} | rmse={row['rmse_mean']:.6f}±{row['rmse_std']:.6f} | "
                f"mae={row['mae_mean']:.6f}±{row['mae_std']:.6f}"
            )
        lines.append("")

    lines.append("Overall ranking")
    for idx, row in enumerate(ranking_overall["ranking"], 1):
        lines.append(
            f"{idx:02d}. {row['condition_name']} | mode={row['experiment_mode']} | "
            f"loss={row['loss_variant_name']} | "
            f"model={row['model_name']} | rmse={row['rmse_mean']:.6f}±{row['rmse_std']:.6f} | "
            f"mae={row['mae_mean']:.6f}±{row['mae_std']:.6f} | "
            f"avg_task_rank={row['avg_rank_across_tasks']:.2f}"
        )

    lines.append("")
    best_overall = ranking_overall["best_overall"]
    if best_overall:
        lines.append(
            f"Best overall: {best_overall['condition_name']} "
            f"(rmse_mean={best_overall['rmse_mean']:.6f})"
        )

    best_constrained = ranking_overall["best_constrained"]
    if best_constrained:
        lines.append(
            f"Best constrained: {best_constrained['condition_name']} "
            f"(rmse_mean={best_constrained['rmse_mean']:.6f})"
        )

    with open(RANKING_TXT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    print("Generating phase 2 run configs...")
    run_plan = generate_run_plan()
    print(f"Generated {len(run_plan)} configs in {GENERATED_CONFIG_DIR}")

    print("Running experiments...")
    results = run_all_experiments(run_plan)
    print(f"Completed {len(results)} runs")

    print("Aggregating by task...")
    summary_by_task = aggregate_by_task(results)
    _write_json(SUMMARY_BY_TASK_JSON_PATH, summary_by_task)
    _write_summary_csv(SUMMARY_BY_TASK_CSV_PATH, summary_by_task)

    print("Aggregating overall...")
    summary_overall = aggregate_overall(results)
    _write_json(SUMMARY_OVERALL_JSON_PATH, summary_overall)
    _write_summary_csv(SUMMARY_OVERALL_CSV_PATH, summary_overall)

    print("Ranking conditions...")
    ranking_by_task = rank_by_task(summary_by_task)
    ranking_overall = rank_overall(summary_overall, ranking_by_task)
    _write_json(RANKING_BY_TASK_JSON_PATH, ranking_by_task)
    _write_json(RANKING_OVERALL_JSON_PATH, ranking_overall)
    _write_ranking_text(ranking_by_task, ranking_overall)

    best_overall = ranking_overall["best_overall"]
    best_constrained = ranking_overall["best_constrained"]
    print(f"Best overall: {best_overall['condition_name'] if best_overall else 'N/A'}")
    print(f"Best constrained: {best_constrained['condition_name'] if best_constrained else 'N/A'}")
    print(f"Outputs written to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
