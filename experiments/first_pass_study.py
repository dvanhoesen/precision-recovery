from copy import deepcopy
import csv
import json
from pathlib import Path
import sys
import statistics

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train import run_experiment
from utils import set_seed


# Edit this section to customize the study.
BASE_CONFIG_PATH = Path("configs/base.yaml")
OUTPUT_ROOT = Path("outputs/first_pass_study")
GENERATED_CONFIG_DIR = OUTPUT_ROOT / "generated_configs"
RUN_OUTPUT_DIR = OUTPUT_ROOT / "run_outputs"
RAW_RESULTS_PATH = OUTPUT_ROOT / "raw_results.jsonl"
SUMMARY_JSON_PATH = OUTPUT_ROOT / "summary.json"
SUMMARY_CSV_PATH = OUTPUT_ROOT / "summary.csv"
RANKING_JSON_PATH = OUTPUT_ROOT / "ranking.json"
RANKING_TXT_PATH = OUTPUT_ROOT / "ranking.txt"

TASKS = ["linear"]
SEEDS = [0, 1, 2]
EPOCHS_OVERRIDE = 20

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

    cfg["experiment_name"] = f"firstpass_{task}_{condition['name']}{loss_suffix}_seed{seed}"

    cfg.setdefault("experiment", {})
    cfg["experiment"]["mode"] = condition["mode"]

    cfg.setdefault("model", {})
    cfg["model"]["name"] = condition["model_name"]

    cfg.setdefault("precision", {})
    cfg["precision"]["quantize_inputs"] = condition["quantize_inputs"]

    cfg.setdefault("data", {})
    cfg["data"]["task"] = task

    if EPOCHS_OVERRIDE is not None:
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


def aggregate_results(results):
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
        rmse_values = [r["rmse"] for r in rows]
        mae_values = [r["mae"] for r in rows]

        hi_values = [r["hi_word_accuracy"] for r in rows if "hi_word_accuracy" in r]
        lo_values = [r["lo_word_accuracy"] for r in rows if "lo_word_accuracy" in r]

        item = {
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
        summary.append(item)

    return summary


def _write_summary_csv(summary):
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
    ]
    with open(SUMMARY_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)


def rank_conditions(summary):
    ranking = sorted(
        summary,
        key=lambda x: (x[RANK_PRIMARY_METRIC], x[RANK_TIEBREAKER_METRIC]),
    )

    constrained_only = [r for r in ranking if r["experiment_mode"] != "float_full"]
    best_overall = ranking[0] if ranking else None
    best_constrained = constrained_only[0] if constrained_only else None

    ranking_payload = {
        "rank_primary_metric": RANK_PRIMARY_METRIC,
        "rank_tiebreaker_metric": RANK_TIEBREAKER_METRIC,
        "best_overall": best_overall,
        "best_constrained": best_constrained,
        "ranking": ranking,
    }
    return ranking_payload


def _write_ranking_text(ranking_payload):
    lines = []
    lines.append("First-pass ranking")
    lines.append(
        f"Sorted by {ranking_payload['rank_primary_metric']} then {ranking_payload['rank_tiebreaker_metric']}"
    )
    lines.append("")

    for idx, row in enumerate(ranking_payload["ranking"], 1):
        lines.append(
            f"{idx:02d}. {row['condition_name']} | mode={row['experiment_mode']} | "
            f"loss={row['loss_variant_name']} | "
            f"model={row['model_name']} | task={row['task']} | "
            f"rmse={row['rmse_mean']:.6f}±{row['rmse_std']:.6f} | "
            f"mae={row['mae_mean']:.6f}±{row['mae_std']:.6f}"
        )

    lines.append("")
    best_overall = ranking_payload["best_overall"]
    if best_overall:
        lines.append(
            f"Best overall: {best_overall['condition_name']} "
            f"(rmse_mean={best_overall['rmse_mean']:.6f})"
        )

    best_constrained = ranking_payload["best_constrained"]
    if best_constrained:
        lines.append(
            f"Best constrained: {best_constrained['condition_name']} "
            f"(rmse_mean={best_constrained['rmse_mean']:.6f})"
        )

    with open(RANKING_TXT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    print("Generating run configs...")
    run_plan = generate_run_plan()
    print(f"Generated {len(run_plan)} configs in {GENERATED_CONFIG_DIR}")

    print("Running experiments...")
    results = run_all_experiments(run_plan)
    print(f"Completed {len(results)} runs")

    print("Aggregating metrics...")
    summary = aggregate_results(results)
    _write_json(SUMMARY_JSON_PATH, summary)
    _write_summary_csv(summary)

    print("Ranking conditions...")
    ranking_payload = rank_conditions(summary)
    _write_json(RANKING_JSON_PATH, ranking_payload)
    _write_ranking_text(ranking_payload)

    best_overall = ranking_payload["best_overall"]
    best_constrained = ranking_payload["best_constrained"]
    print(f"Best overall: {best_overall['condition_name'] if best_overall else 'N/A'}")
    print(f"Best constrained: {best_constrained['condition_name'] if best_constrained else 'N/A'}")
    print(f"Outputs written to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
