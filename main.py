import argparse
from train import run_experiment
from utils import load_config, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    results = run_experiment(cfg)
    print("\nFinal results:")
    for k, v in results.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()