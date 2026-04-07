from utils import parse_args, load_config, set_seed, get_device
from train import run_training


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # Allow a few command-line overrides for quick experiments.
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.task is not None:
        cfg["data"]["task"] = args.task
    if args.model is not None:
        cfg["model"]["name"] = args.model
    if args.input_bits is not None:
        cfg["precision"]["input_bits"] = args.input_bits
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir

    set_seed(cfg["seed"])
    device = get_device()
    print(f"Using device: {device}")
    print(f"Running task={cfg['data']['task']} model={cfg['model']['name']} seed={cfg['seed']}")

    results = run_training(cfg, device=device)
    print("Final test metrics:")
    for key, value in results.items():
        print(f"  {key}: {value:.6f}")


if __name__ == "__main__":
    main()
