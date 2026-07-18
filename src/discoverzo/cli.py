from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .experiments import ExperimentConfig, run_all
from .plotting import plot_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DISCOVER-ZO experiments")
    parser.add_argument("--config", default="configs/quick.yaml")
    parser.add_argument("--output", default="results")
    parser.add_argument("--figures", default="figures")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    cfg = ExperimentConfig(**values)
    run_all(cfg, args.output)
    plot_all(args.output, args.figures)
    print(f"Completed experiments in {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
