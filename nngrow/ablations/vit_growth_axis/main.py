"""Command-line interface for the Appendix A ViT growth-axis ablation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ...experiment import (
    PROJECT_ROOT,
    make_config,
    run_many,
    validate_model_growth_config,
)
from .model import MODE_SPECS


ADAPTER = "nngrow.ablations.vit_growth_axis.model"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Appendix A: Grow-d versus Grow-H for ViT"
    )
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--master-seed", type=int, default=145)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--grow-start", type=int, default=10_000)
    parser.add_argument("--grow-every", type=int, default=2_500)
    # Grow-H requires every event to add a whole number of attention heads.
    parser.add_argument(
        "--grow-steps",
        type=int,
        default=12,
        help="number of growth events; must preserve whole-head increments",
    )
    parser.add_argument("--data-root")
    parser.add_argument("--output-root")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--show-config", action="store_true")
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    output_root = args.output_root or str(
        PROJECT_ROOT / "outputs_compare" / "ablations" / "vit_growth_axis"
    )
    overrides = {
        "adapter_module": ADAPTER,
        "num_runs": args.num_runs,
        "master_seed": args.master_seed,
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "test_batch_size": args.batch_size,
        "grow_start_iter": args.grow_start,
        "grow_every": args.grow_every,
        "grow_steps": args.grow_steps,
        "output_root": output_root,
        "device": args.device,
        "download_data": not args.no_download,
    }
    if args.data_root:
        overrides["data_root"] = str(Path(args.data_root).expanduser())
    config = make_config("vit", args.dataset, **overrides)
    if args.show_config:
        try:
            validate_model_growth_config(config)
        except ValueError as error:
            parser.error(str(error))
        payload = asdict(config)
        payload["modes"] = dict(MODE_SPECS)
        payload["manuscript_scope"] = (
            "Appendix A reports CIFAR-10, Grow-d/Grow-H, and Modes B and D"
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    try:
        run_many(config, seeds=args.seed)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
