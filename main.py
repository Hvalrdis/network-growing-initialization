"""Command-line interface for the main model comparison."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

if __package__:
    from .experiment import (
        MODEL_DATASETS,
        available_pairs,
        make_config,
        mode_specs,
        run_many,
        selected_mode_specs,
    )
else:  # Support direct script execution.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from unified_experiments.experiment import (
        MODEL_DATASETS,
        available_pairs,
        make_config,
        mode_specs,
        run_many,
        selected_mode_specs,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run selected Big Net, Initialization Mode, and GradMax branches."
        ),
    )
    parser.add_argument("--model", choices=sorted(MODEL_DATASETS))
    parser.add_argument("--dataset", choices=["mnist", "cifar10", "cifar100"])
    parser.add_argument(
        "--list", action="store_true", help="list valid model/dataset pairs"
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="print the resolved configuration and exit",
    )
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        help="explicit run seed; may be repeated",
    )
    # Select branches with --modes big_net mode_a mode_c. Each all_modes_*
    # alias selects the growing modes and may be combined with big_net.
    parser.add_argument(
        "--modes",
        dest="selected_modes",
        nargs="+",
        metavar="MODE",
        help=(
            "run big_net, mode_a ... mode_e, and/or gradmax; each all_modes_* "
            "alias selects all growing modes and may be combined with big_net"
        ),
    )
    parser.add_argument("--num-runs", type=int)
    parser.add_argument("--master-seed", type=int)
    parser.add_argument("--data-root")
    parser.add_argument("--output-root")
    parser.add_argument("--device", help="auto, cpu, cuda, or e.g. cuda:1")
    parser.add_argument("--epochs", dest="num_epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--test-batch-size", type=int)
    parser.add_argument("--lr", dest="learning_rate", type=float)
    parser.add_argument("--momentum", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--grow-start", dest="grow_start_iter", type=int)
    parser.add_argument("--grow-every", type=int)
    parser.add_argument("--grow-steps", type=int)
    parser.add_argument("--grow-batch-size", type=int)
    parser.add_argument("--warmup-epochs", type=int)
    parser.add_argument("--label-smoothing", type=float)
    parser.add_argument("--clip-grad", type=float)
    parser.add_argument(
        "--no-download",
        dest="download_data",
        action="store_false",
        default=None,
        help="require the dataset to exist locally instead of downloading it",
    )
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="use_amp", action="store_true")
    amp_group.add_argument("--no-amp", dest="use_amp", action="store_false")
    parser.set_defaults(use_amp=None)
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.list:
        for model, dataset in available_pairs():
            print(f"{model:3s}  {dataset}")
        return
    if args.model is None or args.dataset is None:
        parser.error("--model and --dataset are required unless --list is used")
    if args.selected_modes is not None:
        args.selected_modes = tuple(args.selected_modes)

    values = vars(args)
    control_fields = {"model", "dataset", "list", "show_config", "seed"}
    overrides = {
        key: value
        for key, value in values.items()
        if key not in control_fields and value is not None
    }
    try:
        config = make_config(args.model, args.dataset, **overrides)
    except (TypeError, ValueError) as error:
        parser.error(str(error))

    if args.show_config:
        payload = asdict(config)
        payload["available_branches"] = {
            "big_net": "Big Net",
            **mode_specs(config.model),
        }
        payload["selected_branches"] = selected_mode_specs(
            config.model,
            config.selected_modes,
            adapter_module=config.adapter_module,
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    run_many(config, seeds=args.seed)


if __name__ == "__main__":
    main()
