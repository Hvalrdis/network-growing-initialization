"""Command-line interface for the Appendix B optimizer-state ablation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace

from .experiment import OptimizerAblationConfig, run_case, strategies_for
from .model import INITIALIZATION_MODE_LABELS


TARGET_WIDTHS = ((32, 64, 128), (64, 128, 256), (128, 256, 512))


def _widths(value: str) -> tuple[int, int, int]:
    try:
        result = tuple(int(part) for part in value.replace(",", "-").split("-"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected WIDTH-WIDTH-WIDTH") from error
    if len(result) != 3:
        raise argparse.ArgumentTypeError("expected exactly three widths")
    return result[0], result[1], result[2]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Appendix B: post-growth optimizer-state handling"
    )
    parser.add_argument("--optimizer", choices=["sgd", "adamw"], action="append")
    parser.add_argument(
        "--initialization-mode",
        choices=["a", "b", "d", "e"],
        action="append",
    )
    parser.add_argument(
        "--target-width",
        type=_widths,
        action="append",
        metavar="32-64-128",
    )
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--master-seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--growth-epoch", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--data-root")
    parser.add_argument("--output-root")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument(
        "--include-scheduler-restart",
        action="store_true",
        help="include Rebuild Optimizer & Restart Scheduler",
    )
    parser.add_argument("--show-plan", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    optimizers = args.optimizer or ["sgd", "adamw"]
    modes = args.initialization_mode or ["a", "b", "d", "e"]
    targets = args.target_width or list(TARGET_WIDTHS)
    cases: list[OptimizerAblationConfig] = []
    for optimizer in optimizers:
        for target in targets:
            for mode in modes:
                config = OptimizerAblationConfig(
                    optimizer=optimizer,
                    initialization_mode=mode,
                    target_widths=target,
                    num_runs=args.num_runs,
                    master_seed=args.master_seed,
                    epochs=args.epochs,
                    growth_epoch=args.growth_epoch,
                    batch_size=args.batch_size,
                    learning_rate=args.lr,
                    device=args.device,
                    download_data=not args.no_download,
                    include_scheduler_restart=args.include_scheduler_restart,
                )
                if args.data_root:
                    config = replace(config, data_root=args.data_root)
                if args.output_root:
                    config = replace(config, output_root=args.output_root)
                cases.append(config)

    if args.show_plan:
        payload = []
        for config in cases:
            item = asdict(config)
            item["strategies"] = list(strategies_for(config))
            payload.append(item)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    for case_index, config in enumerate(cases, 1):
        print(
            f"\n===== Optimizer-state ablation {case_index}/{len(cases)}: "
            f"{config.optimizer}, target={config.target_widths}, "
            f"{INITIALIZATION_MODE_LABELS[config.initialization_mode]} =====",
            flush=True,
        )
        run_case(config)


if __name__ == "__main__":
    main()
