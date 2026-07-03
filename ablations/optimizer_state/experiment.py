"""Training pipeline for the optimizer-state ablation in Appendix B."""

from __future__ import annotations

import copy
import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ...experiment import PROJECT_ROOT, resolve_device
from .model import (
    INITIALIZATION_MODE_LABELS,
    GrowingConvNet,
    copy_state_overlap,
    grow_model,
)


STATE_STRATEGY_LABELS = {
    "keep_state": "Keep State",
    "reset_state": "Reset State",
    "keep_moments_reset_step": "Keep Moments, Reset Step",
}
RESTART_STRATEGY = "rebuild_restart_scheduler"


@dataclass(frozen=True)
class OptimizerAblationConfig:
    optimizer: str
    initialization_mode: str
    target_widths: tuple[int, int, int]
    seed_widths: tuple[int, int, int] = (8, 16, 32)
    epochs: int = 80
    growth_epoch: int = 25
    batch_size: int = 500
    learning_rate: float = 5e-4
    momentum: float = 0.9
    weight_decay: float = 0.01
    num_runs: int = 3
    master_seed: int = 17
    data_root: str = str(PROJECT_ROOT / "data")
    output_root: str = str(
        PROJECT_ROOT / "outputs_compare" / "ablations" / "optimizer_state"
    )
    download_data: bool = True
    include_scheduler_restart: bool = False
    device: str = "auto"


def validate_config(config: OptimizerAblationConfig) -> None:
    if config.optimizer not in {"sgd", "adamw"}:
        raise ValueError("optimizer must be sgd or adamw")
    if config.initialization_mode not in INITIALIZATION_MODE_LABELS:
        raise ValueError(
            f"initialization_mode must be one of {sorted(INITIALIZATION_MODE_LABELS)}"
        )
    if len(config.target_widths) != 3 or any(
        final <= initial
        for initial, final in zip(config.seed_widths, config.target_widths)
    ):
        raise ValueError("each target width must be larger than its seed width")
    if not 1 <= config.growth_epoch <= config.epochs:
        raise ValueError("growth_epoch must lie inside the training schedule")
    if config.batch_size <= 0 or config.num_runs <= 0 or config.learning_rate <= 0:
        raise ValueError("batch_size, num_runs, and learning_rate must be positive")


def strategies_for(config: OptimizerAblationConfig) -> tuple[str, ...]:
    strategies = ["keep_state", "reset_state"]
    if config.optimizer == "adamw":
        strategies.append("keep_moments_reset_step")
    if config.include_scheduler_restart:
        strategies.append(RESTART_STRATEGY)
    return tuple(strategies)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_loaders(config: OptimizerAblationConfig, device: torch.device, seed: int):
    transform = transforms.ToTensor()
    root = Path(config.data_root).expanduser()
    train_set = datasets.CIFAR10(
        root=str(root), train=True, transform=transform, download=config.download_data
    )
    test_set = datasets.CIFAR10(
        root=str(root), train=False, transform=transform, download=config.download_data
    )
    generator = torch.Generator().manual_seed(int(seed))
    kwargs = dict(num_workers=0, pin_memory=device.type == "cuda")
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        **kwargs,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False,
        **kwargs,
    )
    return train_loader, test_loader


def _make_optimizer(
    model: nn.Module,
    config: OptimizerAblationConfig,
    lr: float | None = None,
) -> optim.Optimizer:
    learning_rate = config.learning_rate if lr is None else float(lr)
    if config.optimizer == "sgd":
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=config.momentum)
    return optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=config.weight_decay
    )


def _make_scheduler(
    optimizer: optim.Optimizer,
    config: OptimizerAblationConfig,
) -> optim.lr_scheduler.LRScheduler:
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)


def _copy_optimizer_state(
    old_optimizer: optim.Optimizer,
    new_optimizer: optim.Optimizer,
    old_model: nn.Module,
    new_model: nn.Module,
    strategy: str,
) -> None:
    if strategy in {"reset_state", RESTART_STRATEGY}:
        return
    old_parameters = dict(old_model.named_parameters())
    for name, new_parameter in new_model.named_parameters():
        old_parameter = old_parameters.get(name)
        if old_parameter is None or old_parameter not in old_optimizer.state:
            continue
        for key, old_value in old_optimizer.state[old_parameter].items():
            if not torch.is_tensor(old_value):
                new_optimizer.state[new_parameter][key] = copy.deepcopy(old_value)
            elif old_value.ndim == 0:
                value = old_value.detach().clone()
                if strategy == "keep_moments_reset_step" and key == "step":
                    value.zero_()
                new_optimizer.state[new_parameter][key] = value
            else:
                value = torch.zeros_like(new_parameter, dtype=old_value.dtype)
                copy_state_overlap(old_value.to(value.device), value)
                new_optimizer.state[new_parameter][key] = value


def _optimizer_after_growth(
    old_model: nn.Module,
    new_model: nn.Module,
    old_optimizer: optim.Optimizer,
    old_scheduler: optim.lr_scheduler.LRScheduler,
    config: OptimizerAblationConfig,
    strategy: str,
) -> tuple[optim.Optimizer, optim.lr_scheduler.LRScheduler]:
    if strategy == RESTART_STRATEGY:
        optimizer = _make_optimizer(new_model, config, config.learning_rate)
        return optimizer, _make_scheduler(optimizer, config)

    current_lrs = [float(group["lr"]) for group in old_optimizer.param_groups]
    optimizer = _make_optimizer(new_model, config, current_lrs[0])
    _copy_optimizer_state(old_optimizer, optimizer, old_model, new_model, strategy)
    scheduler = _make_scheduler(optimizer, config)
    scheduler.load_state_dict(old_scheduler.state_dict())
    for group, learning_rate in zip(optimizer.param_groups, current_lrs):
        group["lr"] = learning_rate
    return optimizer, scheduler


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    loss_sum = 0.0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(images), labels)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.item())
    return loss_sum / max(1, len(loader))


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss_sum += float(loss_fn(logits, labels).item()) * labels.size(0)
        correct += int((logits.argmax(1) == labels).sum().item())
        total += int(labels.size(0))
    return loss_sum / max(1, total), correct / max(1, total)


def _empty_history() -> dict[str, list[float | int]]:
    return {"train_loss": [], "test_loss": [], "test_acc": [], "time": [], "params": []}


def _append(
    history: dict[str, list[float | int]],
    train_loss: float,
    test_loss: float,
    test_acc: float,
    elapsed: float,
    model: nn.Module,
) -> None:
    history["train_loss"].append(float(train_loss))
    history["test_loss"].append(float(test_loss))
    history["test_acc"].append(float(test_acc))
    history["time"].append(float(elapsed))
    history["params"].append(sum(parameter.numel() for parameter in model.parameters()))


def _write_run(
    path: Path,
    histories: dict[str, dict[str, list]],
    epochs: int,
    display_names: dict[str, str],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "epoch",
                "strategy_key",
                "strategy_label",
                "train_loss",
                "test_loss",
                "test_acc",
                "time_s",
                "params",
            ]
        )
        for epoch in range(epochs):
            for name, history in histories.items():
                writer.writerow(
                    [
                        epoch + 1,
                        name,
                        display_names[name],
                        history["train_loss"][epoch],
                        history["test_loss"][epoch],
                        history["test_acc"][epoch],
                        history["time"][epoch],
                        history["params"][epoch],
                    ]
                )


def run_case(config: OptimizerAblationConfig) -> dict[str, Any]:
    """Run one optimizer, initialization-mode, and target-width setting."""
    validate_config(config)
    device = resolve_device(config.device)
    strategies = strategies_for(config)
    display_names = {
        "big_net": "Big Net",
        **STATE_STRATEGY_LABELS,
        RESTART_STRATEGY: "Rebuild Optimizer & Restart Scheduler",
    }
    width_tag = "-".join(map(str, config.target_widths))
    case_dir = (
        Path(config.output_root)
        / config.optimizer
        / f"mode_{config.initialization_mode}"
        / width_tag
    )
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "config.json").write_text(
        json.dumps(
            {
                **asdict(config),
                "initialization_mode_label": INITIALIZATION_MODE_LABELS[
                    config.initialization_mode
                ],
                "strategies": list(strategies),
                "strategy_labels": {name: display_names[name] for name in strategies},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    all_histories = []
    for run_index in range(config.num_runs):
        seed = config.master_seed + run_index
        _set_seed(seed)
        train_loader, test_loader = _build_loaders(config, device, seed)
        big_model = GrowingConvNet(config.target_widths).to(device)
        seed_model = GrowingConvNet(config.seed_widths).to(device)
        big_optimizer = _make_optimizer(big_model, config)
        seed_optimizer = _make_optimizer(seed_model, config)
        big_scheduler = _make_scheduler(big_optimizer, config)
        seed_scheduler = _make_scheduler(seed_optimizer, config)
        models: dict[str, nn.Module] = {"big_net": big_model, "seed": seed_model}
        optimizers = {"big_net": big_optimizer, "seed": seed_optimizer}
        schedulers = {"big_net": big_scheduler, "seed": seed_scheduler}
        elapsed = {"big_net": 0.0, "seed": 0.0}
        histories = {
            "big_net": _empty_history(),
            **{name: _empty_history() for name in strategies},
        }
        grown = False
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(1, config.epochs + 1):
            if epoch == config.growth_epoch:
                # Reuse one parameter initialization across state-handling strategies.
                old_branch = copy.deepcopy(models["seed"])
                grown_template = grow_model(
                    old_branch, config.target_widths, config.initialization_mode
                )
                for strategy in strategies:
                    branch = copy.deepcopy(grown_template)
                    optimizer, scheduler = _optimizer_after_growth(
                        models["seed"],
                        branch,
                        optimizers["seed"],
                        schedulers["seed"],
                        config,
                        strategy,
                    )
                    models[strategy] = branch
                    optimizers[strategy] = optimizer
                    schedulers[strategy] = scheduler
                    elapsed[strategy] = elapsed["seed"]
                grown = True

            active = ["big_net", *(strategies if grown else ["seed"])]
            epoch_values = {}
            for name in active:
                started = time.perf_counter()
                train_loss = _train_epoch(
                    models[name],
                    train_loader,
                    optimizers[name],
                    loss_fn,
                    device,
                )
                test_loss, test_acc = _evaluate(
                    models[name], test_loader, loss_fn, device
                )
                elapsed[name] += time.perf_counter() - started
                schedulers[name].step()
                epoch_values[name] = (train_loss, test_loss, test_acc)

            train_loss, test_loss, test_acc = epoch_values["big_net"]
            _append(
                histories["big_net"],
                train_loss,
                test_loss,
                test_acc,
                elapsed["big_net"],
                models["big_net"],
            )
            if grown:
                for strategy in strategies:
                    train_loss, test_loss, test_acc = epoch_values[strategy]
                    _append(
                        histories[strategy],
                        train_loss,
                        test_loss,
                        test_acc,
                        elapsed[strategy],
                        models[strategy],
                    )
            else:
                train_loss, test_loss, test_acc = epoch_values["seed"]
                for strategy in strategies:
                    _append(
                        histories[strategy],
                        train_loss,
                        test_loss,
                        test_acc,
                        elapsed["seed"],
                        models["seed"],
                    )

            values = " | ".join(
                f"{display_names[name]}={histories[name]['test_acc'][-1] * 100:.2f}%"
                for name in ("big_net", *strategies)
            )
            print(
                f"[Run {run_index + 1}/{config.num_runs}][Epoch {epoch:03d}] {values}",
                flush=True,
            )

        _write_run(
            case_dir / f"run_{run_index + 1:02d}_seed_{seed}.tsv",
            histories,
            config.epochs,
            display_names,
        )
        all_histories.append(histories)

    summary: dict[str, Any] = {}
    for name in ("big_net", *strategies):
        run_scores = [
            float(np.mean(run[name]["test_acc"][-5:])) for run in all_histories
        ]
        summary[name] = {
            "label": STATE_STRATEGY_LABELS.get(
                name,
                "Big Net"
                if name == "big_net"
                else "Rebuild Optimizer & Restart Scheduler",
            ),
            "run_last5_accuracies": run_scores,
            "mean": float(np.mean(run_scores)),
            "std": float(np.std(run_scores)),
        }
    (case_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    with (case_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["strategy", "label", "mean_last5_test_acc", "std_last5_test_acc"]
        )
        for name, values in summary.items():
            writer.writerow([name, values["label"], values["mean"], values["std"]])
    return {"config": asdict(config), "summary": summary, "output_dir": str(case_dir)}
