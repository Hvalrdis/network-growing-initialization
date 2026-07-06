"""Shared training and evaluation pipeline for the main experiments."""

from __future__ import annotations

import copy
import importlib
import json
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler

from .datasets import build_datasets


MODEL_DATASETS: dict[str, tuple[str, ...]] = {
    "mlp": ("mnist",),
    "vgg": ("cifar10", "cifar100"),
    "wrn": ("cifar10", "cifar100"),
    "vit": ("cifar10", "cifar100"),
    "cvt": ("cifar10", "cifar100"),
}

MODEL_MODULES = {
    "mlp": "nngrow.models.mlp",
    "vgg": "nngrow.models.vgg",
    "wrn": "nngrow.models.wrn",
    "vit": "nngrow.models.vit",
    "cvt": "nngrow.models.cvt",
}

PROJECT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PROJECT_DIR.parent
ALL_MODES_NO_GRADMAX = "all_modes_NoGradMax"
ALL_MODES_WITH_GRADMAX = "all_modes_WithGradMax"
BIG_NET_KEY = "big_net"
BIG_NET_LABEL = "Big Net"


@dataclass(frozen=True)
class ExperimentConfig:
    model: str
    dataset: str
    data_root: str
    adapter_module: str | None = None
    selected_modes: tuple[str, ...] | None = None
    output_root: str = str(PROJECT_ROOT / "outputs_compare" / "unified")
    num_runs: int = 3
    master_seed: int = 17
    seed_strategy: str = "random"
    seed_upper_bound: int = 2**31 - 1
    seed_stride: int = 1
    num_epochs: int = 200
    batch_size: int = 128
    test_batch_size: int = 128
    learning_rate: float = 0.05
    momentum: float = 0.9
    weight_decay: float = 5e-4
    num_workers: int = 0
    grow_start_iter: int = 10000
    grow_every: int = 2_500
    grow_steps: int = 12
    grow_batch_size: int = 128
    seed_width: float = 0.25
    big_width: float = 1.0
    warmup_epochs: int = 0
    label_smoothing: float = 0.0
    clip_grad: float = 0.0
    download_data: bool = True
    use_amp: bool = False
    device: str = "auto"


def available_pairs() -> list[tuple[str, str]]:
    return [
        (model, dataset)
        for model, datasets in MODEL_DATASETS.items()
        for dataset in datasets
    ]


def make_config(model: str, dataset: str, **overrides: Any) -> ExperimentConfig:
    """Build the default configuration for a supported model and dataset."""
    model = model.lower()
    dataset = dataset.lower()
    _validate_pair(model, dataset)

    data_root = str(PROJECT_ROOT / "data")

    preset: dict[str, Any] = {}
    if model == "wrn":
        preset.update(
            test_batch_size=512, learning_rate=0.01, seed_upper_bound=2**5 - 1
        )
    elif model == "vit":
        preset.update(
            master_seed=145,
            learning_rate=1e-3,
            weight_decay=0.1,
            warmup_epochs=10,
            label_smoothing=0.1,
            clip_grad=3.0,
            use_amp=True,
            seed_strategy="sequential",
        )
    elif model == "cvt":
        preset.update(
            master_seed=0,
            num_epochs=100,
            batch_size=64,
            learning_rate=6.25e-5,
            weight_decay=0.05,
            warmup_epochs=20,
            label_smoothing=0.1,
            clip_grad=5.0,
            seed_strategy="stride",
            seed_stride=147,
        )
    preset.setdefault("master_seed", 12345 if model in {"mlp", "vgg"} else 17)

    config = ExperimentConfig(
        model=model,
        dataset=dataset,
        data_root=data_root,
        **preset,
    )
    if overrides:
        unknown = sorted(set(overrides) - set(asdict(config)))
        if unknown:
            raise TypeError(f"Unknown configuration fields: {', '.join(unknown)}")
        config = replace(config, **overrides)
    validate_config(config)
    return config


def _validate_pair(model: str, dataset: str) -> None:
    if model not in MODEL_DATASETS:
        raise ValueError(
            f"Unknown model {model!r}; choose from {sorted(MODEL_DATASETS)}"
        )
    if dataset not in MODEL_DATASETS[model]:
        supported = ", ".join(MODEL_DATASETS[model])
        raise ValueError(f"{model.upper()} supports {supported}, not {dataset}")


def validate_config(config: ExperimentConfig) -> None:
    _validate_pair(config.model, config.dataset)
    selected_mode_specs(
        config.model,
        config.selected_modes,
        adapter_module=config.adapter_module,
    )
    positive_ints = {
        "num_runs": config.num_runs,
        "num_epochs": config.num_epochs,
        "batch_size": config.batch_size,
        "test_batch_size": config.test_batch_size,
        "grow_start_iter": config.grow_start_iter,
        "grow_every": config.grow_every,
        "grow_steps": config.grow_steps,
        "grow_batch_size": config.grow_batch_size,
    }
    invalid = [name for name, value in positive_ints.items() if int(value) <= 0]
    if invalid:
        raise ValueError(f"These settings must be positive: {', '.join(invalid)}")
    if config.model == "cvt" and config.batch_size % 2:
        raise ValueError("CvT batch_size must be even for batch-level Mixup/CutMix")
    if config.num_workers != 0:
        raise ValueError(
            "num_workers must be 0 because the dataset loaders retain tensors "
            "on the selected device"
        )
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if config.seed_strategy not in {"random", "sequential", "stride"}:
        raise ValueError("seed_strategy must be random, sequential or stride")
    if config.seed_upper_bound <= 0 or config.seed_stride <= 0:
        raise ValueError("seed_upper_bound and seed_stride must be positive")
    if config.warmup_epochs < 0 or config.label_smoothing < 0 or config.clip_grad < 0:
        raise ValueError(
            "warmup_epochs, label_smoothing and clip_grad cannot be negative"
        )
    if not 0 < config.seed_width <= config.big_width:
        raise ValueError("Expected 0 < seed_width <= big_width")


def validate_growth_timing(config: ExperimentConfig, total_steps: int) -> None:
    final_growth_step = (
        config.grow_start_iter + (config.grow_steps - 1) * config.grow_every
    )
    if total_steps < final_growth_step:
        raise ValueError(
            f"The training run provides {total_steps} optimization steps, but "
            f"grow_steps={config.grow_steps} with grow_start_iter="
            f"{config.grow_start_iter} and grow_every={config.grow_every} "
            f"requires at least {final_growth_step}. Increase num_epochs, "
            "reduce grow_steps/grow_every, or start growth earlier."
        )


def resolve_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested, but CUDA is not available")
    return device


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_model_module(model: str, adapter_module: str | None = None):
    """Load a main-experiment adapter or an ablation-specific adapter."""
    return importlib.import_module(adapter_module or MODEL_MODULES[model])


def mode_specs(model: str, adapter_module: str | None = None) -> Mapping[str, str]:
    return _load_model_module(model, adapter_module).MODE_SPECS


def resolve_mode_selection(
    available_modes: Mapping[str, str],
    requested_modes: Sequence[str] | str | None,
) -> dict[str, str]:
    """Resolve explicitly selected training branches against an adapter."""
    if requested_modes is None:
        return {BIG_NET_KEY: BIG_NET_LABEL, **available_modes}

    values = (requested_modes,) if isinstance(requested_modes, str) else requested_modes
    requested = [
        token for value in values for token in str(value).replace(",", " ").split()
    ]
    if not requested:
        raise ValueError("At least one training branch must be selected")

    no_gradmax = ALL_MODES_NO_GRADMAX.lower()
    with_gradmax = ALL_MODES_WITH_GRADMAX.lower()
    aliases = {no_gradmax, with_gradmax}
    normalized = [
        BIG_NET_KEY if name.lower() == "big" else name.lower() for name in requested
    ]
    selected_aliases = [name for name in normalized if name in aliases]
    if selected_aliases:
        other_names = [
            name for name in normalized if name not in aliases and name != BIG_NET_KEY
        ]
        if len(selected_aliases) != 1 or other_names:
            raise ValueError(
                f"{ALL_MODES_NO_GRADMAX} and {ALL_MODES_WITH_GRADMAX} "
                f"may only be combined with {BIG_NET_KEY}"
            )
        selected = {BIG_NET_KEY: BIG_NET_LABEL} if BIG_NET_KEY in normalized else {}
        if selected_aliases[0] == no_gradmax:
            selected.update(
                {
                    name: label
                    for name, label in available_modes.items()
                    if name != "gradmax"
                }
            )
            return selected
        if "gradmax" not in available_modes:
            raise ValueError(
                f"{ALL_MODES_WITH_GRADMAX} is unavailable because this model "
                "does not support GradMax"
            )
        selected.update(available_modes)
        return selected

    available_by_lower = {name.lower(): name for name in available_modes}
    selected: dict[str, str] = {}
    for requested_name in normalized:
        if requested_name == BIG_NET_KEY:
            if BIG_NET_KEY in selected:
                raise ValueError(f"Branch {BIG_NET_KEY!r} was selected more than once")
            selected[BIG_NET_KEY] = BIG_NET_LABEL
            continue
        if requested_name not in available_by_lower:
            valid = ", ".join(
                [
                    BIG_NET_KEY,
                    *available_modes,
                    ALL_MODES_NO_GRADMAX,
                    ALL_MODES_WITH_GRADMAX,
                ]
            )
            raise ValueError(f"Unknown mode {requested_name!r}; choose from {valid}")
        canonical_name = available_by_lower[requested_name]
        if canonical_name in selected:
            raise ValueError(f"Mode {canonical_name!r} was selected more than once")
        selected[canonical_name] = available_modes[canonical_name]
    if not selected:
        raise ValueError("At least one branch must be selected")
    return selected


def selected_mode_specs(
    model: str,
    requested_modes: Sequence[str] | str | None,
    adapter_module: str | None = None,
) -> dict[str, str]:
    return resolve_mode_selection(
        mode_specs(model, adapter_module),
        requested_modes,
    )


def _build_and_validate_growth_plan(
    config: ExperimentConfig,
    adapter,
    device: torch.device,
    num_classes: int,
    seed: int,
    mode_order: Sequence[str],
) -> tuple[nn.Module, nn.Module | None, list[list[int]]]:
    target_model = adapter.build_model(num_classes, device, config.big_width, seed)
    if not mode_order:
        return target_model, None, []

    seed_model = adapter.build_model(num_classes, device, config.seed_width, seed)
    plan = adapter.growth_plan(seed_model, target_model, config.grow_steps)
    if len(plan) != config.grow_steps:
        raise RuntimeError("The model adapter returned an invalid growth plan")
    if any(not any(int(addition) > 0 for addition in step) for step in plan):
        raise ValueError(
            "The requested grow_steps creates an empty growth event. Choose a "
            "smaller value for this seed/target width pair."
        )
    if hasattr(adapter, "validate_growth_plan"):
        adapter.validate_growth_plan(seed_model, target_model, plan, mode_order)
    return target_model, seed_model, plan


def validate_model_growth_config(config: ExperimentConfig) -> None:
    """Validate model-specific growth-count constraints without loading data."""
    validate_config(config)
    adapter = _load_model_module(config.model, config.adapter_module)
    selected = resolve_mode_selection(adapter.MODE_SPECS, config.selected_modes)
    mode_order = [name for name in selected if name != BIG_NET_KEY]
    if not mode_order:
        return
    num_classes = 10 if config.dataset in {"mnist", "cifar10"} else 100
    _build_and_validate_growth_plan(
        config,
        adapter,
        torch.device("cpu"),
        num_classes,
        config.master_seed,
        mode_order,
    )


def _build_loaders(
    config: ExperimentConfig, device: torch.device
) -> tuple[DataLoader, DataLoader]:
    train_dataset, test_dataset = build_datasets(config, device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=RandomSampler(train_dataset),
        num_workers=config.num_workers,
        pin_memory=False,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.test_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=False,
    )
    return train_loader, test_loader


def _snapshot_optimizer(
    optimizer: optim.Optimizer, model: nn.Module
) -> dict[str, dict[str, Any]]:
    snapshot: dict[str, dict[str, Any]] = {}
    for name, parameter in model.named_parameters():
        if parameter not in optimizer.state or not optimizer.state[parameter]:
            continue
        snapshot[name] = {}
        for key, value in optimizer.state[parameter].items():
            snapshot[name][key] = (
                value.detach().cpu().clone()
                if torch.is_tensor(value)
                else copy.deepcopy(value)
            )
    return snapshot


def _restore_optimizer(
    optimizer: optim.Optimizer,
    model: nn.Module,
    snapshot: Mapping[str, Mapping[str, Any]],
) -> None:
    for name, parameter in model.named_parameters():
        if name not in snapshot:
            continue
        optimizer.state[parameter] = {}
        for key, old_value in snapshot[name].items():
            if not torch.is_tensor(old_value):
                optimizer.state[parameter][key] = copy.deepcopy(old_value)
                continue
            old_value = old_value.to(device=parameter.device, dtype=parameter.dtype)
            if old_value.ndim != parameter.ndim:
                optimizer.state[parameter][key] = old_value
                continue
            padded = torch.zeros_like(parameter)
            slices = tuple(
                slice(0, min(old, new))
                for old, new in zip(old_value.shape, padded.shape)
            )
            padded[slices] = old_value[slices]
            optimizer.state[parameter][key] = padded


def _make_optimizer_scheduler(
    model: nn.Module,
    config: ExperimentConfig,
    total_steps: int,
    learning_rate: float | None = None,
    adapter=None,
) -> tuple[optim.Optimizer, optim.lr_scheduler.LRScheduler]:
    if adapter is not None and hasattr(adapter, "make_optimizer_scheduler"):
        return adapter.make_optimizer_scheduler(
            model, config, total_steps, learning_rate
        )
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.learning_rate if learning_rate is None else learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    return optimizer, scheduler


def _clone_optimizer_scheduler(
    new_model: nn.Module,
    source_model: nn.Module,
    source_optimizer: optim.Optimizer,
    source_scheduler: optim.lr_scheduler.LRScheduler,
    config: ExperimentConfig,
    total_steps: int,
    adapter=None,
) -> tuple[optim.Optimizer, optim.lr_scheduler.LRScheduler]:
    optimizer_snapshot = _snapshot_optimizer(source_optimizer, source_model)
    scheduler_snapshot = source_scheduler.state_dict()
    optimizer, scheduler = _make_optimizer_scheduler(
        new_model,
        config,
        total_steps,
        learning_rate=source_optimizer.param_groups[0]["lr"],
        adapter=adapter,
    )
    _restore_optimizer(optimizer, new_model, optimizer_snapshot)
    scheduler.load_state_dict(scheduler_snapshot)
    for group, current_lr in zip(optimizer.param_groups, scheduler.get_last_lr()):
        group["lr"] = current_lr
    return optimizer, scheduler


def _rebuild_after_growth(
    model: nn.Module,
    config: ExperimentConfig,
    total_steps: int,
    learning_rate: float,
    optimizer_snapshot: Mapping[str, Mapping[str, Any]],
    scheduler_snapshot: Mapping[str, Any],
    adapter=None,
    old_model: nn.Module | None = None,
    old_optimizer: optim.Optimizer | None = None,
    old_scheduler: optim.lr_scheduler.LRScheduler | None = None,
) -> tuple[optim.Optimizer, optim.lr_scheduler.LRScheduler]:
    if adapter is not None and hasattr(
        adapter, "rebuild_optimizer_scheduler_after_growth"
    ):
        return adapter.rebuild_optimizer_scheduler_after_growth(
            old_model,
            model,
            old_optimizer,
            old_scheduler,
            config,
            total_steps,
        )
    rebuilt_optimizer, rebuilt_scheduler = _make_optimizer_scheduler(
        model,
        config,
        total_steps,
        learning_rate=learning_rate,
        adapter=adapter,
    )
    _restore_optimizer(rebuilt_optimizer, model, optimizer_snapshot)
    rebuilt_scheduler.load_state_dict(scheduler_snapshot)
    for group, current_lr in zip(
        rebuilt_optimizer.param_groups, rebuilt_scheduler.get_last_lr()
    ):
        group["lr"] = current_lr
    rebuilt_optimizer.zero_grad(set_to_none=True)
    return rebuilt_optimizer, rebuilt_scheduler


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    preprocess,
) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x = preprocess(x.to(device))
        y = y.to(device)
        logits = model(x)
        loss_sum += float(loss_fn(logits, y).item()) * y.size(0)
        correct += int((logits.argmax(dim=1) == y).sum().item())
        total += int(y.size(0))
    return loss_sum / max(1, total), correct / max(1, total)


def _empty_stats(names: Iterable[str]) -> dict[str, dict[str, float | int]]:
    return {name: {"loss_sum": 0.0, "correct": 0, "total": 0} for name in names}


def _add_stats(
    stats: dict[str, dict[str, float | int]],
    name: str,
    loss: float,
    correct: int,
    size: int,
) -> None:
    stats[name]["loss_sum"] += loss * size
    stats[name]["correct"] += correct
    stats[name]["total"] += size


def _stats_mean(
    stats: Mapping[str, Mapping[str, float | int]], name: str
) -> tuple[float, float]:
    total = max(1, int(stats[name]["total"]))
    return float(stats[name]["loss_sum"]) / total, int(stats[name]["correct"]) / total


def _count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _new_metrics(names: Iterable[str]) -> dict[str, dict[str, list[float | int]]]:
    return {
        name: {
            "train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": [],
            "time": [],
            "params": [],
        }
        for name in names
    }


def _append_metrics(
    metrics: dict[str, dict[str, list[float | int]]],
    name: str,
    train_loss: float,
    train_acc: float,
    test_loss: float,
    test_acc: float,
    elapsed: float,
    parameters: int,
) -> None:
    metrics[name]["train_loss"].append(train_loss)
    metrics[name]["train_acc"].append(train_acc)
    metrics[name]["test_loss"].append(test_loss)
    metrics[name]["test_acc"].append(test_acc)
    metrics[name]["time"].append(elapsed)
    metrics[name]["params"].append(parameters)


def _json_config(
    config: ExperimentConfig,
    seed: int,
    device: torch.device,
    branches: Mapping[str, str],
) -> dict[str, Any]:
    result = asdict(config)
    result.update(
        {
            "seed": seed,
            "resolved_device": str(device),
            "selected_branches": dict(branches),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
    )
    return result


def _write_results(
    output_dir: Path,
    run_id: int,
    metrics: Mapping[str, Mapping[str, Sequence[float | int]]],
    result_order: Sequence[str],
    display_names: Mapping[str, str],
) -> None:
    path = output_dir / f"results_run_{run_id}.txt"
    epochs = len(metrics[result_order[0]]["train_loss"])
    with path.open("w", encoding="utf-8") as file:
        file.write(
            "Epoch\tModelKey\tModelLabel\tTrainLoss\tTrainAcc\tTestLoss\t"
            "TestAcc\tTime(s)\tParamCount\n"
        )
        for epoch_index in range(epochs):
            for name in result_order:
                values = metrics[name]
                file.write(
                    f"{epoch_index + 1}\t{name}\t{display_names[name]}\t"
                    f"{float(values['train_loss'][epoch_index]):.6f}\t"
                    f"{float(values['train_acc'][epoch_index]):.6f}\t"
                    f"{float(values['test_loss'][epoch_index]):.6f}\t"
                    f"{float(values['test_acc'][epoch_index]):.6f}\t"
                    f"{float(values['time'][epoch_index]):.2f}\t"
                    f"{int(values['params'][epoch_index])}\n"
                )


def _write_plots(
    output_dir: Path,
    metrics: Mapping[str, Mapping[str, Sequence[float | int]]],
    result_order: Sequence[str],
    display_names: Mapping[str, str],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_specs = [
        (
            "test_accuracy_vs_time.png",
            "Time (s)",
            "Test Accuracy",
            "time",
            "test_acc",
        ),
        (
            "train_accuracy_vs_time.png",
            "Time (s)",
            "Train Accuracy",
            "time",
            "train_acc",
        ),
        ("test_loss_vs_time.png", "Time (s)", "Test Loss", "time", "test_loss"),
        ("train_loss_vs_time.png", "Time (s)", "Train Loss", "time", "train_loss"),
        (
            "test_accuracy_vs_epoch.png",
            "Epoch",
            "Test Accuracy",
            None,
            "test_acc",
        ),
        (
            "train_accuracy_vs_epoch.png",
            "Epoch",
            "Train Accuracy",
            None,
            "train_acc",
        ),
        ("test_loss_vs_epoch.png", "Epoch", "Test Loss", None, "test_loss"),
        ("train_loss_vs_epoch.png", "Epoch", "Train Loss", None, "train_loss"),
        ("parameter_count_vs_epoch.png", "Epoch", "Parameter Count", None, "params"),
    ]
    for filename, xlabel, ylabel, x_key, y_key in plot_specs:
        plt.figure()
        for name in result_order:
            y_values = metrics[name][y_key]
            x_values = (
                range(1, len(y_values) + 1) if x_key is None else metrics[name][x_key]
            )
            plt.plot(x_values, y_values, label=display_names[name])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir / filename)
        plt.close()


def run_one(
    config: ExperimentConfig,
    seed: int,
    run_id: int = 1,
) -> dict[str, dict[str, list[float | int]]]:
    """Run the selected baseline and initialization-mode branches for one seed."""
    validate_config(config)
    device = resolve_device(config.device)
    adapter = _load_model_module(config.model, config.adapter_module)
    selected_branches = resolve_mode_selection(
        adapter.MODE_SPECS, config.selected_modes
    )
    big_selected = BIG_NET_KEY in selected_branches
    modes = {
        name: label for name, label in selected_branches.items() if name != BIG_NET_KEY
    }
    mode_order = list(modes)
    result_order = ([BIG_NET_KEY] if big_selected else []) + mode_order
    display_names = dict(selected_branches)

    _set_seed(seed)
    print(
        f"Device: {device} | model={config.model} | dataset={config.dataset} | seed={seed}",
        flush=True,
    )
    num_classes = 10 if config.dataset in {"mnist", "cifar10"} else 100

    target_model, seed_model, plan = _build_and_validate_growth_plan(
        config,
        adapter,
        device,
        num_classes,
        seed,
        mode_order,
    )

    train_loader, test_loader = _build_loaders(config, device)
    steps_per_epoch = len(train_loader)
    total_steps = config.num_epochs * steps_per_epoch
    if mode_order:
        validate_growth_timing(config, total_steps)

    models: dict[str, nn.Module] = {}
    optimizers: dict[str, optim.Optimizer] = {}
    schedulers: dict[str, optim.lr_scheduler.LRScheduler] = {}
    global_steps: dict[str, int] = {}
    elapsed: dict[str, float] = {}
    if big_selected:
        models[BIG_NET_KEY] = target_model
        optimizers[BIG_NET_KEY], schedulers[BIG_NET_KEY] = _make_optimizer_scheduler(
            target_model,
            config,
            total_steps,
            adapter=adapter,
        )
        global_steps[BIG_NET_KEY] = 0
        elapsed[BIG_NET_KEY] = 0.0
    else:
        del target_model
    if seed_model is not None:
        models["seed"] = seed_model
        optimizers["seed"], schedulers["seed"] = _make_optimizer_scheduler(
            seed_model,
            config,
            total_steps,
            adapter=adapter,
        )
        global_steps["seed"] = 0
        elapsed["seed"] = 0.0

    grow_done = {name: 0 for name in mode_order}
    split_done = not mode_order
    metrics = _new_metrics(result_order)
    loss_fn = (
        adapter.build_loss(config)
        if hasattr(adapter, "build_loss")
        else nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    )
    amp_enabled = bool(config.use_amp and device.type == "cuda")

    def new_scaler() -> torch.cuda.amp.GradScaler:
        return torch.cuda.amp.GradScaler(enabled=amp_enabled)

    scalers = {name: new_scaler() for name in models}

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(config.output_root)
        / config.model
        / config.dataset
        / (f"{timestamp}_run{run_id:02d}_seed{seed}_lr{config.learning_rate}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        run_config = _json_config(config, seed, device, selected_branches)
        run_config["growth_plan"] = plan
        json.dump(run_config, file, indent=2, ensure_ascii=False)

    def train_step(
        name: str, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[float, int, int]:
        model = models[name]
        optimizer = optimizers[name]
        scheduler = schedulers[name]
        scaler = scalers[name]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            if hasattr(adapter, "compute_training_loss"):
                logits, loss = adapter.compute_training_loss(model, x, y, loss_fn)
            else:
                logits = model(x)
                loss = loss_fn(logits, y)
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss for {name} at step {global_steps[name] + 1}"
            )
        scaler.scale(loss).backward()
        if config.clip_grad > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        global_steps[name] += 1
        return (
            float(loss.item()),
            int((logits.argmax(1) == y).sum().item()),
            int(y.size(0)),
        )

    def grow(name: str, x: torch.Tensor, y: torch.Tensor) -> None:
        growth_index = grow_done[name]
        learning_rate = float(optimizers[name].param_groups[0]["lr"])
        optimizer_snapshot = _snapshot_optimizer(optimizers[name], models[name])
        scheduler_snapshot = schedulers[name].state_dict()
        old_model = models[name]
        old_optimizer = optimizers[name]
        old_scheduler = schedulers[name]
        grow_x, grow_y = x, y
        if name == "gradmax" and getattr(adapter, "GRADMAX_USES_SEPARATE_BATCH", False):
            batches_x, batches_y, collected = [], [], 0
            for candidate_x, candidate_y in train_loader:
                batches_x.append(candidate_x)
                batches_y.append(candidate_y)
                collected += int(candidate_y.size(0))
                if collected >= config.grow_batch_size:
                    break
            grow_x = adapter.preprocess(
                torch.cat(batches_x, dim=0)[: config.grow_batch_size].to(device)
            )
            grow_y = torch.cat(batches_y, dim=0)[: config.grow_batch_size].to(device)

        grown_model = adapter.grow_model(
            models[name],
            name,
            plan[growth_index],
            grow_x,
            grow_y,
            loss_fn,
            config.grow_batch_size,
        )
        if grown_model is not None:
            models[name] = grown_model
        optimizers[name], schedulers[name] = _rebuild_after_growth(
            models[name],
            config,
            total_steps,
            learning_rate,
            optimizer_snapshot,
            scheduler_snapshot,
            adapter=adapter,
            old_model=old_model,
            old_optimizer=old_optimizer,
            old_scheduler=old_scheduler,
        )
        grow_done[name] += 1
        print(
            f"[{display_names[name]}] step {global_steps[name]}: "
            f"growth event {grow_done[name]}/{config.grow_steps} complete",
            flush=True,
        )

    def split_seed_models(stats, x: torch.Tensor, y: torch.Tensor) -> None:
        nonlocal seed_model, split_done
        if seed_model is None:
            raise RuntimeError("The seed model is unavailable for branch creation")
        print(
            f"[Split] optimization step {config.grow_start_iter}: creating "
            + ", ".join(display_names[name] for name in mode_order),
            flush=True,
        )
        for name in mode_order:
            models[name] = copy.deepcopy(seed_model)
            optimizers[name], schedulers[name] = _clone_optimizer_scheduler(
                models[name],
                seed_model,
                optimizers["seed"],
                schedulers["seed"],
                config,
                total_steps,
                adapter=adapter,
            )
            global_steps[name] = global_steps["seed"]
            elapsed[name] = elapsed["seed"]
            stats[name] = copy.deepcopy(stats["seed"])
            scalers[name] = new_scaler()
            scalers[name].load_state_dict(scalers["seed"].state_dict())
            grow_started = time.perf_counter()
            grow(name, x, y)
            elapsed[name] += time.perf_counter() - grow_started

        seed_model = None
        del models["seed"], optimizers["seed"], schedulers["seed"], scalers["seed"]
        split_done = True
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    try:
        for epoch in range(1, config.num_epochs + 1):
            stats_names = [*result_order]
            if not split_done:
                stats_names.append("seed")
            stats = _empty_stats(stats_names)
            for x, y in train_loader:
                x = adapter.preprocess(x.to(device))
                y = y.to(device)

                if big_selected:
                    started = time.perf_counter()
                    loss, correct, size = train_step(BIG_NET_KEY, x, y)
                    elapsed[BIG_NET_KEY] += time.perf_counter() - started
                    _add_stats(stats, BIG_NET_KEY, loss, correct, size)

                grow_before_step = bool(getattr(adapter, "GROW_BEFORE_STEP", False))
                if (
                    mode_order
                    and not split_done
                    and grow_before_step
                    and global_steps["seed"] + 1 == config.grow_start_iter
                ):
                    split_seed_models(stats, x, y)

                if mode_order and not split_done:
                    started = time.perf_counter()
                    loss, correct, size = train_step("seed", x, y)
                    elapsed["seed"] += time.perf_counter() - started
                    _add_stats(stats, "seed", loss, correct, size)

                    if global_steps["seed"] == config.grow_start_iter:
                        split_seed_models(stats, x, y)
                    continue

                for name in mode_order:
                    started = time.perf_counter()
                    next_trigger = (
                        config.grow_start_iter + grow_done[name] * config.grow_every
                    )
                    if (
                        grow_before_step
                        and grow_done[name] < config.grow_steps
                        and global_steps[name] + 1 == next_trigger
                    ):
                        grow(name, x, y)
                    loss, correct, size = train_step(name, x, y)
                    _add_stats(stats, name, loss, correct, size)
                    if (
                        not grow_before_step
                        and global_steps[name] == next_trigger
                        and grow_done[name] < config.grow_steps
                    ):
                        grow(name, x, y)
                    elapsed[name] += time.perf_counter() - started

            report = []
            if big_selected:
                big_train_loss, big_train_acc = _stats_mean(stats, BIG_NET_KEY)
                big_test_loss, big_test_acc = _evaluate(
                    models[BIG_NET_KEY],
                    test_loader,
                    loss_fn,
                    device,
                    adapter.preprocess,
                )
                _append_metrics(
                    metrics,
                    BIG_NET_KEY,
                    big_train_loss,
                    big_train_acc,
                    big_test_loss,
                    big_test_acc,
                    elapsed[BIG_NET_KEY],
                    _count_parameters(models[BIG_NET_KEY]),
                )
                report.append(
                    f"[Epoch {epoch:03d}] {BIG_NET_LABEL}={big_test_acc * 100:.2f}%"
                )

            if mode_order and not split_done:
                if seed_model is None:
                    raise RuntimeError("The shared seed model is unavailable")
                train_loss, train_acc = _stats_mean(stats, "seed")
                test_loss, test_acc = _evaluate(
                    seed_model, test_loader, loss_fn, device, adapter.preprocess
                )
                for name in mode_order:
                    _append_metrics(
                        metrics,
                        name,
                        train_loss,
                        train_acc,
                        test_loss,
                        test_acc,
                        elapsed["seed"],
                        _count_parameters(seed_model),
                    )
                prefix = f"[Epoch {epoch:03d}] " if not report else ""
                report.append(f"{prefix}Shared Seed={test_acc * 100:.2f}%")
            elif mode_order:
                for name in mode_order:
                    train_loss, train_acc = _stats_mean(stats, name)
                    test_loss, test_acc = _evaluate(
                        models[name], test_loader, loss_fn, device, adapter.preprocess
                    )
                    _append_metrics(
                        metrics,
                        name,
                        train_loss,
                        train_acc,
                        test_loss,
                        test_acc,
                        elapsed[name],
                        _count_parameters(models[name]),
                    )
                    prefix = f"[Epoch {epoch:03d}] " if not report else ""
                    report.append(
                        f"{prefix}{display_names[name]}={test_acc * 100:.2f}%"
                    )
            print(" | ".join(report), flush=True)
            _write_results(
                output_dir,
                run_id,
                metrics,
                result_order,
                display_names,
            )

        _write_plots(output_dir, metrics, result_order, display_names)
        print(f"Experiment completed: {output_dir}", flush=True)
        return metrics
    except Exception:
        error_path = output_dir / "error.txt"
        import traceback

        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        raise


def generate_seeds(config: ExperimentConfig) -> list[int]:
    if config.seed_strategy == "sequential":
        return [config.master_seed + index for index in range(config.num_runs)]
    if config.seed_strategy == "stride":
        return [
            config.master_seed + index * config.seed_stride
            for index in range(config.num_runs)
        ]
    rng = np.random.default_rng(config.master_seed)
    return rng.integers(
        0,
        config.seed_upper_bound,
        size=config.num_runs,
        dtype=np.int64,
    ).tolist()


def run_many(
    config: ExperimentConfig,
    seeds: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    selected_seeds = list(seeds) if seeds is not None else generate_seeds(config)
    results = []
    for run_id, seed in enumerate(selected_seeds, 1):
        print(
            f"\n===== Run {run_id}/{len(selected_seeds)}, seed={int(seed)} =====",
            flush=True,
        )
        results.append(run_one(config, int(seed), run_id))
    return results
