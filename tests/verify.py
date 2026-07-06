"""Structural regression checks for the main experiments and ablations."""

from __future__ import annotations

import contextlib
import copy
import io
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from nngrow.ablations.optimizer_state import experiment as optimizer_ablation
from nngrow.ablations.optimizer_state import model as optimizer_model
from nngrow.ablations.vit_growth_axis import model as vit_axis
from nngrow.datasets import build_cvt_transforms
from nngrow.experiment import (
    ALL_MODES_NO_GRADMAX,
    ALL_MODES_WITH_GRADMAX,
    BIG_NET_KEY,
    _write_plots,
    generate_seeds,
    make_config,
    selected_mode_specs,
    validate_growth_timing,
)
from nngrow.models import cvt as model_cvt
from nngrow.models import mlp as model_mlp
from nngrow.models import vgg as model_vgg
from nngrow.models import vit as model_vit
from nngrow.models import wrn as model_wrn


EXPECTED_MODES_WITH_GRADMAX = [
    "mode_a",
    "mode_b",
    "mode_c",
    "mode_d",
    "mode_e",
    "gradmax",
]
EXPECTED_TRANSFORMER_MODES = EXPECTED_MODES_WITH_GRADMAX[:-1]
EXPECTED_MODE_LABELS = {
    "mode_a": "Mode A: Column-Zero Initialization",
    "mode_b": "Mode B: Row-First Column-Zero Initialization",
    "mode_c": "Mode C: Row-Zero Initialization",
    "mode_d": "Mode D: Homogeneous Initialization",
    "mode_e": "Mode E: Homogeneous Initialization with Empirical Variance",
    "gradmax": "GradMax",
}
EXPECTED_VIT_AXIS_LABELS = {
    "grow_d_b": "Grow-d (B)",
    "grow_d_d": "Grow-d (D)",
    "grow_h_b": "Grow-H (B)",
    "grow_h_d": "Grow-H (D)",
}
EXPECTED_STATE_STRATEGY_LABELS = {
    "keep_state": "Keep State",
    "reset_state": "Reset State",
    "keep_moments_reset_step": "Keep Moments, Reset Step",
}
EXPECTED_ADDED_BLOCK_LAYOUTS = {
    "mode_a": (False, True, True),
    "mode_b": (False, True, False),
    "mode_c": (True, False, False),
    "mode_d": (True, True, True),
    "mode_e": (True, True, True),
}


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _check_plan(module, classes: int) -> None:
    device = torch.device("cpu")
    seed = module.build_model(classes, device, 0.25, 11)
    big = module.build_model(classes, device, 1.0, 11)
    plan = module.growth_plan(seed, big, 12)
    assert len(plan) == 12
    if module is model_mlp:
        reached = [
            layer.out_features + sum(step[index] for step in plan)
            for index, layer in enumerate(seed.layers)
        ]
        target = [layer.out_features for layer in big.layers]
    elif module is model_vgg:
        seed_layers = [layer for _, layer in module._conv_layers(seed)]
        big_layers = [layer for _, layer in module._conv_layers(big)]
        reached = [
            layer.out_channels + sum(step[index] for step in plan)
            for index, layer in enumerate(seed_layers)
        ]
        target = [layer.out_channels for layer in big_layers]
    elif module is model_wrn:
        seed_tuples, big_tuples = (
            seed.get_grow_layer_tuples(),
            big.get_grow_layer_tuples(),
        )
        reached = [
            layers[0].out_channels + sum(step[index] for step in plan)
            for index, layers in enumerate(seed_tuples)
        ]
        target = [layers[0].out_channels for layers in big_tuples]
    elif module is model_vit:
        reached = seed.config["hidden_dim"] + sum(step[0] for step in plan)
        target = big.config["hidden_dim"]
    else:
        reached = tuple(
            seed.arch["dims"][index] + sum(step[index] for step in plan)
            for index in range(3)
        )
        target = big.arch["dims"]
    assert reached == target, (module.__name__, reached, target)


def _check_initialization_modes(
    module, classes: int, width: float, x: torch.Tensor, additions
) -> None:
    device = torch.device("cpu")
    labels = torch.arange(x.size(0)) % classes
    loss = nn.CrossEntropyLoss()
    base = module.build_model(classes, device, width, 7)
    expected = tuple(base(x).shape)
    with contextlib.redirect_stdout(io.StringIO()):
        for mode in module.MODE_SPECS:
            grown = module.grow_model(
                copy.deepcopy(base),
                mode,
                additions,
                x,
                labels,
                loss,
                x.size(0),
            )
            grown = base if grown is None else grown
            assert tuple(grown(x).shape) == expected


def _new_blocks(old: torch.Tensor, new: torch.Tensor, heads_out: int, heads_in: int):
    old_out, old_in = old.shape[:2]
    new_out, new_in = new.shape[:2]
    old_ohd, old_ihd = old_out // heads_out, old_in // heads_in
    view = new.view(
        heads_out,
        new_out // heads_out,
        heads_in,
        new_in // heads_in,
        *new.shape[2:],
    )
    return (
        view[:, :old_ohd, :, old_ihd:],
        view[:, old_ohd:, :, :old_ihd],
        view[:, old_ohd:, :, old_ihd:],
    )


def _check_added_block_layouts() -> None:
    device = torch.device("cpu")
    cases = (
        (
            model_vit,
            [4],
            4,
            lambda model: model.encoder.layers[0].self_attention.out_proj.weight,
        ),
        (
            model_cvt,
            [1, 3, 6],
            6,
            lambda model: model.layers[2][2].layers[0][0].fn.to_out[0].weight,
        ),
    )
    for module, additions, heads, get_weight in cases:
        base = module.build_model(10, device, 0.125, 2)
        old_weight = get_weight(base)
        for mode, expected in EXPECTED_ADDED_BLOCK_LAYOUTS.items():
            grown = module.grow_model(
                copy.deepcopy(base), mode, additions, None, None, None, 0
            )
            blocks = _new_blocks(old_weight, get_weight(grown), heads, heads)
            observed = tuple(bool(torch.count_nonzero(block)) for block in blocks)
            assert observed == expected, (module.__name__, mode, observed, expected)


def _check_cvt_batchnorm_growth() -> None:
    device = torch.device("cpu")
    base = model_cvt.build_model(10, device, 0.25, 3)
    old_bn = base.layers[0][2].layers[0][0].fn.to_q.net[1]
    with torch.no_grad():
        old_bn.weight.copy_(torch.linspace(0.5, 1.5, old_bn.num_features))
        old_bn.bias.copy_(torch.linspace(-0.2, 0.2, old_bn.num_features))
        old_bn.running_mean.copy_(torch.linspace(-1.0, 1.0, old_bn.num_features))
        old_bn.running_var.copy_(torch.linspace(0.5, 2.0, old_bn.num_features))
    for mode in model_cvt.MODE_SPECS:
        grown = model_cvt.grow_model(
            copy.deepcopy(base), mode, [1, 3, 6], None, None, None, 0
        )
        new_bn = grown.layers[0][2].layers[0][0].fn.to_q.net[1]
        assert torch.equal(
            new_bn.weight[: old_bn.num_features], old_bn.weight
        ), mode
        assert torch.equal(new_bn.bias[: old_bn.num_features], old_bn.bias), mode
        assert torch.all(new_bn.weight[old_bn.num_features :] == 1), mode
        assert torch.all(new_bn.bias[old_bn.num_features :] == 0), mode
        assert torch.equal(
            new_bn.running_mean[: old_bn.num_features], old_bn.running_mean
        ), mode
        assert torch.equal(
            new_bn.running_var[: old_bn.num_features], old_bn.running_var
        ), mode
        assert torch.all(new_bn.running_mean[old_bn.num_features :] == 0), mode
        assert torch.all(new_bn.running_var[old_bn.num_features :] == 1), mode


def _check_cvt_special_growth_parameters() -> None:
    device = torch.device("cpu")
    base = model_cvt.CvT(num_classes=10, dims=(16, 48, 96)).to(device)
    additions = [1, 3, 6]
    old_head_dim = 96 // 6

    mode_c = model_cvt.grow_model(
        copy.deepcopy(base), "mode_c", additions, None, None, None, 0
    )
    mode_c_block = mode_c.layers[2][2].layers[0]
    depthwise = mode_c_block[0].fn.to_q.net[0].weight.view(6, 17, 1, 3, 3)
    norm_gain = mode_c_block[0].norm.g.view(1, 6, 17, 1, 1)
    assert torch.count_nonzero(depthwise[:, old_head_dim:]) > 0
    assert torch.count_nonzero(norm_gain[:, :, old_head_dim:]) == 0

    mode_d = model_cvt.grow_model(
        copy.deepcopy(base), "mode_d", additions, None, None, None, 0
    )
    mode_d_bias = mode_d.layers[2][0].bias.view(6, 17)
    assert torch.count_nonzero(mode_d_bias[:, old_head_dim:]) > 0

    mode_e = model_cvt.grow_model(
        copy.deepcopy(base), "mode_e", additions, None, None, None, 0
    )
    mode_e_norm = mode_e.layers[2][2].layers[0][0].norm
    mode_e_gain = mode_e_norm.g.view(1, 6, 17, 1, 1)
    mode_e_bias = mode_e_norm.b.view(1, 6, 17, 1, 1)
    assert torch.all(mode_e_gain[:, :, old_head_dim:] == 1)
    assert torch.all(mode_e_bias[:, :, old_head_dim:] == 0)


def _check_cvt_training_protocol() -> None:
    train_transform, test_transform = build_cvt_transforms()
    assert [type(transform).__name__ for transform in train_transform.transforms] == [
        "RandomCrop",
        "RandomHorizontalFlip",
        "RandAugment",
        "ToTensor",
        "Normalize",
        "RandomErasing",
    ]
    assert [type(transform).__name__ for transform in test_transform.transforms] == [
        "ToTensor",
        "Normalize",
    ]

    criterion = model_cvt.build_loss(make_config("cvt", "cifar10"))
    assert criterion.mixup.mixup_alpha == 0.8
    assert criterion.mixup.cutmix_alpha == 1.0
    assert criterion.mixup.mix_prob == 1.0
    assert criterion.mixup.switch_prob == 0.5
    inputs = torch.randn(4, 3, 32, 32)
    labels = torch.tensor([0, 1, 2, 3])
    mixed, targets = criterion.mixup(inputs.clone(), labels)
    assert mixed.shape == inputs.shape
    assert targets.shape == (4, 10)
    assert torch.allclose(targets.sum(dim=1), torch.ones(4))

    config = make_config("cvt", "cifar10", num_epochs=5, warmup_epochs=1)
    optimizer, scheduler = model_cvt.make_optimizer_scheduler(
        nn.Linear(2, 2), config, total_steps=10
    )
    assert optimizer.param_groups[0]["lr"] == config.learning_rate * 1e-3
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == config.learning_rate * 1e-3


def _check_presets() -> None:
    mlp = make_config("mlp", "mnist")
    wrn = make_config("wrn", "cifar10")
    vit = make_config("vit", "cifar10")
    cvt = make_config("cvt", "cifar10")
    assert (mlp.learning_rate, mlp.batch_size) == (0.05, 128)
    assert (wrn.learning_rate, wrn.test_batch_size) == (0.01, 512)
    assert (vit.learning_rate, vit.warmup_epochs, vit.use_amp) == (1e-3, 10, True)
    assert (cvt.learning_rate, cvt.batch_size, cvt.warmup_epochs, cvt.num_epochs) == (
        6.25e-5,
        64,
        20,
        100,
    )
    assert generate_seeds(wrn) == [22, 26, 3]
    assert generate_seeds(vit) == [145, 146, 147]
    assert generate_seeds(cvt) == [0, 147, 294]
    _expect_value_error(lambda: make_config("cvt", "cifar10", batch_size=63))


def _check_mode_selection() -> None:
    assert list(selected_mode_specs("vgg", None)) == [
        BIG_NET_KEY,
        *EXPECTED_MODES_WITH_GRADMAX,
    ]
    assert list(selected_mode_specs("vgg", (BIG_NET_KEY,))) == [BIG_NET_KEY]
    assert list(selected_mode_specs("vgg", ("mode_b",))) == ["mode_b"]
    assert list(
        selected_mode_specs("vgg", (BIG_NET_KEY, "MODE_A", "mode_c", "mode_e"))
    ) == [
        BIG_NET_KEY,
        "mode_a",
        "mode_c",
        "mode_e",
    ]
    assert list(selected_mode_specs("vgg", (ALL_MODES_NO_GRADMAX,))) == (
        EXPECTED_TRANSFORMER_MODES
    )
    assert list(selected_mode_specs("vgg", (ALL_MODES_WITH_GRADMAX,))) == (
        EXPECTED_MODES_WITH_GRADMAX
    )
    assert list(selected_mode_specs("vgg", (BIG_NET_KEY, ALL_MODES_WITH_GRADMAX))) == [
        BIG_NET_KEY,
        *EXPECTED_MODES_WITH_GRADMAX,
    ]
    assert list(selected_mode_specs("vit", (ALL_MODES_NO_GRADMAX,))) == (
        EXPECTED_TRANSFORMER_MODES
    )

    invalid_selections = (
        ("vit", (ALL_MODES_WITH_GRADMAX,)),
        ("vgg", (ALL_MODES_NO_GRADMAX, "mode_a")),
        ("vgg", (BIG_NET_KEY, BIG_NET_KEY)),
        ("vgg", ("mode_a", "mode_a")),
        ("vgg", ("unknown",)),
    )
    for model, selection in invalid_selections:
        try:
            selected_mode_specs(model, selection)
        except ValueError:
            continue
        raise AssertionError((model, selection))


def _expect_value_error(function, *args) -> None:
    try:
        function(*args)
    except ValueError:
        return
    raise AssertionError(
        f"Expected ValueError from {function.__module__}.{function.__name__}"
    )


def _check_grow_step_constraints() -> None:
    device = torch.device("cpu")

    mlp_seed = model_mlp.build_model(10, device, 0.25, 41)
    mlp_big = model_mlp.build_model(10, device, 1.0, 41)
    assert len(model_mlp.growth_plan(mlp_seed, mlp_big, 192)) == 192
    _expect_value_error(model_mlp.growth_plan, mlp_seed, mlp_big, 193)

    vgg_seed = model_vgg.build_model(10, device, 0.25, 41)
    vgg_big = model_vgg.build_model(10, device, 1.0, 41)
    assert len(model_vgg.growth_plan(vgg_seed, vgg_big, 48)) == 48
    _expect_value_error(model_vgg.growth_plan, vgg_seed, vgg_big, 49)
    vgg_gradmax_invalid = model_vgg.growth_plan(vgg_seed, vgg_big, 4)
    _expect_value_error(
        model_vgg.validate_growth_plan,
        vgg_seed,
        vgg_big,
        vgg_gradmax_invalid,
        ("gradmax",),
    )
    vgg_gradmax_valid = model_vgg.growth_plan(vgg_seed, vgg_big, 5)
    model_vgg.validate_growth_plan(vgg_seed, vgg_big, vgg_gradmax_valid, ("gradmax",))

    wrn_seed = model_wrn.build_model(10, device, 0.25, 41)
    wrn_big = model_wrn.build_model(10, device, 1.0, 41)
    wrn_plan = model_wrn.growth_plan(wrn_seed, wrn_big, 12)
    assert len(wrn_plan) == 12
    model_wrn.validate_growth_plan(wrn_seed, wrn_big, wrn_plan, ("gradmax",))
    _expect_value_error(model_wrn.growth_plan, wrn_seed, wrn_big, 13)

    vit_seed = model_vit.build_model(10, device, 0.25, 41)
    vit_big = model_vit.build_model(10, device, 1.0, 41)
    assert len(model_vit.growth_plan(vit_seed, vit_big, 48)) == 48
    _expect_value_error(model_vit.growth_plan, vit_seed, vit_big, 49)

    cvt_seed = model_cvt.build_model(10, device, 0.25, 41)
    cvt_big = model_cvt.build_model(10, device, 1.0, 41)
    assert len(model_cvt.growth_plan(cvt_seed, cvt_big, 48)) == 48
    _expect_value_error(model_cvt.growth_plan, cvt_seed, cvt_big, 49)

    axis_plan = vit_axis.growth_plan(vit_seed, vit_big, 6)
    vit_axis.validate_growth_plan(
        vit_seed,
        vit_big,
        axis_plan,
        tuple(vit_axis.MODE_SPECS),
    )
    invalid_axis_plan = vit_axis.growth_plan(vit_seed, vit_big, 5)
    _expect_value_error(
        vit_axis.validate_growth_plan,
        vit_seed,
        vit_big,
        invalid_axis_plan,
        tuple(vit_axis.MODE_SPECS),
    )

    timing = make_config(
        "mlp",
        "mnist",
        grow_start_iter=10,
        grow_every=5,
        grow_steps=3,
    )
    validate_growth_timing(timing, 20)
    _expect_value_error(validate_growth_timing, timing, 19)


def _check_transformer_optimizer_state() -> None:
    inputs = torch.randn(2, 3, 32, 32)
    labels = torch.tensor([0, 1])
    criterion = nn.CrossEntropyLoss()
    for module, model_name, additions, heads in (
        (model_vit, "vit", [4], 4),
        (model_cvt, "cvt", [1, 3, 6], 6),
    ):
        config = make_config(
            model_name, "cifar10", num_epochs=2, warmup_epochs=0, use_amp=False
        )
        old_model = module.build_model(10, torch.device("cpu"), 0.125, 4)
        optimizer, scheduler = module.make_optimizer_scheduler(
            old_model, config, total_steps=4
        )
        loss = criterion(old_model(inputs), labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if module is model_vit:
            old_parameter = old_model.encoder.layers[0].self_attention.out_proj.weight
        else:
            old_parameter = old_model.layers[2][2].layers[0][0].fn.to_out[0].weight
        old_average = optimizer.state[old_parameter]["exp_avg"].clone()

        new_model = module.grow_model(
            old_model,
            "mode_c",
            additions,
            inputs,
            labels,
            criterion,
            inputs.size(0),
        )
        if module is model_vit:
            new_parameter = new_model.encoder.layers[0].self_attention.out_proj.weight
        else:
            new_parameter = new_model.layers[2][2].layers[0][0].fn.to_out[0].weight
        new_optimizer, _ = module.rebuild_optimizer_scheduler_after_growth(
            old_model,
            new_model,
            optimizer,
            scheduler,
            config,
            total_steps=4,
        )
        new_average = new_optimizer.state[new_parameter]["exp_avg"]
        old_out, old_in = old_average.shape[:2]
        new_out, new_in = new_average.shape[:2]
        old_view = old_average.view(heads, old_out // heads, heads, old_in // heads)
        new_view = new_average.view(heads, new_out // heads, heads, new_in // heads)
        assert torch.equal(
            new_view[:, : old_out // heads, :, : old_in // heads],
            old_view,
        )
        new_region = new_view.clone()
        new_region[:, : old_out // heads, :, : old_in // heads] = 0
        assert torch.count_nonzero(new_region) == 0


def _check_ablation_models() -> None:
    inputs = torch.randn(2, 3, 32, 32)
    labels = torch.tensor([0, 1])

    small = optimizer_model.GrowingConvNet((8, 16, 32))
    assert tuple(small(inputs).shape) == (2, 10)
    small.eval()
    reference_logits = small(inputs)
    for mode in optimizer_model.INITIALIZATION_MODE_LABELS:
        grown = optimizer_model.grow_model(copy.deepcopy(small), (32, 64, 128), mode)
        grown.eval()
        assert tuple(grown(inputs).shape) == (2, 10)
        if mode in {"a", "b"}:
            assert torch.allclose(grown(inputs), reference_logits, atol=1e-6, rtol=1e-6)
        old_weight = small.convs[1].weight
        new_weight = grown.convs[1].weight
        old_out, old_in = old_weight.shape[:2]
        assert torch.equal(old_weight, new_weight[:old_out, :old_in])
        w_new1 = new_weight[:old_out, old_in:]
        w_new2 = new_weight[old_out:, :old_in]
        w_new3 = new_weight[old_out:, old_in:]
        if mode in {"a", "b"}:
            assert torch.count_nonzero(w_new1) == 0
        if mode == "b":
            assert torch.count_nonzero(w_new2) > 0
            assert torch.count_nonzero(w_new3) == 0
        if mode in {"d", "e"}:
            assert all(
                torch.count_nonzero(block) > 0 for block in (w_new1, w_new2, w_new3)
            )

    config = optimizer_ablation.OptimizerAblationConfig(
        optimizer="adamw", initialization_mode="b", target_widths=(32, 64, 128)
    )
    optimizer = optimizer_ablation._make_optimizer(small, config)
    scheduler = optimizer_ablation._make_scheduler(optimizer, config)
    nn.CrossEntropyLoss()(small(inputs), labels).backward()
    optimizer.step()
    scheduler.step()
    grown = optimizer_model.grow_model(copy.deepcopy(small), config.target_widths, "b")
    for strategy in optimizer_ablation.strategies_for(config):
        new_optimizer, new_scheduler = optimizer_ablation._optimizer_after_growth(
            small, grown, optimizer, scheduler, config, strategy
        )
        assert new_scheduler.last_epoch == scheduler.last_epoch
        if strategy == "reset_state":
            assert not new_optimizer.state
        elif strategy == "keep_moments_reset_step":
            assert all(
                float(state["step"].item()) == 0
                for state in new_optimizer.state.values()
            )

    seed = vit_axis.build_model(10, torch.device("cpu"), 0.25, 9)
    big = vit_axis.build_model(10, torch.device("cpu"), 1.0, 9)
    plan = vit_axis.growth_plan(seed, big, 12)
    assert (
        len(plan) == 12 and seed.config["hidden_dim"] + sum(x[0] for x in plan) == 256
    )
    for mode in vit_axis.MODE_SPECS:
        grown = vit_axis.grow_model(
            seed, mode, plan[0], inputs, labels, nn.CrossEntropyLoss(), 2
        )
        assert tuple(grown(inputs).shape) == (2, 10)
        assert grown.config["hidden_dim"] == 80
        assert grown.config["num_heads"] == (5 if mode.startswith("grow_h") else 4)


def _check_plot_outputs() -> None:
    metrics = {
        "mode_c": {
            "train_loss": [1.4, 1.0, 0.7],
            "train_acc": [0.4, 0.6, 0.8],
            "test_loss": [1.5, 1.1, 0.8],
            "test_acc": [0.35, 0.55, 0.75],
            "time": [2.0, 4.5, 7.0],
            "params": [100, 120, 140],
        }
    }
    expected = {
        "test_accuracy_vs_time.png",
        "train_accuracy_vs_time.png",
        "test_loss_vs_time.png",
        "train_loss_vs_time.png",
        "test_accuracy_vs_epoch.png",
        "train_accuracy_vs_epoch.png",
        "test_loss_vs_epoch.png",
        "train_loss_vs_epoch.png",
        "parameter_count_vs_epoch.png",
    }
    with tempfile.TemporaryDirectory() as directory:
        output_dir = Path(directory)
        _write_plots(output_dir, metrics, ["mode_c"], {"mode_c": "Mode C"})
        generated = {path.name for path in output_dir.glob("*.png")}
        assert generated == expected
        assert all((output_dir / filename).stat().st_size > 0 for filename in expected)


def main() -> None:
    torch.set_num_threads(1)
    for module in (model_mlp, model_vgg, model_wrn):
        assert dict(module.MODE_SPECS) == EXPECTED_MODE_LABELS
    for module in (model_vit, model_cvt):
        assert dict(module.MODE_SPECS) == {
            key: EXPECTED_MODE_LABELS[key] for key in EXPECTED_TRANSFORMER_MODES
        }
    assert dict(vit_axis.MODE_SPECS) == EXPECTED_VIT_AXIS_LABELS
    assert optimizer_ablation.STATE_STRATEGY_LABELS == EXPECTED_STATE_STRATEGY_LABELS

    for module, classes in (
        (model_mlp, 10),
        (model_vgg, 100),
        (model_wrn, 100),
        (model_vit, 100),
        (model_cvt, 100),
    ):
        _check_plan(module, classes)

    _check_initialization_modes(model_mlp, 10, 0.02, torch.randn(2, 784), [1, 1])
    _check_initialization_modes(
        model_vgg, 10, 0.0625, torch.randn(2, 3, 32, 32), [1] * 8 + [0]
    )
    _check_initialization_modes(
        model_wrn, 10, 0.125, torch.randn(2, 3, 32, 32), [1] * 12
    )
    _check_initialization_modes(model_vit, 10, 0.125, torch.randn(2, 3, 32, 32), [4])
    _check_initialization_modes(
        model_cvt, 10, 0.125, torch.randn(2, 3, 32, 32), [1, 3, 6]
    )

    assert _parameter_count(model_cvt.CvT(num_classes=10)) == 19_555_146
    _check_added_block_layouts()
    _check_cvt_batchnorm_growth()
    _check_cvt_special_growth_parameters()
    _check_cvt_training_protocol()
    _check_transformer_optimizer_state()
    _check_ablation_models()
    _check_presets()
    _check_mode_selection()
    _check_grow_step_constraints()
    _check_plot_outputs()
    print("All structural checks passed.")


if __name__ == "__main__":
    main()
