import copy
import os
import time
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler

from mycifar10 import MyCIFAR10_2, MyCIFAR100
import imageclassificationnet_NZZ as imageclassificationnet


dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

cuda_available = torch.cuda.is_available()
print("CUDA disponible :", cuda_available)
if cuda_available:
    print("Nombre de GPUs disponibles :", torch.cuda.device_count())
    print("GPU actuel :", torch.cuda.get_device_name(torch.cuda.current_device()))


MODE_SPECS = [
    ("mode_NNN", "NNN"),
    ("mode_NZZ", "NZZ"),
    ("mode_ZNN", "ZNN"),
    ("mode_ZNZ", "ZNZ"),
]
MODE_NAMES = [name for name, _ in MODE_SPECS]


class _ReLU1Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.clamp_min(0)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        mask = (x >= 0).to(dtype=grad_output.dtype)
        return grad_output * mask


class ReLU1(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return _ReLU1Fn.apply(x)



def _replace_relu_with_relu1(seq: nn.Sequential):
    layers = []
    for layer in seq:
        if isinstance(layer, nn.ReLU):
            layers.append(ReLU1(inplace=getattr(layer, "inplace", False)))
        else:
            layers.append(layer)
    return nn.Sequential(*layers)



def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def _snapshot_optimizer_state_by_name(optimizer: optim.Optimizer, model: nn.Module):
    snap = {}
    for name, p in model.named_parameters():
        if p in optimizer.state and optimizer.state[p]:
            st = optimizer.state[p]
            snap[name] = {}
            for k, v in st.items():
                if torch.is_tensor(v):
                    snap[name][k] = v.detach().cpu().clone()
                else:
                    snap[name][k] = copy.deepcopy(v)
    return snap



def _restore_optimizer_state_by_name_with_padding(
    optimizer: optim.Optimizer,
    model: nn.Module,
    snap,
):
    for name, p_new in model.named_parameters():
        if name not in snap:
            continue

        old_state = snap[name]
        optimizer.state[p_new] = {}

        for k, v_old in old_state.items():
            if not torch.is_tensor(v_old):
                optimizer.state[p_new][k] = copy.deepcopy(v_old)
                continue

            v_old = v_old.to(device=p_new.device, dtype=p_new.dtype)

            if v_old.ndim == p_new.data.ndim:
                buf = torch.zeros_like(p_new.data)
                slices = tuple(slice(0, min(a, b)) for a, b in zip(v_old.shape, buf.shape))
                buf[slices] = v_old[slices]
                optimizer.state[p_new][k] = buf
            else:
                optimizer.state[p_new][k] = v_old



def _count_params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())

# 检查 grow 后新增卷积权重块的梯度是否为 0
GRAD_ZERO_CHECK_ENABLED = True
GRAD_ZERO_ATOL = 1e-12
GRAD_ZERO_WATCH_STEPS = 3
GRAD_ZERO_MAX_PRINT = 80


def _snapshot_conv_shapes(net: nn.Module):
    """
    记录 grow 之前每个 Conv2d 的 weight 形状。
    返回:
        {backbone_index: (out_channels, in_channels)}
    """
    shapes = {}
    if not hasattr(net, "backbone"):
        return shapes

    for layer_idx, layer in enumerate(net.backbone):
        if isinstance(layer, nn.Conv2d):
            shapes[int(layer_idx)] = (int(layer.out_channels), int(layer.in_channels))

    return shapes


def _register_new_conv_grad_watch(tag: str, net: nn.Module, old_shapes: dict, global_step: int, grow_done: int):
    """
      W_new1: old output rows, new input columns
      W_new2: new output rows, old input columns
      W_new3: new output rows, new input columns
    """
    if not GRAD_ZERO_CHECK_ENABLED:
        return

    watch_items = []

    if not hasattr(net, "backbone"):
        return

    for layer_idx, layer in enumerate(net.backbone):
        if not isinstance(layer, nn.Conv2d):
            continue

        if int(layer_idx) not in old_shapes:
            continue

        old_out, old_in = old_shapes[int(layer_idx)]
        new_out = int(layer.out_channels)
        new_in = int(layer.in_channels)

        if new_out < old_out or new_in < old_in:
            raise RuntimeError(
                f"[GradZeroWatch] 非法 shrink: tag={tag}, layer={layer_idx}, "
                f"old=(out={old_out}, in={old_in}), new=(out={new_out}, in={new_in})"
            )

        # W_new1
        if new_in > old_in and old_out > 0:
            watch_items.append({
                "layer_idx": int(layer_idx),
                "kind": "W_new1(old_out,new_in)",
                "slices": (
                    slice(0, old_out),
                    slice(old_in, new_in),
                    slice(None),
                    slice(None),
                ),
                "shape": (old_out, new_in - old_in, *layer.weight.shape[2:]),
            })

        # W_new2
        if new_out > old_out and old_in > 0:
            watch_items.append({
                "layer_idx": int(layer_idx),
                "kind": "W_new2(new_out,old_in)",
                "slices": (
                    slice(old_out, new_out),
                    slice(0, old_in),
                    slice(None),
                    slice(None),
                ),
                "shape": (new_out - old_out, old_in, *layer.weight.shape[2:]),
            })

        # W_new3
        if new_out > old_out and new_in > old_in:
            watch_items.append({
                "layer_idx": int(layer_idx),
                "kind": "W_new3(new_out,new_in)",
                "slices": (
                    slice(old_out, new_out),
                    slice(old_in, new_in),
                    slice(None),
                    slice(None),
                ),
                "shape": (new_out - old_out, new_in - old_in, *layer.weight.shape[2:]),
            })

        # bias
        if layer.bias is not None and new_out > old_out:
            watch_items.append({
                "layer_idx": int(layer_idx),
                "kind": "bias_new",
                "slices": (slice(old_out, new_out),),
                "shape": (new_out - old_out,),
            })

    net._grad_zero_watch_items = watch_items
    net._grad_zero_watch_steps_left = int(GRAD_ZERO_WATCH_STEPS)
    net._grad_zero_watch_meta = {
        "tag": str(tag),
        "grow_global_step": int(global_step),
        "grow_id": int(grow_done) + 1,
    }

    print(
        f"[GradZeroWatch][{tag}] iter {global_step}: registered "
        f"{len(watch_items)} new parameter blocks after grow {grow_done + 1}.",
        flush=True,
    )


def _grad_block_stats(g: torch.Tensor, atol: float):
    """
    返回一个梯度块的统计。
    """
    if g.numel() == 0:
        return {
            "numel": 0,
            "finite": True,
            "abs_max": 0.0,
            "norm": 0.0,
            "mean_abs": 0.0,
            "nonzero": 0,
            "all_zero": True,
        }

    finite = bool(torch.isfinite(g).all().item())
    abs_g = g.detach().abs()
    abs_max = float(abs_g.max().item())
    norm = float(torch.linalg.norm(g.detach().reshape(-1), ord=2).item())
    mean_abs = float(abs_g.mean().item())
    nonzero = int((abs_g > float(atol)).sum().item())
    all_zero = bool(abs_max <= float(atol))

    return {
        "numel": int(g.numel()),
        "finite": finite,
        "abs_max": abs_max,
        "norm": norm,
        "mean_abs": mean_abs,
        "nonzero": nonzero,
        "all_zero": all_zero,
    }


def _check_registered_new_conv_grads(tag: str, net: nn.Module, global_step: int):
    """
    在 loss.backward() 之后、optimizer.step() 之前调用。
    只检查最近一次 grow 后新增出来的权重块。
    """
    if not GRAD_ZERO_CHECK_ENABLED:
        return

    watch_items = getattr(net, "_grad_zero_watch_items", None)
    if not watch_items:
        return

    steps_left = int(getattr(net, "_grad_zero_watch_steps_left", 0))
    if steps_left <= 0:
        net._grad_zero_watch_items = []
        return

    meta = getattr(net, "_grad_zero_watch_meta", {})
    grow_step = meta.get("grow_global_step", "?")
    grow_id = meta.get("grow_id", "?")

    printed = 0
    zero_count = 0
    missing_grad_count = 0

    print(
        f"[GradZeroWatch][{tag}] train_iter={global_step}, "
        f"checking grow_id={grow_id}, grow_iter={grow_step}, "
        f"steps_left={steps_left}",
        flush=True,
    )

    for item in watch_items:
        layer_idx = int(item["layer_idx"])
        kind = str(item["kind"])

        layer = net.backbone[layer_idx]
        if kind == "bias_new":
            grad = layer.bias.grad if layer.bias is not None else None
        else:
            grad = layer.weight.grad

        if grad is None:
            missing_grad_count += 1
            status = "NO_GRAD"
            msg = (
                f"[GradZeroWatch][{tag}] {status} "
                f"layer={layer_idx} block={kind} shape={item['shape']}"
            )
            print(msg, flush=True)
            continue

        g_block = grad[item["slices"]]
        stats = _grad_block_stats(g_block, atol=GRAD_ZERO_ATOL)

        if stats["all_zero"]:
            zero_count += 1
            status = "ZERO"
        elif not stats["finite"]:
            status = "NON_FINITE"
        else:
            status = "OK"

        if printed < GRAD_ZERO_MAX_PRINT or status != "OK":
            print(
                f"[GradZeroWatch][{tag}] {status} "
                f"layer={layer_idx} block={kind} shape={item['shape']} "
                f"numel={stats['numel']} nonzero={stats['nonzero']} "
                f"abs_max={stats['abs_max']:.6e} "
                f"norm={stats['norm']:.6e} "
                f"mean_abs={stats['mean_abs']:.6e}",
                flush=True,
            )
            printed += 1

    if zero_count > 0 or missing_grad_count > 0:
        print(
            f"[GradZeroWatch][{tag}] WARNING: zero_blocks={zero_count}, "
            f"missing_grad_blocks={missing_grad_count}. ",
            flush=True,
        )

    net._grad_zero_watch_steps_left = steps_left - 1
    if net._grad_zero_watch_steps_left <= 0:
        net._grad_zero_watch_items = []

    
@torch.no_grad()
def evaluate(net, loader, loss_fn, device):
    net.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = net(x)
        loss = loss_fn(logits, y)
        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return total_loss / max(1, total), correct / max(1, total)



def build_vgg11_seed_net(
    num_classes: int,
    device: torch.device,
    with_batchnorm: bool = False,
    width_multiplier: float = 0.25,
):
    blocklist = [[1], [2], [4, 4], [8, 8], [8, 8]]
    base_width = int(64 * width_multiplier)

    backbone_cfg = []
    for block in blocklist:
        backbone_cfg.append([base_width * m for m in block])

    net = imageclassificationnet.SimpleChainImageClassificationNetwork(
        input_size=(32, 32),
        input_channels=3,
        device=device,
    )
    net.build(
        input_size=(32, 32),
        input_channels=3,
        backbone_config=backbone_cfg,
        classifier_config=[],
        with_batchnorm=with_batchnorm,
        with_maxpool=False,
        with_bias=False,
        with_relu=True,
        vgg_style=True,
        final_conv_out_channels=num_classes,
        final_conv_stride=2,
        final_conv_add_bn_relu=False,
        bn_after_relu=True,
    )

    net.gradmax_scale_method = "mean_norm"
    net.gradmax_init_scale = 0.5
    net.backbone = _replace_relu_with_relu1(net.backbone).to(device)

    return net



def grow_mode_NNN(net, config_add):
    conv_indices = [i for i, layer in enumerate(net.backbone) if isinstance(layer, torch.nn.Conv2d)]
    for i, layer_index in enumerate(conv_indices):
        nb_increase = config_add[i] if i < len(config_add) else 0
        if nb_increase > 0:
            net.expand_conv_layer(layer_index, nb_increase)
    return net



def grow_mode_NZZ(net, config_add):
    conv_indices = [i for i, layer in enumerate(net.backbone) if isinstance(layer, torch.nn.Conv2d)]
    last_increase = 0
    print(f'mode_NZZ conv_indices = {conv_indices}')
    for i, layer_index in enumerate(conv_indices):
        nb_increase = config_add[i] if i < len(config_add) else 0
        if nb_increase > 0:
            net.expand_conv_layer_mode2(layer_index, nb_increase, last_increase)
            last_increase = nb_increase
    return net



def grow_mode_ZNN(net, config_add):
    conv_indices = [i for i, layer in enumerate(net.backbone) if isinstance(layer, torch.nn.Conv2d)]
    print(f'mode_ZNN conv_indices = {conv_indices}')
    for i, layer_index in enumerate(conv_indices):
        nb_increase = config_add[i] if i < len(config_add) else 0
        if nb_increase > 0:
            net.expand_conv_layer_mode3(layer_index, nb_increase)
    return net



def grow_mode_ZNZ(net, config_add):
    conv_indices = [i for i, layer in enumerate(net.backbone) if isinstance(layer, torch.nn.Conv2d)]
    for i, layer_index in enumerate(conv_indices):
        nb_increase = config_add[i] if i < len(config_add) else 0
        if nb_increase > 0:
            net.expand_conv_layer_mode4(layer_index, nb_increase)
    return net


GROW_IMPLS = {
    "mode_NNN": grow_mode_NNN,
    "mode_NZZ": grow_mode_NZZ,
    "mode_ZNN": grow_mode_ZNN,
    "mode_ZNZ": grow_mode_ZNZ,
}



def run_compare(seed: int, run_id: int = 1):
    num_epochs = 200
    batch_size = 128
    lr = 0.05
    momentum = 0.9
    weight_decay = 5e-4
    num_workers = 0

    grow_start_iter = 10000
    grow_every = 2500
    grow_steps = 12

    data_root = "../cifar100"

    device = dev
    _set_seed(seed)

    output_dir = os.path.join(
        "outputs_compare",
        "vgg_four_modes",
        f"{time.strftime('%Y%m%d_%H%M%S')}_run{run_id:02d}_seed{seed}_lr{lr}",
    )
    os.makedirs(output_dir, exist_ok=True)

    dataset_train = MyCIFAR100(root=data_root, train=True, device=device, data_aug=True)
    dataset_test = MyCIFAR100(root=data_root, train=False, device=device, data_aug=False)

    loaderTrain = DataLoader(
        dataset_train,
        batch_size=batch_size,
        sampler=RandomSampler(dataset_train),
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
    )
    loaderTest = DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    loss_fn = nn.CrossEntropyLoss()

    steps_per_epoch = len(loaderTrain)
    total_steps = num_epochs * steps_per_epoch
    grow_triggers = set(grow_start_iter + i * grow_every for i in range(grow_steps))

    add_per_grow = [4, 8, 16, 16, 32, 32, 32, 32, 0]
    nbClasses = 100

    model_seed = build_vgg11_seed_net(nbClasses, device=device, with_batchnorm=False, width_multiplier=0.25).to(device)

    def make_opt_sch(net):
        opt = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
        return opt, sch

    optimizer_seed, scheduler_seed = make_opt_sch(model_seed)
    models = {name: None for name in MODE_NAMES}
    optimizers = {name: None for name in MODE_NAMES}
    schedulers = {name: None for name in MODE_NAMES}
    global_steps = {name: 0 for name in MODE_NAMES}
    grow_done = {name: 0 for name in MODE_NAMES}
    total_times = {name: 0.0 for name in MODE_NAMES}

    global_step_seed = 0
    total_time_seed = 0.0

    test_accuracies = {name: [] for name in MODE_NAMES}
    time_records = {name: [] for name in MODE_NAMES}
    avg_param_counts = {name: [] for name in MODE_NAMES}
    train_losses_hist = {name: [] for name in MODE_NAMES}
    train_accs_hist = {name: [] for name in MODE_NAMES}
    test_losses_hist = {name: [] for name in MODE_NAMES}

    def _train_one_step(tag, net, optimizer, scheduler, x, y, current_global_step, current_grow_done, grow_impl):
        net.train()
        optimizer.zero_grad(set_to_none=True)
        logits = net(x)
        loss = loss_fn(logits, y)
        loss.backward()

        # 新增：必须放在 loss.backward() 之后、optimizer.step() 之前
        _check_registered_new_conv_grads(
            tag=tag,
            net=net,
            global_step=current_global_step + 1,
        )

        optimizer.step()
        scheduler.step()

        bs = y.size(0)
        correct = (logits.argmax(dim=1) == y).sum().item()
        current_global_step += 1

        if (grow_impl is not None) and (current_global_step in grow_triggers) and (current_grow_done < grow_steps):
            current_lr = optimizer.param_groups[0]["lr"]
            opt_snap = _snapshot_optimizer_state_by_name(optimizer, net)
            sched_snap = scheduler.state_dict()

            old_conv_shapes = _snapshot_conv_shapes(net)

            grow_impl(net, add_per_grow)

            _register_new_conv_grad_watch(
                tag=tag,
                net=net,
                old_shapes=old_conv_shapes,
                global_step=current_global_step,
                grow_done=current_grow_done,
            )

            optimizer = optim.SGD(
                net.parameters(),
                lr=current_lr,
                momentum=momentum,
                weight_decay=weight_decay,
            )
            _restore_optimizer_state_by_name_with_padding(optimizer, net, opt_snap)

            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
            scheduler.load_state_dict(sched_snap)
            for pg, lr_now in zip(optimizer.param_groups, scheduler.get_last_lr()):
                pg["lr"] = lr_now

            optimizer.zero_grad(set_to_none=True)
            current_grow_done += 1
            print(f"[{tag}] iter {current_global_step}: grow {current_grow_done}/{grow_steps} finished", flush=True)

        return loss.item() * bs, correct, bs, optimizer, scheduler, current_global_step, current_grow_done

    def _clone_opt_sch_from_seed(new_net, seed_net, seed_opt, seed_sch):
        current_lr = seed_opt.param_groups[0]["lr"]
        opt_snap = _snapshot_optimizer_state_by_name(seed_opt, seed_net)
        sched_snap = seed_sch.state_dict()

        new_opt = optim.SGD(new_net.parameters(), lr=current_lr, momentum=momentum, weight_decay=weight_decay)
        _restore_optimizer_state_by_name_with_padding(new_opt, new_net, opt_snap)

        new_sch = optim.lr_scheduler.CosineAnnealingLR(new_opt, T_max=total_steps)
        new_sch.load_state_dict(sched_snap)
        for pg, lr_now in zip(new_opt.param_groups, new_sch.get_last_lr()):
            pg["lr"] = lr_now

        return new_opt, new_sch

    def _maybe_grow_only(tag, net, optimizer, scheduler, current_global_step, current_grow_done, grow_impl):
        if (grow_impl is None) or (current_grow_done >= grow_steps) or (current_global_step not in grow_triggers):
            return optimizer, scheduler, current_grow_done

        current_lr = optimizer.param_groups[0]["lr"]
        opt_snap = _snapshot_optimizer_state_by_name(optimizer, net)
        sched_snap = scheduler.state_dict()

        # 新增：grow 前记录旧 shape
        old_conv_shapes = _snapshot_conv_shapes(net)

        grow_impl(net, add_per_grow)

        _register_new_conv_grad_watch(
            tag=tag,
            net=net,
            old_shapes=old_conv_shapes,
            global_step=current_global_step,
            grow_done=current_grow_done,
        )

        optimizer = optim.SGD(
            net.parameters(),
            lr=current_lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        _restore_optimizer_state_by_name_with_padding(optimizer, net, opt_snap)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
        scheduler.load_state_dict(sched_snap)
        for pg, lr_now in zip(optimizer.param_groups, scheduler.get_last_lr()):
            pg["lr"] = lr_now

        optimizer.zero_grad(set_to_none=True)
        current_grow_done += 1
        print(f"[{tag}] iter {current_global_step}: grow {current_grow_done}/{grow_steps} finished (grow-only@split)", flush=True)

        return optimizer, scheduler, current_grow_done

    for epoch in range(1, num_epochs + 1):
        train_sum = {name: 0.0 for name in MODE_NAMES}
        train_correct = {name: 0 for name in MODE_NAMES}
        train_total = {name: 0 for name in MODE_NAMES}

        seed_train_sum = 0.0
        seed_train_correct = 0
        seed_train_total = 0

        for x, y in loaderTrain:
            x = x.to(device)
            y = y.to(device)

            if models[MODE_NAMES[0]] is None:
                t0 = time.time()
                loss_sum, corr, bs, optimizer_seed, scheduler_seed, global_step_seed, _ = _train_one_step(
                    tag="seed(shared)",
                    net=model_seed,
                    optimizer=optimizer_seed,
                    scheduler=scheduler_seed,
                    x=x,
                    y=y,
                    current_global_step=global_step_seed,
                    current_grow_done=0,
                    grow_impl=None,
                )
                total_time_seed += time.time() - t0
                seed_train_sum += loss_sum
                seed_train_correct += corr
                seed_train_total += bs

                if global_step_seed == grow_start_iter:
                    print(f"[INFO] Step {global_step_seed}: 触发分裂，复制出四个模式网络。", flush=True)

                    for mode_name in MODE_NAMES:
                        models[mode_name] = copy.deepcopy(model_seed)
                        optimizers[mode_name], schedulers[mode_name] = _clone_opt_sch_from_seed(
                            models[mode_name], model_seed, optimizer_seed, scheduler_seed
                        )
                        global_steps[mode_name] = global_step_seed
                        total_times[mode_name] = total_time_seed
                        train_sum[mode_name] = seed_train_sum
                        train_correct[mode_name] = seed_train_correct
                        train_total[mode_name] = seed_train_total

                    for mode_name in MODE_NAMES:
                        optimizers[mode_name], schedulers[mode_name], grow_done[mode_name] = _maybe_grow_only(
                            mode_name,
                            models[mode_name],
                            optimizers[mode_name],
                            schedulers[mode_name],
                            global_steps[mode_name],
                            grow_done[mode_name],
                            GROW_IMPLS[mode_name],
                        )

                    del model_seed, optimizer_seed, scheduler_seed
                    torch.cuda.empty_cache()
                    print("[INFO] 模型分裂完成。", flush=True)

                continue

            for mode_name in MODE_NAMES:
                t0 = time.time()
                loss_sum, corr, bs, optimizers[mode_name], schedulers[mode_name], global_steps[mode_name], grow_done[mode_name] = _train_one_step(
                    mode_name,
                    models[mode_name],
                    optimizers[mode_name],
                    schedulers[mode_name],
                    x,
                    y,
                    global_steps[mode_name],
                    grow_done[mode_name],
                    GROW_IMPLS[mode_name],
                )
                total_times[mode_name] += time.time() - t0
                train_sum[mode_name] += loss_sum
                train_correct[mode_name] += corr
                train_total[mode_name] += bs

        if models[MODE_NAMES[0]] is None:
            shared_train_loss = seed_train_sum / max(1, seed_train_total)
            shared_train_acc = seed_train_correct / max(1, seed_train_total)
            shared_test_loss, shared_test_acc = evaluate(model_seed, loaderTest, loss_fn, device)

            for mode_name in MODE_NAMES:
                train_losses_hist[mode_name].append(shared_train_loss)
                train_accs_hist[mode_name].append(shared_train_acc)
                test_losses_hist[mode_name].append(shared_test_loss)
                test_accuracies[mode_name].append(shared_test_acc)
                time_records[mode_name].append(total_time_seed)
                avg_param_counts[mode_name].append(_count_params(model_seed))

            print(
                f"[Epoch {epoch:03d}] shared-before-split: "
                + " ".join(f"{mode_name}={shared_test_acc * 100:.2f}%" for mode_name in MODE_NAMES),
                flush=True,
            )
            continue

        epoch_report_parts = [f"[Epoch {epoch:03d}]"]
        for mode_name in MODE_NAMES:
            train_loss = train_sum[mode_name] / max(1, train_total[mode_name])
            train_acc = train_correct[mode_name] / max(1, train_total[mode_name])
            test_loss, test_acc = evaluate(models[mode_name], loaderTest, loss_fn, device)

            train_losses_hist[mode_name].append(train_loss)
            train_accs_hist[mode_name].append(train_acc)
            test_losses_hist[mode_name].append(test_loss)
            test_accuracies[mode_name].append(test_acc)
            time_records[mode_name].append(total_times[mode_name])
            avg_param_counts[mode_name].append(_count_params(models[mode_name]))
            epoch_report_parts.append(f"{mode_name}={test_acc * 100:.2f}%")

        print(" ".join(epoch_report_parts), flush=True)

    epochs_arr = np.arange(1, num_epochs + 1)
    results = {"epochs": epochs_arr}
    for mode_name in MODE_NAMES:
        results[f"test_accuracies_{mode_name}"] = test_accuracies[mode_name]
        results[f"time_record_{mode_name}"] = time_records[mode_name]
        results[f"avg_param_counts_{mode_name}"] = avg_param_counts[mode_name]

    filename = os.path.join(output_dir, f"results_run_{run_id}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write("Epoch\tModel\tTrainLoss\tTrainAcc\tTestLoss\tTestAcc\tTime(s)\tParamCount\n")
        for i in range(num_epochs):
            for mode_name in MODE_NAMES:
                f.write(
                    f"{i+1}\t{mode_name}\t"
                    f"{train_losses_hist[mode_name][i]:.6f}\t{train_accs_hist[mode_name][i]:.6f}\t"
                    f"{test_losses_hist[mode_name][i]:.6f}\t{test_accuracies[mode_name][i]:.6f}\t"
                    f"{time_records[mode_name][i]:.2f}\t{avg_param_counts[mode_name][i]}\n"
                )

    plt.figure()
    for mode_name in MODE_NAMES:
        plt.plot(time_records[mode_name], test_accuracies[mode_name], label=mode_name)
    plt.xlabel("Time (s)")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy vs Time")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "test_accuracy_vs_time.png"))
    plt.close()

    plt.figure()
    for mode_name in MODE_NAMES:
        plt.plot(time_records[mode_name], test_losses_hist[mode_name], label=mode_name)
    plt.xlabel("Time (s)")
    plt.ylabel("Test Loss")
    plt.title("Test Loss vs Time")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "test_loss_vs_time.png"))
    plt.close()

    plt.figure()
    for mode_name in MODE_NAMES:
        plt.plot(epochs_arr, avg_param_counts[mode_name], label=mode_name)
    plt.xlabel("Epoch")
    plt.ylabel("Average Parameter Count")
    plt.title("Average Parameter Count vs Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "avg_param_count_vs_epochs.png"))
    plt.close()

    print(f"Experiment completed. Results saved in: {output_dir}", flush=True)
    return results


if __name__ == '__main__':
    num_runs = 3

    master_seed = 417
    rng = np.random.default_rng(master_seed)
    seeds = rng.integers(low=0, high=2**5 - 1, size=num_runs, dtype=np.int64).tolist()

    for run_id, seed in enumerate(seeds, 1):
        print(f"\n===== Run {run_id}/{num_runs}, seed={int(seed)} =====", flush=True)
        run_compare(seed=int(seed), run_id=run_id)
