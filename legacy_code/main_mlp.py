
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler, RandomSampler
import torch
import copy, os, time
import matplotlib.pyplot as plt
import random

from mymnist import MyMNIST
import mymlp as mymlp


dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Vérifie si CUDA est disponible
cuda_available = torch.cuda.is_available()
print("CUDA disponible :", cuda_available)
if cuda_available:
    print("Nombre de GPUs disponibles :", torch.cuda.device_count())
    print("GPU actuel :", torch.cuda.get_device_name(0))


class MySubsetSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


def _flatten_mnist(x: torch.Tensor) -> torch.Tensor:
    if x.ndim > 2:
        return x.view(x.size(0), -1)
    return x


def test_expand_layer_mode1(net, config_add):
    hidden_cnt = len(net.layers)
    increases = list(config_add)

    for layer_index in range(hidden_cnt):
        nb_increase = increases[layer_index] if layer_index < len(increases) else 0
        if nb_increase > 0:
            net.expand_layer_mode1(layer_index, int(nb_increase))
    return net


def test_expand_layer_mode2(net_mode2, config_add):
    hidden_cnt = len(net_mode2.layers)
    increases = list(config_add)
    last_increase = 0
    print(f"mode2 hidden_cnt = {hidden_cnt}")

    for layer_index in range(hidden_cnt):
        nb_increase = increases[layer_index] if layer_index < len(increases) else 0
        if nb_increase > 0:
            net_mode2.expand_layer_mode2(layer_index, int(nb_increase), int(last_increase))
            last_increase = int(nb_increase)
        else:
            last_increase = 0

    return net_mode2


def test_expand_layer_mode3(net_mode3, config_add):
    hidden_cnt = len(net_mode3.layers)
    increases = list(config_add)

    for layer_index in range(hidden_cnt):
        nb_increase = increases[layer_index] if layer_index < len(increases) else 0
        if nb_increase > 0:
            net_mode3.expand_layer_mode3(layer_index, int(nb_increase))

    return net_mode3


def test_expand_layer_mode4(net_mode4, config_add):
    hidden_cnt = len(net_mode4.layers)
    increases = list(config_add)

    for layer_index in range(hidden_cnt):
        nb_increase = increases[layer_index] if layer_index < len(increases) else 0
        if nb_increase > 0:
            net_mode4.expand_layer_mode4(layer_index, int(nb_increase))

    return net_mode4


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
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def _grow_batch(loader: DataLoader, device: torch.device, grow_batch_size: int):
    xs, ys = [], []
    total = 0
    for x, y in loader:
        xs.append(x)
        ys.append(y)
        total += x.size(0)
        if total >= grow_batch_size:
            break

    x = torch.cat(xs, dim=0)[:grow_batch_size].to(device)
    y = torch.cat(ys, dim=0)[:grow_batch_size].to(device)
    x = _flatten_mnist(x)
    return x, y


def gradmax_grow(
    net,
    add_per_layer: list,
    loader_train: DataLoader,
    loss_fn,
    grow_batch_size: int = 128,
):
    device = next(net.parameters()).device
    x, y = _grow_batch(loader_train, device=device, grow_batch_size=grow_batch_size)

    net.train()
    net.zero_grad(set_to_none=True)

    net.initForGradMax()
    logits = net.forwardForGradMax(x)
    loss = loss_fn(logits, y)
    loss.backward()

    hidden_cnt = len(net.layers)
    if hidden_cnt <= 0:
        return

    nbToGrow = [0 for _ in range(hidden_cnt)]
    for i in range(min(len(add_per_layer), hidden_cnt)):
        nbToGrow[i] = int(add_per_layer[i])

    net.growGradMax(nbToGrow=nbToGrow)


def test_expand_layer_mode5(
    net_mode5,
    config_add,
    loaderTrain,
    loss_fn,
    grow_batch_size: int = 128,
):
    gradmax_grow(
        net_mode5,
        add_per_layer=list(config_add),
        loader_train=loaderTrain,
        loss_fn=loss_fn,
        grow_batch_size=grow_batch_size,
    )
    return net_mode5


class _ReLU1Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.clamp_min(0)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        # sub-gradient at 0 is 1
        mask = (x >= 0).to(dtype=grad_output.dtype)
        return grad_output * mask


class ReLU1(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return _ReLU1Fn.apply(x)


def _replace_relu_with_relu1(module: nn.Module) -> nn.Module:
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, ReLU1(inplace=getattr(child, "inplace", False)))
        else:
            _replace_relu_with_relu1(child)
    return module


def _enforce_no_bias(net: nn.Module) -> nn.Module:
    for n, p in net.named_parameters():
        if n.endswith("bias") or n.endswith(".bias"):
            with torch.no_grad():
                p.zero_()
            p.requires_grad_(False)
    return net


def build_mlp_seed_net(
    device: torch.device,
    width_multiplier: float = 0.25,
):
    # config final：784 -> 512 -> 256 -> 10
    h1 = int(512 * width_multiplier)
    h2 = int(256 * width_multiplier)
    h1 = max(1, h1)
    h2 = max(1, h2)

    net = mymlp.MLP(
        input_size=28 * 28,
        hidden_sizes=[h1, h2],
        output_size=10,
        activation=nn.ReLU,
        init_type="kaiming",
        with_bias=False, 
    )

    net.gradmax_scale_method = "mean_norm"
    net.gradmax_init_scale = 0.5

    net = _replace_relu_with_relu1(net).to(device)
    net = _enforce_no_bias(net)
    return net


@torch.no_grad()
def evaluate(net: nn.Module, loader: DataLoader, loss_fn, device: torch.device):
    net.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = _flatten_mnist(x).to(device)
        y = y.to(device)
        logits = net(x)
        loss = loss_fn(logits, y)
        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return total_loss / max(1, total), correct / max(1, total)


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
    grow_batch = 128

    data_root = "../MNIST"
    device = dev

    _set_seed(seed)

    output_dir = f"outputs_compare/mlp_scaled/{time.strftime('%Y%m%d_%H%M%S')}_run{run_id:02d}_seed{seed}_lr{lr}"
    os.makedirs(output_dir, exist_ok=True)

    dataset_train = MyMNIST(root=data_root, train=True, device=device)
    dataset_test = MyMNIST(root=data_root, train=False, device=device)

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

    # seed: 784->128->64->10；final: 784->512->256->10
    # 总增量：+384, +192；均分到 12 次：每次 +32, +16
    add_per_grow = [32, 16]

    model_c = build_mlp_seed_net(device=device, width_multiplier=1.0).to(device)
    model_seed = build_mlp_seed_net(device=device, width_multiplier=0.25).to(device)

    model_a = model_b = model_d = model_e = model_f = None

    optimizer_c = optim.SGD(
        model_c.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    scheduler_c = optim.lr_scheduler.CosineAnnealingLR(optimizer_c, T_max=total_steps)

    optimizer_seed = optim.SGD(
        model_seed.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    scheduler_seed = optim.lr_scheduler.CosineAnnealingLR(optimizer_seed, T_max=total_steps)

    optimizer_a = scheduler_a = None
    optimizer_b = scheduler_b = None
    optimizer_d = scheduler_d = None
    optimizer_e = scheduler_e = None
    optimizer_f = scheduler_f = None

    global_step_c = 0
    global_step_seed = 0
    global_step_a = global_step_b = global_step_d = global_step_e = global_step_f = 0

    grow_done_a = grow_done_b = grow_done_d = grow_done_e = grow_done_f = 0

    test_accuracies_c, time_record_c, avg_param_counts_c = [], [], []
    test_accuracies_a, time_record_a, avg_param_counts_a = [], [], []
    test_accuracies_b, time_record_b, avg_param_counts_b = [], [], []
    test_accuracies_d, time_record_d, avg_param_counts_d = [], [], []
    test_accuracies_e, time_record_e, avg_param_counts_e = [], [], []
    test_accuracies_f, time_record_f, avg_param_counts_f = [], [], []

    train_losses_c, train_accs_c, test_losses_c = [], [], []
    train_losses_a, train_accs_a, test_losses_a = [], [], []
    train_losses_b, train_accs_b, test_losses_b = [], [], []
    train_losses_d, train_accs_d, test_losses_d = [], [], []
    train_losses_e, train_accs_e, test_losses_e = [], [], []
    train_losses_f, train_accs_f, test_losses_f = [], [], []

    total_time_c = total_time_a = total_time_b = total_time_d = total_time_e = total_time_f = 0.0
    total_time_seed = 0.0

    def _train_one_step(tag, net, optimizer, scheduler, x, y, global_step, grow_done, grow_impl):
        net.train()
        optimizer.zero_grad(set_to_none=True)
        logits = net(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        scheduler.step()

        bs = y.size(0)
        correct = (logits.argmax(dim=1) == y).sum().item()
        global_step += 1

        if (grow_impl is not None) and (global_step in grow_triggers) and (grow_done < grow_steps):
            current_lr = optimizer.param_groups[0]["lr"]
            opt_snap = _snapshot_optimizer_state_by_name(optimizer, net)
            sched_snap = scheduler.state_dict()

            grow_impl(net)

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
            grow_done += 1
            print(f"[{tag}] iter {global_step}: grow {grow_done}/{grow_steps} finished", flush=True)

        return loss.item() * bs, correct, bs, optimizer, scheduler, global_step, grow_done


    def _clone_opt_sch_from_seed(new_net, seed_net, seed_opt, seed_sch):
        current_lr = seed_opt.param_groups[0]["lr"]
        opt_snap = _snapshot_optimizer_state_by_name(seed_opt, seed_net)
        sched_snap = seed_sch.state_dict()

        new_opt = optim.SGD(
            new_net.parameters(),
            lr=current_lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        _restore_optimizer_state_by_name_with_padding(new_opt, new_net, opt_snap)

        new_sch = optim.lr_scheduler.CosineAnnealingLR(new_opt, T_max=total_steps)
        new_sch.load_state_dict(sched_snap)
        for pg, lr_now in zip(new_opt.param_groups, new_sch.get_last_lr()):
            pg["lr"] = lr_now

        return new_opt, new_sch


    def _maybe_grow_only(tag, net, optimizer, scheduler, global_step, grow_done, grow_impl):
        if (grow_impl is None) or (grow_done >= grow_steps) or (global_step not in grow_triggers):
            return optimizer, scheduler, grow_done

        current_lr = optimizer.param_groups[0]["lr"]
        opt_snap = _snapshot_optimizer_state_by_name(optimizer, net)
        sched_snap = scheduler.state_dict()

        grow_impl(net)

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
        grow_done += 1
        print(f"[{tag}] iter {global_step}: grow {grow_done}/{grow_steps} finished (grow-only@split)", flush=True)

        return optimizer, scheduler, grow_done


    def grow_impl_mode1(net):
        test_expand_layer_mode1(net, add_per_grow)
        _enforce_no_bias(net)

    def grow_impl_mode2(net):
        test_expand_layer_mode2(net, add_per_grow)
        _enforce_no_bias(net)

    def grow_impl_mode3(net):
        test_expand_layer_mode3(net, add_per_grow)
        _enforce_no_bias(net)

    def grow_impl_mode4(net):
        test_expand_layer_mode4(net, add_per_grow)
        _enforce_no_bias(net)

    def grow_impl_mode5(net):
        test_expand_layer_mode5(net, add_per_grow, loaderTrain, loss_fn, grow_batch_size=grow_batch)
        _enforce_no_bias(net)


    for epoch in range(1, num_epochs + 1):
        tr_loss_c = tr_correct_c = tr_total_c = 0
        tr_loss_seed = tr_correct_seed = tr_total_seed = 0
        tr_loss_a = tr_correct_a = tr_total_a = 0
        tr_loss_b = tr_correct_b = tr_total_b = 0
        tr_loss_d = tr_correct_d = tr_total_d = 0
        tr_loss_e = tr_correct_e = tr_total_e = 0
        tr_loss_f = tr_correct_f = tr_total_f = 0

        for x, y in loaderTrain:
            x = _flatten_mnist(x).to(device)
            y = y.to(device)

            # big model c
            t0 = time.time()
            loss_sum, corr, bs, optimizer_c, scheduler_c, global_step_c, _ = _train_one_step(
                tag="c",
                net=model_c,
                optimizer=optimizer_c,
                scheduler=scheduler_c,
                x=x, y=y,
                global_step=global_step_c,
                grow_done=0,
                grow_impl=None,
            )
            total_time_c += (time.time() - t0)
            tr_loss_c += loss_sum
            tr_correct_c += corr
            tr_total_c += bs

            # seed/shared before split
            if model_a is None:
                t0 = time.time()
                loss_sum, corr, bs, optimizer_seed, scheduler_seed, global_step_seed, _ = _train_one_step(
                    tag="seed(shared)",
                    net=model_seed,
                    optimizer=optimizer_seed,
                    scheduler=scheduler_seed,
                    x=x, y=y,
                    global_step=global_step_seed,
                    grow_done=0,
                    grow_impl=None,
                )
                total_time_seed += (time.time() - t0)
                tr_loss_seed += loss_sum
                tr_correct_seed += corr
                tr_total_seed += bs

                # split at grow_start_iter
                if global_step_seed == grow_start_iter:
                    print(f"[INFO] Step {global_step_seed}: 触发分裂，复制出 mode1~5 网络。", flush=True)

                    model_a = copy.deepcopy(model_seed)
                    model_b = copy.deepcopy(model_seed)
                    model_d = copy.deepcopy(model_seed)
                    model_e = copy.deepcopy(model_seed)
                    model_f = copy.deepcopy(model_seed)

                    optimizer_a, scheduler_a = _clone_opt_sch_from_seed(model_a, model_seed, optimizer_seed, scheduler_seed)
                    optimizer_b, scheduler_b = _clone_opt_sch_from_seed(model_b, model_seed, optimizer_seed, scheduler_seed)
                    optimizer_d, scheduler_d = _clone_opt_sch_from_seed(model_d, model_seed, optimizer_seed, scheduler_seed)
                    optimizer_e, scheduler_e = _clone_opt_sch_from_seed(model_e, model_seed, optimizer_seed, scheduler_seed)
                    optimizer_f, scheduler_f = _clone_opt_sch_from_seed(model_f, model_seed, optimizer_seed, scheduler_seed)

                    global_step_a = global_step_b = global_step_d = global_step_e = global_step_f = global_step_seed
                    total_time_a = total_time_b = total_time_d = total_time_e = total_time_f = total_time_seed

                    tr_loss_a = tr_loss_b = tr_loss_d = tr_loss_e = tr_loss_f = tr_loss_seed
                    tr_correct_a = tr_correct_b = tr_correct_d = tr_correct_e = tr_correct_f = tr_correct_seed
                    tr_total_a = tr_total_b = tr_total_d = tr_total_e = tr_total_f = tr_total_seed

                    optimizer_a, scheduler_a, grow_done_a = _maybe_grow_only(
                        "a/mode1", model_a, optimizer_a, scheduler_a, global_step_a, grow_done_a, grow_impl_mode1
                    )
                    optimizer_b, scheduler_b, grow_done_b = _maybe_grow_only(
                        "b/mode2", model_b, optimizer_b, scheduler_b, global_step_b, grow_done_b, grow_impl_mode2
                    )
                    optimizer_d, scheduler_d, grow_done_d = _maybe_grow_only(
                        "d/mode3", model_d, optimizer_d, scheduler_d, global_step_d, grow_done_d, grow_impl_mode3
                    )
                    optimizer_e, scheduler_e, grow_done_e = _maybe_grow_only(
                        "e/mode4", model_e, optimizer_e, scheduler_e, global_step_e, grow_done_e, grow_impl_mode4
                    )
                    optimizer_f, scheduler_f, grow_done_f = _maybe_grow_only(
                        "f/GradMax", model_f, optimizer_f, scheduler_f, global_step_f, grow_done_f, grow_impl_mode5
                    )

                    del model_seed, optimizer_seed, scheduler_seed
                    torch.cuda.empty_cache()
                    print("[INFO] 模型分裂完成。", flush=True)

                continue

            # after split: train each model on same batch
            t0 = time.time()
            loss_sum, corr, bs, optimizer_a, scheduler_a, global_step_a, grow_done_a = _train_one_step(
                "a/mode1", model_a, optimizer_a, scheduler_a, x, y, global_step_a, grow_done_a, grow_impl_mode1
            )
            total_time_a += (time.time() - t0)
            tr_loss_a += loss_sum; tr_correct_a += corr; tr_total_a += bs

            t0 = time.time()
            loss_sum, corr, bs, optimizer_b, scheduler_b, global_step_b, grow_done_b = _train_one_step(
                "b/mode2", model_b, optimizer_b, scheduler_b, x, y, global_step_b, grow_done_b, grow_impl_mode2
            )
            total_time_b += (time.time() - t0)
            tr_loss_b += loss_sum; tr_correct_b += corr; tr_total_b += bs

            t0 = time.time()
            loss_sum, corr, bs, optimizer_d, scheduler_d, global_step_d, grow_done_d = _train_one_step(
                "d/mode3", model_d, optimizer_d, scheduler_d, x, y, global_step_d, grow_done_d, grow_impl_mode3
            )
            total_time_d += (time.time() - t0)
            tr_loss_d += loss_sum; tr_correct_d += corr; tr_total_d += bs

            t0 = time.time()
            loss_sum, corr, bs, optimizer_e, scheduler_e, global_step_e, grow_done_e = _train_one_step(
                "e/mode4", model_e, optimizer_e, scheduler_e, x, y, global_step_e, grow_done_e, grow_impl_mode4
            )
            total_time_e += (time.time() - t0)
            tr_loss_e += loss_sum; tr_correct_e += corr; tr_total_e += bs

            t0 = time.time()
            loss_sum, corr, bs, optimizer_f, scheduler_f, global_step_f, grow_done_f = _train_one_step(
                "f/GradMax", model_f, optimizer_f, scheduler_f, x, y, global_step_f, grow_done_f, grow_impl_mode5
            )
            total_time_f += (time.time() - t0)
            tr_loss_f += loss_sum; tr_correct_f += corr; tr_total_f += bs

        # epoch end: compute train stats
        train_loss_c = tr_loss_c / max(1, tr_total_c)
        train_acc_c  = tr_correct_c / max(1, tr_total_c)

        train_loss_a = tr_loss_a / max(1, tr_total_a)
        train_acc_a  = tr_correct_a / max(1, tr_total_a)

        train_loss_b = tr_loss_b / max(1, tr_total_b)
        train_acc_b  = tr_correct_b / max(1, tr_total_b)

        train_loss_d = tr_loss_d / max(1, tr_total_d)
        train_acc_d  = tr_correct_d / max(1, tr_total_d)

        train_loss_e = tr_loss_e / max(1, tr_total_e)
        train_acc_e  = tr_correct_e / max(1, tr_total_e)

        train_loss_f = tr_loss_f / max(1, tr_total_f)
        train_acc_f  = tr_correct_f / max(1, tr_total_f)

        test_loss_c, test_acc_c = evaluate(model_c, loaderTest, loss_fn, device)
        test_losses_c.append(test_loss_c)
        test_accuracies_c.append(test_acc_c)
        train_losses_c.append(train_loss_c)
        train_accs_c.append(train_acc_c)
        time_record_c.append(total_time_c)
        avg_param_counts_c.append(_count_params(model_c))

        if model_a is None:
            test_loss_seed, test_acc_seed = evaluate(model_seed, loaderTest, loss_fn, device)

            train_loss_seed = tr_loss_seed / max(1, tr_total_seed)
            train_acc_seed  = tr_correct_seed / max(1, tr_total_seed)

            for L in [test_losses_a, test_losses_b, test_losses_d, test_losses_e, test_losses_f]:
                L.append(test_loss_seed)
            for A in [test_accuracies_a, test_accuracies_b, test_accuracies_d, test_accuracies_e, test_accuracies_f]:
                A.append(test_acc_seed)

            for L in [train_losses_a, train_losses_b, train_losses_d, train_losses_e, train_losses_f]:
                L.append(train_loss_seed)
            for A in [train_accs_a, train_accs_b, train_accs_d, train_accs_e, train_accs_f]:
                A.append(train_acc_seed)

            seed_params = _count_params(model_seed)
            for t_list in [time_record_a, time_record_b, time_record_d, time_record_e, time_record_f]:
                t_list.append(total_time_seed)
            for p_list in [avg_param_counts_a, avg_param_counts_b, avg_param_counts_d, avg_param_counts_e, avg_param_counts_f]:
                p_list.append(seed_params)

            print(
                f"[Epoch {epoch:03d}] "
                f"c(big)={test_acc_c*100:.2f}% "
                f"seed(shared)={test_acc_seed*100:.2f}% "
                f"(a/b/d/e/f 统计=seed, split@{grow_start_iter})",
                flush=True
            )
        else:
            test_loss_a, test_acc_a = evaluate(model_a, loaderTest, loss_fn, device)
            test_loss_b, test_acc_b = evaluate(model_b, loaderTest, loss_fn, device)
            test_loss_d, test_acc_d = evaluate(model_d, loaderTest, loss_fn, device)
            test_loss_e, test_acc_e = evaluate(model_e, loaderTest, loss_fn, device)
            test_loss_f, test_acc_f = evaluate(model_f, loaderTest, loss_fn, device)

            test_losses_a.append(test_loss_a)
            test_accuracies_a.append(test_acc_a)
            train_losses_a.append(train_loss_a)
            train_accs_a.append(train_acc_a)
            time_record_a.append(total_time_a)
            avg_param_counts_a.append(_count_params(model_a))

            test_losses_b.append(test_loss_b)
            test_accuracies_b.append(test_acc_b)
            train_losses_b.append(train_loss_b)
            train_accs_b.append(train_acc_b)
            time_record_b.append(total_time_b)
            avg_param_counts_b.append(_count_params(model_b))

            test_losses_d.append(test_loss_d)
            test_accuracies_d.append(test_acc_d)
            train_losses_d.append(train_loss_d)
            train_accs_d.append(train_acc_d)
            time_record_d.append(total_time_d)
            avg_param_counts_d.append(_count_params(model_d))

            test_losses_e.append(test_loss_e)
            test_accuracies_e.append(test_acc_e)
            train_losses_e.append(train_loss_e)
            train_accs_e.append(train_acc_e)
            time_record_e.append(total_time_e)
            avg_param_counts_e.append(_count_params(model_e))

            test_losses_f.append(test_loss_f)
            test_accuracies_f.append(test_acc_f)
            train_losses_f.append(train_loss_f)
            train_accs_f.append(train_acc_f)
            time_record_f.append(total_time_f)
            avg_param_counts_f.append(_count_params(model_f))

            print(
                f"[Epoch {epoch:03d}] "
                f"c(big)={test_acc_c*100:.2f}% "
                f"a(m1)={test_acc_a*100:.2f}% "
                f"b(m2)={test_acc_b*100:.2f}% "
                f"d(m3)={test_acc_d*100:.2f}% "
                f"e(m4)={test_acc_e*100:.2f}% "
                f"f(GradMax)={test_acc_f*100:.2f}%",
                flush=True
            )

    results = {
        "test_accuracies_c": test_accuracies_c,
        "time_record_c": time_record_c,
        "avg_param_counts_c": avg_param_counts_c,

        "test_accuracies_a": test_accuracies_a,
        "time_record_a": time_record_a,
        "avg_param_counts_a": avg_param_counts_a,

        "test_accuracies_b": test_accuracies_b,
        "time_record_b": time_record_b,
        "avg_param_counts_b": avg_param_counts_b,

        "test_accuracies_d": test_accuracies_d,
        "time_record_d": time_record_d,
        "avg_param_counts_d": avg_param_counts_d,

        "test_accuracies_e": test_accuracies_e,
        "time_record_e": time_record_e,
        "avg_param_counts_e": avg_param_counts_e,

        "test_accuracies_f": test_accuracies_f,
        "time_record_f": time_record_f,
        "avg_param_counts_f": avg_param_counts_f,
    }

    filename = os.path.join(output_dir, f"results_run_{run_id}.txt")
    with open(filename, "w") as f:
        f.write("Epoch\tModel\tTrainLoss\tTrainAcc\tTestLoss\tTestAcc\tTime(s)\tParamCount\n")
        for i in range(num_epochs):
            f.write(
                f"{i+1}\tc\t{train_losses_c[i]:.6f}\t{train_accs_c[i]:.6f}\t"
                f"{test_losses_c[i]:.6f}\t{test_accuracies_c[i]:.6f}\t{time_record_c[i]:.2f}\t{avg_param_counts_c[i]}\n"
            )

            f.write(
                f"{i+1}\ta\t{train_losses_a[i]:.6f}\t{train_accs_a[i]:.6f}\t"
                f"{test_losses_a[i]:.6f}\t{test_accuracies_a[i]:.6f}\t{time_record_a[i]:.2f}\t{avg_param_counts_a[i]}\n"
            )
            f.write(
                f"{i+1}\tb\t{train_losses_b[i]:.6f}\t{train_accs_b[i]:.6f}\t"
                f"{test_losses_b[i]:.6f}\t{test_accuracies_b[i]:.6f}\t{time_record_b[i]:.2f}\t{avg_param_counts_b[i]}\n"
            )
            f.write(
                f"{i+1}\td\t{train_losses_d[i]:.6f}\t{train_accs_d[i]:.6f}\t"
                f"{test_losses_d[i]:.6f}\t{test_accuracies_d[i]:.6f}\t{time_record_d[i]:.2f}\t{avg_param_counts_d[i]}\n"
            )
            f.write(
                f"{i+1}\te\t{train_losses_e[i]:.6f}\t{train_accs_e[i]:.6f}\t"
                f"{test_losses_e[i]:.6f}\t{test_accuracies_e[i]:.6f}\t{time_record_e[i]:.2f}\t{avg_param_counts_e[i]}\n"
            )
            f.write(
                f"{i+1}\tf\t{train_losses_f[i]:.6f}\t{train_accs_f[i]:.6f}\t"
                f"{test_losses_f[i]:.6f}\t{test_accuracies_f[i]:.6f}\t{time_record_f[i]:.2f}\t{avg_param_counts_f[i]}\n"
            )

    plt.figure()
    plt.plot(time_record_c, test_accuracies_c, label="Big Net", color="blue")
    plt.plot(time_record_a, test_accuracies_a, label="Model a", color="orange")
    plt.plot(time_record_b, test_accuracies_b, label="Model b", color="brown")
    plt.plot(time_record_d, test_accuracies_d, label="Model d", color="green")
    plt.plot(time_record_e, test_accuracies_e, label="Model e", color="deeppink")
    plt.plot(time_record_f, test_accuracies_f, label="GradMax", color="purple")
    plt.xlabel("Time (s)")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy vs Time")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "test_accuracy_vs_time.png"))
    plt.close()

    plt.figure()
    plt.plot(time_record_c, test_losses_c, label="Big Net", color="blue")
    plt.plot(time_record_a, test_losses_a, label="Model a", color="orange")
    plt.plot(time_record_b, test_losses_b, label="Model b", color="brown")
    plt.plot(time_record_d, test_losses_d, label="Model d", color="green")
    plt.plot(time_record_e, test_losses_e, label="Model e", color="deeppink")
    plt.plot(time_record_f, test_losses_f, label="GradMax", color="purple")
    plt.xlabel("Time (s)")
    plt.ylabel("Test Loss")
    plt.title("Test Loss vs Time")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "test_loss_vs_time.png"))
    plt.close()

    epochs_arr = np.arange(1, num_epochs + 1)
    plt.figure()
    plt.plot(epochs_arr, avg_param_counts_c, label="Big Net", color="blue")
    plt.plot(epochs_arr, avg_param_counts_a, label="Model a", color="orange")
    plt.plot(epochs_arr, avg_param_counts_b, label="Model b", color="brown")
    plt.plot(epochs_arr, avg_param_counts_d, label="Model d", color="green")
    plt.plot(epochs_arr, avg_param_counts_e, label="Model e", color="deeppink")
    plt.plot(epochs_arr, avg_param_counts_f, label="GradMax", color="purple")
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

    master_seed = 12345
    rng = np.random.default_rng(master_seed)
    seeds = rng.integers(low=0, high=2**31 - 1, size=num_runs, dtype=np.int64).tolist()

    for run_id, seed in enumerate(seeds, 1):
        print(f"\n===== Run {run_id}/{num_runs}, seed={int(seed)} =====", flush=True)
        run_compare(seed=int(seed), run_id=run_id)