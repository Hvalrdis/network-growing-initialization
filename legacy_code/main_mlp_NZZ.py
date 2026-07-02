import copy
import os
import time
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler, RandomSampler

from mymnist import MyMNIST
import mymlp_NZZ as mymlp


dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
    # final config: 784 -> 512 -> 256 -> 10
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


def test_expand_layer_nzz(net, config_add):
    hidden_cnt = len(net.layers)
    increases = list(config_add)
    last_increase = 0
    print(f"mode_NZZ hidden_cnt = {hidden_cnt}")

    for layer_index in range(hidden_cnt):
        nb_increase = int(increases[layer_index]) if layer_index < len(increases) else 0
        if nb_increase > 0:
            net.expand_layer_nzz(layer_index, nb_increase, int(last_increase))
            last_increase = nb_increase
        else:
            last_increase = 0

    return net


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

    data_root = "../MNIST"
    device = dev

    _set_seed(seed)

    output_dir = f"outputs_compare/mlp_NZZ/{time.strftime('%Y%m%d_%H%M%S')}_run{run_id:02d}_seed{seed}_lr{lr}"
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

    # seed: 784 -> 128 -> 64 -> 10; final: 784 -> 512 -> 256 -> 10
    # total increments: +384, +192; split into 12 grows: +32, +16 each time.
    add_per_grow = [32, 16]

    model_nzz = build_mlp_seed_net(device=device, width_multiplier=0.25).to(device)

    def make_opt_sch(net):
        opt = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
        return opt, sch

    optimizer_nzz, scheduler_nzz = make_opt_sch(model_nzz)

    global_step_nzz = 0
    grow_done_nzz = 0
    total_time_nzz = 0.0

    train_losses_nzz, train_accs_nzz, test_losses_nzz = [], [], []
    test_accuracies_nzz, time_record_nzz, avg_param_counts_nzz = [], [], []

    def _grow_nzz(net):
        test_expand_layer_nzz(net, add_per_grow)
        _enforce_no_bias(net)

    def _train_one_step(tag, net, optimizer, scheduler, x, y, global_step, grow_done):
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

        if (global_step in grow_triggers) and (grow_done < grow_steps):
            current_lr = optimizer.param_groups[0]["lr"]
            opt_snap = _snapshot_optimizer_state_by_name(optimizer, net)
            sched_snap = scheduler.state_dict()

            _grow_nzz(net)

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

    for epoch in range(1, num_epochs + 1):
        tr_loss_nzz = 0.0
        tr_correct_nzz = 0
        tr_total_nzz = 0

        for x, y in loaderTrain:
            x = _flatten_mnist(x).to(device)
            y = y.to(device)

            t0 = time.time()
            loss_sum, corr, bs, optimizer_nzz, scheduler_nzz, global_step_nzz, grow_done_nzz = _train_one_step(
                tag="NZZ",
                net=model_nzz,
                optimizer=optimizer_nzz,
                scheduler=scheduler_nzz,
                x=x,
                y=y,
                global_step=global_step_nzz,
                grow_done=grow_done_nzz,
            )
            total_time_nzz += time.time() - t0
            tr_loss_nzz += loss_sum
            tr_correct_nzz += corr
            tr_total_nzz += bs

        train_loss_nzz = tr_loss_nzz / max(1, tr_total_nzz)
        train_acc_nzz = tr_correct_nzz / max(1, tr_total_nzz)
        test_loss_nzz, test_acc_nzz = evaluate(model_nzz, loaderTest, loss_fn, device)

        train_losses_nzz.append(train_loss_nzz)
        train_accs_nzz.append(train_acc_nzz)
        test_losses_nzz.append(test_loss_nzz)
        test_accuracies_nzz.append(test_acc_nzz)
        time_record_nzz.append(total_time_nzz)
        avg_param_counts_nzz.append(_count_params(model_nzz))

        print(
            f"[Epoch {epoch:03d}] NZZ={test_acc_nzz * 100:.2f}% "
            f"grow_done={grow_done_nzz}/{grow_steps} params={avg_param_counts_nzz[-1]}",
            flush=True,
        )

    filename = os.path.join(output_dir, f"results_run_{run_id}_NZZ.txt")
    with open(filename, "w") as f:
        f.write("Epoch\tModel\tTrainLoss\tTrainAcc\tTestLoss\tTestAcc\tTime(s)\tParamCount\n")
        for i in range(num_epochs):
            f.write(
                f"{i+1}\tNZZ\t{train_losses_nzz[i]:.6f}\t{train_accs_nzz[i]:.6f}\t"
                f"{test_losses_nzz[i]:.6f}\t{test_accuracies_nzz[i]:.6f}\t"
                f"{time_record_nzz[i]:.2f}\t{avg_param_counts_nzz[i]}\n"
            )

    plt.figure()
    plt.plot(time_record_nzz, test_accuracies_nzz, label="NZZ")
    plt.xlabel("Time (s)")
    plt.ylabel("Test Accuracy")
    plt.title("NZZ Test Accuracy vs Time")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "nzz_test_accuracy_vs_time.png"))
    plt.close()

    plt.figure()
    plt.plot(time_record_nzz, test_losses_nzz, label="NZZ")
    plt.xlabel("Time (s)")
    plt.ylabel("Test Loss")
    plt.title("NZZ Test Loss vs Time")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "nzz_test_loss_vs_time.png"))
    plt.close()

    epochs_arr = np.arange(1, num_epochs + 1)
    plt.figure()
    plt.plot(epochs_arr, avg_param_counts_nzz, label="NZZ")
    plt.xlabel("Epoch")
    plt.ylabel("Average Parameter Count")
    plt.title("NZZ Parameter Count vs Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "nzz_param_count_vs_epochs.png"))
    plt.close()

    print(f"NZZ-only experiment completed. Results saved in: {output_dir}", flush=True)
    return {
        "test_accuracies_nzz": test_accuracies_nzz,
        "time_record_nzz": time_record_nzz,
        "avg_param_counts_nzz": avg_param_counts_nzz,
    }


if __name__ == '__main__':
    num_runs = 3

    master_seed = 12345
    rng = np.random.default_rng(master_seed)
    seeds = rng.integers(low=0, high=2**31 - 1, size=num_runs, dtype=np.int64).tolist()

    for run_id, seed in enumerate(seeds, 1):
        print(f"\n===== NZZ-only Run {run_id}/{num_runs}, seed={int(seed)} =====", flush=True)
        run_compare(seed=int(seed), run_id=run_id)
