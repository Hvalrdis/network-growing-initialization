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

from mycifar10 import MyCIFAR100
import imageclassificationnet_NZZ as imageclassificationnet


dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

cuda_available = torch.cuda.is_available()
print("CUDA disponible :", cuda_available)
if cuda_available:
    print("Nombre de GPUs disponibles :", torch.cuda.device_count())
    print("GPU actuel :", torch.cuda.get_device_name(torch.cuda.current_device()))


MODE_NAME = "mode_NZZ"


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


def grow_mode_NZZ(net, config_add):
    """Apply only the NZZ growth rule to all convolutional layers.

    For imageclassificationnet_NZZ, mode 2 is mapped to NZZ:
      W_new1 = N, W_new2 = Z, W_new3 = Z.
    """
    conv_indices = [i for i, layer in enumerate(net.backbone) if isinstance(layer, torch.nn.Conv2d)]
    last_increase = 0
    print(f'{MODE_NAME} conv_indices = {conv_indices}')

    for i, layer_index in enumerate(conv_indices):
        nb_increase = int(config_add[i]) if i < len(config_add) else 0
        if nb_increase > 0:
            net.expand_conv_layer_mode2(layer_index, nb_increase, last_increase)
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

    data_root = "../cifar100"

    device = dev
    _set_seed(seed)

    output_dir = os.path.join(
        "outputs_compare",
        "vgg_NZZ_only",
        "cifar100",
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

    model_nzz = build_vgg11_seed_net(
        nbClasses,
        device=device,
        with_batchnorm=False,
        width_multiplier=0.25,
    ).to(device)

    def make_opt_sch(net):
        opt = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
        return opt, sch

    optimizer_nzz, scheduler_nzz = make_opt_sch(model_nzz)

    global_step_nzz = 0
    grow_done_nzz = 0
    total_time_nzz = 0.0

    test_accuracies_nzz = []
    time_record_nzz = []
    avg_param_counts_nzz = []
    train_losses_nzz = []
    train_accs_nzz = []
    test_losses_nzz = []

    def _train_one_step(tag, net, optimizer, scheduler, x, y, current_global_step, current_grow_done):
        net.train()
        optimizer.zero_grad(set_to_none=True)
        logits = net(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        scheduler.step()

        bs = y.size(0)
        correct = (logits.argmax(dim=1) == y).sum().item()
        current_global_step += 1

        if (current_global_step in grow_triggers) and (current_grow_done < grow_steps):
            current_lr = optimizer.param_groups[0]["lr"]
            opt_snap = _snapshot_optimizer_state_by_name(optimizer, net)
            sched_snap = scheduler.state_dict()

            grow_mode_NZZ(net, add_per_grow)

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

    for epoch in range(1, num_epochs + 1):
        train_sum = 0.0
        train_correct = 0
        train_total = 0

        for x, y in loaderTrain:
            x = x.to(device)
            y = y.to(device)

            t0 = time.time()
            loss_sum, corr, bs, optimizer_nzz, scheduler_nzz, global_step_nzz, grow_done_nzz = _train_one_step(
                MODE_NAME,
                model_nzz,
                optimizer_nzz,
                scheduler_nzz,
                x,
                y,
                global_step_nzz,
                grow_done_nzz,
            )
            total_time_nzz += time.time() - t0
            train_sum += loss_sum
            train_correct += corr
            train_total += bs

        train_loss = train_sum / max(1, train_total)
        train_acc = train_correct / max(1, train_total)
        test_loss, test_acc = evaluate(model_nzz, loaderTest, loss_fn, device)

        train_losses_nzz.append(train_loss)
        train_accs_nzz.append(train_acc)
        test_losses_nzz.append(test_loss)
        test_accuracies_nzz.append(test_acc)
        time_record_nzz.append(total_time_nzz)
        avg_param_counts_nzz.append(_count_params(model_nzz))

        print(
            f"[Epoch {epoch:03d}] {MODE_NAME}={test_acc * 100:.2f}% "
            f"train={train_acc * 100:.2f}% "
            f"grow_done={grow_done_nzz}/{grow_steps} "
            f"params={avg_param_counts_nzz[-1]}",
            flush=True,
        )

    epochs_arr = np.arange(1, num_epochs + 1)
    results = {
        "epochs": epochs_arr,
        "test_accuracies_mode_NZZ": test_accuracies_nzz,
        "time_record_mode_NZZ": time_record_nzz,
        "avg_param_counts_mode_NZZ": avg_param_counts_nzz,
    }

    filename = os.path.join(output_dir, f"results_run_{run_id}_NZZ.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write("Epoch\tModel\tTrainLoss\tTrainAcc\tTestLoss\tTestAcc\tTime(s)\tParamCount\n")
        for i in range(num_epochs):
            f.write(
                f"{i + 1}\t{MODE_NAME}\t"
                f"{train_losses_nzz[i]:.6f}\t{train_accs_nzz[i]:.6f}\t"
                f"{test_losses_nzz[i]:.6f}\t{test_accuracies_nzz[i]:.6f}\t"
                f"{time_record_nzz[i]:.2f}\t{avg_param_counts_nzz[i]}\n"
            )

    plt.figure()
    plt.plot(time_record_nzz, test_accuracies_nzz, label=MODE_NAME)
    plt.xlabel("Time (s)")
    plt.ylabel("Test Accuracy")
    plt.title("NZZ Test Accuracy vs Time")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "nzz_test_accuracy_vs_time.png"))
    plt.close()

    plt.figure()
    plt.plot(time_record_nzz, test_losses_nzz, label=MODE_NAME)
    plt.xlabel("Time (s)")
    plt.ylabel("Test Loss")
    plt.title("NZZ Test Loss vs Time")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "nzz_test_loss_vs_time.png"))
    plt.close()

    plt.figure()
    plt.plot(epochs_arr, avg_param_counts_nzz, label=MODE_NAME)
    plt.xlabel("Epoch")
    plt.ylabel("Average Parameter Count")
    plt.title("NZZ Average Parameter Count vs Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "nzz_avg_param_count_vs_epochs.png"))
    plt.close()

    print(f"NZZ-only experiment completed. Results saved in: {output_dir}", flush=True)
    return results


if __name__ == '__main__':
    num_runs = 3

    master_seed = 417
    rng = np.random.default_rng(master_seed)
    seeds = rng.integers(low=0, high=2**5 - 1, size=num_runs, dtype=np.int64).tolist()

    for run_id, seed in enumerate(seeds, 1):
        print(f"\n===== Run {run_id}/{num_runs}, seed={int(seed)} =====", flush=True)
        run_compare(seed=int(seed), run_id=run_id)