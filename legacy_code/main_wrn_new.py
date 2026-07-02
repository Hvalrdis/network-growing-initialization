import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler, RandomSampler
import torch
import copy, os, time
import matplotlib.pyplot as plt
import random
import json
import sys
import platform
import logging
import traceback


from mycifar10 import MyCIFAR10_2, MyCIFAR100
from wrn_gradmax_net import WideResNetGradMax


dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Vérifie si CUDA est disponible
cuda_available = torch.cuda.is_available()
print("CUDA disponible :", cuda_available)

# Affiche le nombre de GPUs disponibles et le GPU actuellement utilisé
if cuda_available:
    print("Nombre de GPUs disponibles :", torch.cuda.device_count())
    print("GPU actuel :", torch.cuda.get_device_name(torch.cuda.current_device()))

DBs = ['cifar100','cifar10']
# Like SubsetRandomSampler without random shuffle
class MySubsetSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


class _TeeIO:

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
        self.flush()

    def flush(self):
        for s in self._streams:
            s.flush()

    def isatty(self):
        return any(getattr(s, "isatty", lambda: False)() for s in self._streams)


class _RunArtifacts:

    def __init__(self, output_dir: str, config: dict):
        self.output_dir = output_dir
        self.config = config

        cfg_path = os.path.join(self.output_dir, "config.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2, sort_keys=True)

        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        self.run_log_path = os.path.join(self.output_dir, "run.log")
        self._run_log_f = open(self.run_log_path, "w", buffering=1, encoding="utf-8")
        sys.stdout = _TeeIO(self._orig_stdout, self._run_log_f)
        sys.stderr = _TeeIO(self._orig_stderr, self._run_log_f)

        self.log_path = os.path.join(self.output_dir, "log.txt")
        self.logger = logging.getLogger(f"wrn_run_{os.path.basename(self.output_dir)}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

        self._file_handler = logging.FileHandler(self.log_path, encoding="utf-8")
        self._file_handler.setLevel(logging.INFO)
        self._file_handler.setFormatter(fmt)
        self.logger.addHandler(self._file_handler)

        self._stream_handler = logging.StreamHandler(stream=sys.stdout)
        self._stream_handler.setLevel(logging.INFO)
        self._stream_handler.setFormatter(fmt)
        self.logger.addHandler(self._stream_handler)

        self.logger.info("Run artifacts initialized：output_dir=%s", self.output_dir)

    def close(self):
        try:
            sys.stdout = self._orig_stdout
            sys.stderr = self._orig_stderr
        except Exception:
            pass

        try:
            if hasattr(self, "logger") and self.logger is not None:
                for h in list(self.logger.handlers):
                    try:
                        h.flush()
                    except Exception:
                        pass
                    try:
                        h.close()
                    except Exception:
                        pass
                    self.logger.removeHandler(h)
        except Exception:
            pass

        try:
            if hasattr(self, "_run_log_f") and self._run_log_f is not None:
                self._run_log_f.flush()
                self._run_log_f.close()
        except Exception:
            pass


def _build_run_config(
    *,
    db: str,
    seed: int,
    run_id: int,
    output_dir: str,
    device,
    num_epochs: int,
    batch_size: int,
    batch_size_test: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    num_workers: int,
    grow_start_iter: int,
    grow_every: int,
    grow_batch: int,
) -> dict:
    cfg = {
        "db": db,
        "seed": seed,
        "run_id": run_id,
        "output_dir": output_dir,
        "device": str(device),
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "batch_size_test": batch_size_test,
        "lr": lr,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "num_workers": num_workers,
        "grow_start_iter": grow_start_iter,
        "grow_every": grow_every,
        "grow_batch": grow_batch,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "env": {
            "python": sys.version.replace("\\n", " "),
            "platform": platform.platform(),
            "numpy": getattr(np, "__version__", None),
            "torch": getattr(torch, "__version__", None),
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": getattr(torch.version, "cuda", None),
            "cudnn_version": torch.backends.cudnn.version() if hasattr(torch.backends, "cudnn") else None,
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "gpu_name": torch.cuda.get_device_name(torch.cuda.current_device())
            if torch.cuda.is_available() and torch.cuda.device_count() > 0
            else None,
        },
    }
    return cfg


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
    return x, y


def gradmax_grow_one_block(
    net: WideResNetGradMax,
    tuple_id: int,
    n_new: int,
    loader_train: DataLoader,
    loss_fn,
    grow_batch_size: int = 128,
    x=None,
    y=None,
):
    device = next(net.parameters()).device

    if (x is None) or (y is None):
        x, y = _grow_batch(loader_train, device=device, grow_batch_size=grow_batch_size)
    else:
        x = x[:grow_batch_size].to(device).detach()
        y = y[:grow_batch_size].to(device).detach()

    net.train()
    net.zero_grad(set_to_none=True)

    net.initForGradMax()
    logits = net.forwardForGradMax(x)
    loss = loss_fn(logits, y)
    loss.backward()

    net.growGradMax(tuple_id=int(tuple_id), n_new=int(n_new))


def gradmax_grow_all_at_once(
    net: WideResNetGradMax,
    n_new_per_tuple: list,
    loader_train: DataLoader,
    loss_fn,
    grow_batch_size: int = 128,
    x=None,
    y=None,
):

    device = next(net.parameters()).device

    if (x is None) or (y is None):
        x, y = _grow_batch(loader_train, device=device, grow_batch_size=grow_batch_size)
    else:
        x = x[:grow_batch_size].to(device).detach()
        y = y[:grow_batch_size].to(device).detach()

    net.train()
    net.zero_grad(set_to_none=True)

    net.initForGradMax()
    logits = net.forwardForGradMax(x)
    loss = loss_fn(logits, y)
    loss.backward()

    for tuple_id, n_new in enumerate(n_new_per_tuple):
        n_new = int(n_new)
        if n_new <= 0:
            continue
        net.growGradMax(tuple_id=int(tuple_id), n_new=int(n_new))

    for group in net.groups:
        for blk in group:
            if hasattr(blk.conv2, "Waux"):
                blk.conv2.Waux = None
                if hasattr(blk.conv2, "Waux_stride"):
                    delattr(blk.conv2, "Waux_stride")


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


def build_wrn_net(
    num_classes: int,
    device: torch.device,
    depth: int = 28,
    width_multiplier: int = 1,
    block_width_multiplier: float = 1.0,
    normalization_type: str = "batchnorm",
    seed: int = 42,
):
    net = WideResNetGradMax(
        num_classes=num_classes,
        depth=depth,
        width_multiplier=width_multiplier,
        block_width_multiplier=block_width_multiplier,
        normalization_type=normalization_type,
        device=device,
        with_bias=False,
        fc_bias=True,
        seed=seed,
    ).to(device)

    net.gradmax_scale_method = "mean_norm"
    net.gradmax_init_scale = 0.5
    net.gradmax_epsilon = 0.0
    return net


def _compute_all_at_once_plan(
    seed_net: WideResNetGradMax,
    big_net: WideResNetGradMax,
    grow_steps: int = 12,
) -> list:

    seed_tuples = seed_net.get_grow_layer_tuples()
    big_tuples = big_net.get_grow_layer_tuples()


    diffs = []
    for t_seed, t_big in zip(seed_tuples, big_tuples):
        conv1_seed = t_seed[0]
        conv1_big = t_big[0]
        diff = int(conv1_big.out_channels) - int(conv1_seed.out_channels)

        diffs.append(diff)

    plan = []
    for g in range(int(grow_steps)):
        plan.append([0 for _ in range(len(diffs))])

    for i, diff in enumerate(diffs):
        base = diff // int(grow_steps)
        rem = diff % int(grow_steps)
        for g in range(int(grow_steps)):
            plan[g][i] = base + (1 if g < rem else 0)

    return plan




def run_compare(seed: int, run_id: int = 1, db: str = "cifar10"):

    num_epochs = 200
    batch_size = 128
    batch_size_test = 512
    lr = 0.01
    momentum = 0.9
    weight_decay = 5e-4
    num_workers = 0

    grow_start_iter = 10000
    grow_every = 2500
    grow_batch = batch_size
    grow_steps = 12

    if db == "cifar10":
        data_root = "../cifar10"
    elif db == "cifar100":
        data_root = "../cifar100"
    else:
        raise ValueError(f"Unsupported dataset: {db}")

    device = dev
    _set_seed(seed)

    output_dir = f"outputs_compare/wrn_new_paper/{db}/{time.strftime('%Y%m%d_%H%M%S')}_run{run_id:02d}_seed{seed}_lr{lr}"
    os.makedirs(output_dir, exist_ok=True)
    run_cfg = _build_run_config(
        db=db,
        seed=seed,
        run_id=run_id,
        output_dir=output_dir,
        device=device,
        num_epochs=num_epochs,
        batch_size=batch_size,
        batch_size_test=batch_size_test,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        num_workers=num_workers,
        grow_start_iter=grow_start_iter,
        grow_every=grow_every,
        grow_batch=grow_batch,
    )
    run_cfg["mode_mapping"] = {
        "big": "fixed WRN-28-1 baseline trained from scratch",
        "mode_a": "Mode A / Column-Zero: Conv1 new filters Kaiming, Conv2 new input columns zero",
        "mode_b": "Mode B / Row-First Column-Zero: degenerates to Mode A for this WRN conv1-only growth",
        "mode_c": "Mode C / Row-Zero: Conv1 new filters zero, Conv2 new input columns Kaiming",
        "mode_d": "Mode D / Homogeneous Kaiming: Conv1 and Conv2 new weights Kaiming",
        "mode_e": "Mode E / Empirical Variance: Conv1 and Conv2 new weights sampled with empirical std",
        "gradmax": "GradMax SVD-based initialization",
    }

    _artifacts = _RunArtifacts(output_dir, run_cfg)
    logger = _artifacts.logger
    try:
        if db == "cifar10":
            dataset_train = MyCIFAR10_2(root=data_root, train=True, device=device, data_aug=True)
            dataset_test = MyCIFAR10_2(root=data_root, train=False, device=device, data_aug=False)
        else:
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
            batch_size=batch_size_test,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        )

        loss_fn = nn.CrossEntropyLoss()
        steps_per_epoch = len(loaderTrain)
        total_steps = num_epochs * steps_per_epoch
        nbClasses = 10 if db == "cifar10" else 100

        model_big = build_wrn_net(nbClasses, device=device, block_width_multiplier=1.0, seed=seed)
        model_seed = build_wrn_net(nbClasses, device=device, block_width_multiplier=0.25, seed=seed)

        all_at_once_plan = _compute_all_at_once_plan(model_seed, model_big, grow_steps=grow_steps)
        grow_triggers = set(grow_start_iter + i * grow_every for i in range(grow_steps))

        mode_order = ["mode_a", "mode_b", "mode_c", "mode_d", "mode_e", "gradmax"]
        result_order = ["big"] + mode_order
        display_names = {
            "big": "Big Net",
            "mode_a": "Mode A",
            "mode_b": "Mode B",
            "mode_c": "Mode C",
            "mode_d": "Mode D",
            "mode_e": "Mode E",
            "gradmax": "GradMax",
        }
        result_file_names = {
            "big": "big",
            "mode_a": "mode_a",
            "mode_b": "mode_b",
            "mode_c": "mode_c",
            "mode_d": "mode_d",
            "mode_e": "mode_e",
            "gradmax": "gradmax",
        }

        def make_opt_sch(net):
            opt = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
            sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
            return opt, sch

        optimizer_big, scheduler_big = make_opt_sch(model_big)
        optimizer_seed, scheduler_seed = make_opt_sch(model_seed)

        models = {"big": model_big}
        optimizers = {"big": optimizer_big, "seed": optimizer_seed}
        schedulers = {"big": scheduler_big, "seed": scheduler_seed}
        global_steps = {"big": 0, "seed": 0}
        total_times = {"big": 0.0, "seed": 0.0}
        grow_done = {name: 0 for name in mode_order}
        split_done = False

        metrics = {
            name: {
                "train_loss": [],
                "train_acc": [],
                "test_loss": [],
                "test_acc": [],
                "time": [],
                "params": [],
            }
            for name in result_order
        }

        def _apply_article_mode(net, grow_iter: int, mode_id: int):
            adds = all_at_once_plan[grow_iter]
            for tuple_id, n_new in enumerate(adds):
                n_new = int(n_new)
                if n_new <= 0:
                    continue
                if mode_id == 1:
                    net.expand_conv_layer_mode_a(int(tuple_id), n_new)
                elif mode_id == 2:
                    net.expand_conv_layer_mode_b(int(tuple_id), n_new)
                elif mode_id == 3:
                    net.expand_conv_layer_mode_c(int(tuple_id), n_new)
                elif mode_id == 4:
                    net.expand_conv_layer_mode_d(int(tuple_id), n_new)
                elif mode_id == 5:
                    net.expand_conv_layer_mode_e(int(tuple_id), n_new)
                else:
                    raise ValueError(f"Unknown article mode_id: {mode_id}")

        def grow_impl_mode_a(net, grow_iter: int, x=None, y=None):
            _apply_article_mode(net, grow_iter, mode_id=1)

        def grow_impl_mode_b(net, grow_iter: int, x=None, y=None):
            _apply_article_mode(net, grow_iter, mode_id=2)

        def grow_impl_mode_c(net, grow_iter: int, x=None, y=None):
            _apply_article_mode(net, grow_iter, mode_id=3)

        def grow_impl_mode_d(net, grow_iter: int, x=None, y=None):
            _apply_article_mode(net, grow_iter, mode_id=4)

        def grow_impl_mode_e(net, grow_iter: int, x=None, y=None):
            _apply_article_mode(net, grow_iter, mode_id=5)

        def grow_impl_gradmax(net, grow_iter: int, x=None, y=None):
            adds = all_at_once_plan[grow_iter]
            if any(int(v) > 0 for v in adds):
                gradmax_grow_all_at_once(
                    net=net,
                    n_new_per_tuple=adds,
                    loader_train=loaderTrain,
                    loss_fn=loss_fn,
                    grow_batch_size=grow_batch,
                    x=x,
                    y=y,
                )

        grow_impls = {
            "mode_a": grow_impl_mode_a,
            "mode_b": grow_impl_mode_b,
            "mode_c": grow_impl_mode_c,
            "mode_d": grow_impl_mode_d,
            "mode_e": grow_impl_mode_e,
            "gradmax": grow_impl_gradmax,
        }

        def _train_one_step(tag, net, optimizer, scheduler, x, y, global_step, grow_done_count, grow_impl):
            net.train()
            optimizer.zero_grad(set_to_none=True)
            logits = net(x)
            loss = loss_fn(logits, y)
            loss.backward()

            if not torch.isfinite(loss):
                print(f"[NaN] loss is {loss.item()} at step {global_step} tag={tag}")
                raise RuntimeError("loss became NaN/Inf")

            for n, p in net.named_parameters():
                if p.grad is None:
                    continue
                if not torch.isfinite(p.grad).all():
                    print(f"[NaN] grad non-finite at {n} step={global_step} tag={tag}")
                    raise RuntimeError("grad became NaN/Inf")

            optimizer.step()

            for n, p in net.named_parameters():
                if not torch.isfinite(p).all():
                    print(f"[NaN] param non-finite at {n} step={global_step} tag={tag}")
                    raise RuntimeError("param became NaN/Inf")

            scheduler.step()

            bs = y.size(0)
            correct = (logits.argmax(dim=1) == y).sum().item()
            global_step += 1

            if (grow_impl is not None) and (global_step in grow_triggers) and (grow_done_count < grow_steps):
                current_lr = optimizer.param_groups[0]["lr"]
                opt_snap = _snapshot_optimizer_state_by_name(optimizer, net)
                sched_snap = scheduler.state_dict()

                grow_impl(net, grow_done_count, x=x, y=y)

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
                grow_done_count += 1
                print(f"[{tag}] iter {global_step}: grow {grow_done_count}/{grow_steps} finished", flush=True)

            return loss.item() * bs, correct, bs, optimizer, scheduler, global_step, grow_done_count

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

        def _maybe_grow_only(tag, net, optimizer, scheduler, global_step, grow_done_count, grow_impl, x=None, y=None):
            if (grow_impl is None) or (grow_done_count >= grow_steps) or (global_step not in grow_triggers):
                return optimizer, scheduler, grow_done_count

            current_lr = optimizer.param_groups[0]["lr"]
            opt_snap = _snapshot_optimizer_state_by_name(optimizer, net)
            sched_snap = scheduler.state_dict()

            grow_impl(net, grow_done_count, x=x, y=y)

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
            grow_done_count += 1
            print(f"[{tag}] iter {global_step}: grow {grow_done_count}/{grow_steps} finished (grow-only@split)", flush=True)
            return optimizer, scheduler, grow_done_count

        def _zero_epoch_stats():
            return {name: {"loss_sum": 0.0, "correct": 0, "total": 0} for name in result_order + ["seed"]}

        def _add_train_stats(epoch_stats, name, loss_sum, correct, batch_size_now):
            epoch_stats[name]["loss_sum"] += float(loss_sum)
            epoch_stats[name]["correct"] += int(correct)
            epoch_stats[name]["total"] += int(batch_size_now)

        def _train_summary(epoch_stats, name):
            total = max(1, epoch_stats[name]["total"])
            return epoch_stats[name]["loss_sum"] / total, epoch_stats[name]["correct"] / total

        for epoch in range(1, num_epochs + 1):
            epoch_stats = _zero_epoch_stats()

            for x, y in loaderTrain:
                x = x.to(device)
                if not torch.isfinite(x).all():
                    raise RuntimeError("input x has NaN/Inf")
                y = y.to(device)

                t0 = time.time()
                loss_sum, corr, bs, optimizers["big"], schedulers["big"], global_steps["big"], _ = _train_one_step(
                    tag="big",
                    net=models["big"],
                    optimizer=optimizers["big"],
                    scheduler=schedulers["big"],
                    x=x,
                    y=y,
                    global_step=global_steps["big"],
                    grow_done_count=0,
                    grow_impl=None,
                )
                total_times["big"] += time.time() - t0
                _add_train_stats(epoch_stats, "big", loss_sum, corr, bs)

                if not split_done:
                    t0 = time.time()
                    loss_sum, corr, bs, optimizers["seed"], schedulers["seed"], global_steps["seed"], _ = _train_one_step(
                        tag="seed(shared)",
                        net=model_seed,
                        optimizer=optimizers["seed"],
                        scheduler=schedulers["seed"],
                        x=x,
                        y=y,
                        global_step=global_steps["seed"],
                        grow_done_count=0,
                        grow_impl=None,
                    )
                    total_times["seed"] += time.time() - t0
                    _add_train_stats(epoch_stats, "seed", loss_sum, corr, bs)

                    if global_steps["seed"] == grow_start_iter:
                        print(f"[INFO] Step {global_steps['seed']}: split seed into Mode A/B/C/D/E and GradMax.", flush=True)

                        for name in mode_order:
                            models[name] = copy.deepcopy(model_seed)
                            optimizers[name], schedulers[name] = _clone_opt_sch_from_seed(
                                models[name], model_seed, optimizers["seed"], schedulers["seed"]
                            )
                            global_steps[name] = global_steps["seed"]
                            total_times[name] = total_times["seed"]
                            epoch_stats[name] = copy.deepcopy(epoch_stats["seed"])

                        for name in mode_order:
                            optimizers[name], schedulers[name], grow_done[name] = _maybe_grow_only(
                                display_names[name],
                                models[name],
                                optimizers[name],
                                schedulers[name],
                                global_steps[name],
                                grow_done[name],
                                grow_impls[name],
                                x=x,
                                y=y,
                            )

                        del model_seed, optimizers["seed"], schedulers["seed"]
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        split_done = True
                        print("[INFO] Split and first scheduled growth completed.", flush=True)

                    continue

                for name in mode_order:
                    t0 = time.time()
                    loss_sum, corr, bs, optimizers[name], schedulers[name], global_steps[name], grow_done[name] = _train_one_step(
                        tag=display_names[name],
                        net=models[name],
                        optimizer=optimizers[name],
                        scheduler=schedulers[name],
                        x=x,
                        y=y,
                        global_step=global_steps[name],
                        grow_done_count=grow_done[name],
                        grow_impl=grow_impls[name],
                    )
                    total_times[name] += time.time() - t0
                    _add_train_stats(epoch_stats, name, loss_sum, corr, bs)

            # End-of-epoch evaluation.
            train_loss_big, train_acc_big = _train_summary(epoch_stats, "big")
            test_loss_big, test_acc_big = evaluate(models["big"], loaderTest, loss_fn, device)
            metrics["big"]["train_loss"].append(train_loss_big)
            metrics["big"]["train_acc"].append(train_acc_big)
            metrics["big"]["test_loss"].append(test_loss_big)
            metrics["big"]["test_acc"].append(test_acc_big)
            metrics["big"]["time"].append(total_times["big"])
            metrics["big"]["params"].append(_count_params(models["big"]))

            if not split_done:
                train_loss_seed, train_acc_seed = _train_summary(epoch_stats, "seed")
                test_loss_seed, test_acc_seed = evaluate(model_seed, loaderTest, loss_fn, device)

                for name in mode_order:
                    metrics[name]["train_loss"].append(train_loss_seed)
                    metrics[name]["train_acc"].append(train_acc_seed)
                    metrics[name]["test_loss"].append(test_loss_seed)
                    metrics[name]["test_acc"].append(test_acc_seed)
                    metrics[name]["time"].append(total_times["seed"])
                    metrics[name]["params"].append(_count_params(model_seed))

                print(
                    f"[Epoch {epoch:03d}] "
                    f"big={test_acc_big*100:.2f}% "
                    f"seed(shared)={test_acc_seed*100:.2f}% "
                    f"(modes use seed stats until split@{grow_start_iter})",
                    flush=True,
                )
            else:
                parts = [f"[Epoch {epoch:03d}] big={test_acc_big*100:.2f}%"]
                for name in mode_order:
                    train_loss, train_acc = _train_summary(epoch_stats, name)
                    test_loss, test_acc = evaluate(models[name], loaderTest, loss_fn, device)
                    metrics[name]["train_loss"].append(train_loss)
                    metrics[name]["train_acc"].append(train_acc)
                    metrics[name]["test_loss"].append(test_loss)
                    metrics[name]["test_acc"].append(test_acc)
                    metrics[name]["time"].append(total_times[name])
                    metrics[name]["params"].append(_count_params(models[name]))
                    parts.append(f"{display_names[name]}={test_acc*100:.2f}%")
                print(" ".join(parts), flush=True)

        epochs_arr = np.arange(1, num_epochs + 1)

        filename = os.path.join(output_dir, f"results_run_{run_id}.txt")
        with open(filename, "w", encoding="utf-8") as f:
            f.write("Epoch\tModel\tTrainLoss\tTrainAcc\tTestLoss\tTestAcc\tTime(s)\tParamCount\n")
            for i in range(num_epochs):
                for name in result_order:
                    f.write(
                        f"{i+1}\t{result_file_names[name]}\t"
                        f"{metrics[name]['train_loss'][i]:.6f}\t"
                        f"{metrics[name]['train_acc'][i]:.6f}\t"
                        f"{metrics[name]['test_loss'][i]:.6f}\t"
                        f"{metrics[name]['test_acc'][i]:.6f}\t"
                        f"{metrics[name]['time'][i]:.2f}\t"
                        f"{metrics[name]['params'][i]}\n"
                    )

        plot_specs = [
            ("test_accuracy_vs_time.png", "Time (s)", "Test Accuracy", "Test Accuracy vs Time", "time", "test_acc"),
            ("train_accuracy_vs_time.png", "Time (s)", "Train Accuracy", "Train Accuracy vs Time", "time", "train_acc"),
            ("test_loss_vs_time.png", "Time (s)", "Test Loss", "Test Loss vs Time", "time", "test_loss"),
            ("train_loss_vs_time.png", "Time (s)", "Train Loss", "Train Loss vs Time", "time", "train_loss"),
            ("avg_param_count_vs_epochs.png", "Epoch", "Parameter Count", "Parameter Count vs Epochs", None, "params"),
        ]

        for filename_plot, xlabel, ylabel, title, x_key, y_key in plot_specs:
            plt.figure()
            for name in result_order:
                x_values = epochs_arr if x_key is None else metrics[name][x_key]
                plt.plot(x_values, metrics[name][y_key], label=display_names[name])
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, filename_plot))
            plt.close()

        print(f"Experiment completed. Results saved in: {output_dir}", flush=True)

    except Exception:
        try:
            logger.exception("error during run:")
        except Exception:
            pass
        raise
    finally:
        _artifacts.close()


if __name__ == '__main__':
    num_runs = 3

    master_seed = 17
    rng = np.random.default_rng(master_seed)
    seeds = rng.integers(low=0, high=2**5 - 1, size=num_runs, dtype=np.int64).tolist()

    for db in DBs:
        for run_id, seed in enumerate(seeds, 1):
            print(f"\n===== Run {run_id}/{num_runs}, seed={int(seed)} =====", flush=True)
            run_compare(seed=int(seed), run_id=run_id, db=db)