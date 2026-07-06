# Paper-to-code map

This document maps the current manuscript
*Neural Network Growing: On the Impact of Different Initialization Modes* to
the accompanying experiment code.

## Main comparison (Sections 4.1-4.2, Tables 2-3)

Entry point: `python3 -m nngrow main`.

| Paper backbone | CLI model/dataset | Default epochs | Notes |
|---|---|---:|---|
| MLP | `--model mlp --dataset mnist` | 200 | widths 512-256; seed width 1/4 |
| VGG-11 | `--model vgg --dataset cifar10` or `cifar100` | 200 | stride downsampling; no BatchNorm |
| WRN-28-1 | `--model wrn --dataset cifar10` or `cifar100` | 200 | internal block convolutions grow |
| ViT | `--model vit --dataset cifar10` | 200 | D=256, depth=8, H=4, patch=2 |
| CvT-13 | `--model cvt --dataset cifar10` | 100 | three-stage dimensions 64-192-384 |

All main runs use a 1/4-width seed, split at optimization step 10,000, then
grow 12 times at intervals of 2,500 steps. Mode C denotes Row-Zero
Initialization.
GradMax is included only for MLP, VGG, and WRN, as in the manuscript.
Use `--modes` followed by one or more mode keys to run a subset. The aliases
`all_modes_NoGradMax` and `all_modes_WithGradMax` select the complete mode set
without or with GradMax, respectively. Add `big_net` explicitly when the Big
Net baseline should be trained; omitting `--modes` retains the full default run.
The number of seed-to-target expansions is configurable through `--grow-steps`.
Model adapters reject counts that produce empty layer growth, violate MHA
head partitions, exceed a GradMax SVD-rank limit, or cannot finish within the
configured optimization schedule.

The CvT-13 preset uses the reported 100-epoch protocol, preserves AdamW state
across every growth event, and applies the reference timm RandAugment, random
erasing, Mixup, and CutMix pipeline.

Each main run exports train/test accuracy and loss against both wall-clock time
and epoch, together with the parameter-count trajectory against epoch.

## ViT growth-axis ablation (Appendix A, Table 4)

Entry point:

```bash
python3 -m nngrow vit-axis --dataset cifar10
```

Both branches start at hidden dimension 64 and finish at 256 in 12 equal
growth steps:

| Paper label | Fixed quantity | Grown quantity | Final layout |
|---|---|---|---|
| Grow-d | H=4 heads | per-head dimension d: 16 to 64 | H=4, d=64 |
| Grow-H | d=16 | head count H: 4 to 16 | H=16, d=16 |

The package exposes the four rows reported in the manuscript: Grow-d (B),
Grow-d (D), Grow-H (B), and Grow-H (D).

## Optimizer-state ablation (Appendix B, Table 5)

Entry point:

```bash
python3 -m nngrow optimizer-state
```

The release preserves the paper design: CIFAR-10, no data augmentation, a
three-stage Conv-BN-ReLU-MaxPool network, seed widths 8-16-32, one growth before
epoch 25, 80 epochs, batch size 500, LR 5e-4, and three runs. It evaluates SGD
(momentum 0.9) and AdamW (weight decay 0.01) over target widths 32-64-128,
64-128-256, and 128-256-512, with Modes A, B, D, and E.

Reported state strategies:

| CLI/result key | Paper column | Behavior |
|---|---|---|
| `keep_state` | Keep State | copy old tensor state; zero state for added entries; preserve scalar step |
| `reset_state` | Reset State | reset optimizer state while continuing the cosine schedule |
| `keep_moments_reset_step` | Keep Moments, Reset Step | AdamW only; copy moments but reset scalar step |

`rebuild_restart_scheduler` is an optional condition selected with
`--include-scheduler-restart`. It is omitted by default because restarting the
cosine schedule changes the learning-rate treatment and is not a clean optimizer-
state-only comparison, matching the manuscript's explanation.

## Consolidated implementation

The model modules contain the architectures and parameter-growth operations used
by the release. The structured ablation modules provide publication entry points
that preserve the reported settings while making each controlled variable
explicit.
