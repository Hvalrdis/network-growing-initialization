# ViT Grow-d versus Grow-H

```bash
python3 -m unified_experiments.ablations.vit_growth_axis.main --show-config
python3 -m unified_experiments.ablations.vit_growth_axis.main --dataset cifar10
```

The four trained branches are Grow-d (B/D) and Grow-H (B/D). They share the
same seed model, data order, optimizer, scheduler, growth times, total hidden
dimension, and final parameter count. Only the hidden-width decomposition over
attention heads changes.

