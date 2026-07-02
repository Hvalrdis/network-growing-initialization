# Data

This repository does not redistribute raw datasets.

The current legacy scripts expect the following relative paths when executed from `legacy_code/`:

```text
../MNIST
../cifar10
../cifar100
```

The experiments use public datasets:

- MNIST
- CIFAR-10
- CIFAR-100

Download the datasets from their original providers and place/extract them according to the layout expected by the corresponding loader scripts in `legacy_code/`.
