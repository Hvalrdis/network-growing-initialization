# Neural Network Growing: Initialization Modes

This repository contains preliminary reproducibility material for the manuscript:

**Neural Network Growing: On the Impact of Different Initialization Modes**

The manuscript studies progressive neural-network growth and compares several initialization modes for newly added parameters. The current public release is intended to provide an accessible entry point for reviewers and readers. The code will be cleaned and documented further while keeping this repository URL unchanged.

## Current contents

```text
legacy_code/      Current PyTorch implementation and experiment scripts
results/          Placeholder for result summaries and reproduction notes
data/             Dataset download and preparation notes
requirements.txt  Minimal Python dependency list
```

The `legacy_code/` directory currently contains the working research code used during the project. It includes implementations for growth operations, GradMax-style expansion, MLP experiments, VGG-11 experiments, WideResNet experiments, and support code for ViT-style growth. The scripts are preserved in their current form at this preliminary stage so that the submitted implementation remains traceable.

## Methods covered

The code is related to the following experimental components of the manuscript:

- progressive width growth of neural networks;
- initialization modes for newly introduced parameters;
- comparisons with a fixed large baseline;
- comparisons with GradMax-style growth for selected convolutional and fully connected architectures;
- experiments on MNIST, CIFAR-10, and CIFAR-100.

The manuscript also reports experiments on transformer-based architectures. The repository will be updated with a cleaner organisation of all final experiment scripts.

## Data

No new dataset was generated in this study. The experiments use public datasets:

- MNIST
- CIFAR-10
- CIFAR-100

The datasets are not redistributed in this repository. Please download them from the original providers and follow their corresponding licences and terms of use. See `data/README.md` for the expected local directory layout used by the current scripts.

## Environment

The current code was developed with Python and PyTorch. A minimal dependency list is provided in `requirements.txt`.

Install dependencies with:

```bash
pip install -r requirements.txt
```

A GPU-enabled PyTorch installation is recommended for reproducing the full experiments.

## Preliminary usage

The current scripts use relative dataset paths. After downloading the datasets, organise the directories as described in `data/README.md`, then run scripts from inside `legacy_code/`.

Examples:

```bash
cd legacy_code
python main_mlp.py       # MLP on MNIST
python main_vgg.py       # VGG-11 on CIFAR-10
python main_vgg100.py    # VGG-11 on CIFAR-100
python main_wrn_new.py   # WRN-28-1 on CIFAR-10/CIFAR-100
```

The scripts write outputs to `outputs_compare/` by default. Large output folders, raw datasets, caches, and checkpoints should not be committed to the repository.

## Reproducibility status

This is a preliminary public repository for the submitted manuscript. The immediate purpose is to provide a stable public link containing the current implementation and data-access information. The repository will be progressively updated with:

- a cleaner source-code layout;
- command-line configuration files;
- complete reproduction commands for each table and figure;
- final result summaries;
- versioned release material after the code clean-up is complete.

The repository URL should remain unchanged.

## Citation

If this repository is used, please cite the corresponding manuscript after publication.
