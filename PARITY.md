# Implementation parity

The unified code keeps the experiment mechanics that materially affect the
comparison:

| Backbone | Optimizer and schedule                                   | Seed and target widths                                |
| -------- | -------------------------------------------------------- | ----------------------------------------------------- |
| MLP      | SGD 0.05, momentum 0.9, cosine per step                  | 128/64 to 512/256 over 12 growth events               |
| VGG-11   | SGD 0.05, momentum 0.9, cosine per step                  | one-quarter width to full width over 12 growth events |
| WRN-28-1 | SGD 0.01, momentum 0.9, cosine per step                  | one-quarter internal width to full internal width     |
| ViT      | AdamW 1e-3, 10-epoch warmup plus cosine, AMP, CutMix     | hidden dimension 64 to 256 over 12 growth events      |
| CvT-13   | AdamW 6.25e-5, 20-epoch warmup plus cosine, Mixup/CutMix | stage dimensions (16, 48, 96) to (64, 192, 384)       |


For MLP and VGG, GradMax continues to draw its own growth batch as in the source
programs. WRN continues to use the current batch. Optimizer state is retained
across all growth operations; ViT and CvT use head-wise state placement.

Torchvision performs official download and checksum verification, while the
included CIFAR dataset classes preserve the reported augmentation and
normalization pipelines. MLP MNIST uses
`ToTensor()` and is flattened by the model adapter, which is numerically
equivalent to the original raw-byte `/255` conversion.

The CvT-13 adapter consolidates the architecture and growth operations into a
single module and uses the same package-level CIFAR pipeline as ViT. Its
10-class configuration retains exact parameter-count parity at 19,555,146
parameters.

Run the structural regression suite without downloading data:

```bash
python3 -m unified_experiments.verify
```
