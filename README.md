# ai-hw-summer-2026-nn

Small image recognition neural networks trained on MNIST, comparing three architectures:

- **MLP** — shallow multi-layer perceptron
- **CNN** — convolutional neural network
- **Transformer** — ViT-style transformer encoder (patch embedding + class token)

## Data

[MNIST](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html) via `torchvision.datasets.MNIST`. Trained on the train split, evaluated on the test split.

## Project structure

```
models/
  mlp.py            # MLP
  cnn.py            # CNN
  transformer.py    # Transformer encoder (ViT-style)
  __init__.py
```

## Status

Model architectures are implemented and verified to produce correct output shapes (`(batch, 10)` logits). Training loop, evaluation, and image augmentation experiments are still to come.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision
```

## Results

TBD — will be added after training/testing.
