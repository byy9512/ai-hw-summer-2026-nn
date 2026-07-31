# ai-hw-summer-2026-nn

Small image recognition neural networks trained on MNIST, comparing three architectures:

- **MLP** — shallow multi-layer perceptron
- **CNN** — convolutional neural network
- **Transformer** — ViT-style transformer encoder (patch embedding + class token)

## Data

[MNIST](http://yann.lecun.com/exdb/mnist/) — grayscale images of handwritten digits (0–9).

- **Source**: [`torchvision.datasets.MNIST`](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html) (mirrors [huggingface.co/datasets/ylecun/mnist](https://huggingface.co/datasets/ylecun/mnist))
- **Image format**: 1 channel (grayscale), 28×28 pixels
- **Classes**: 10 (digits 0–9)
- **Splits**: 60,000 training images / 10,000 test images
- **Usage rule**: the model is trained only on the train split; the test split is used only for evaluation and is never seen during training

## Models

All three models take a batch of `(B, 1, 28, 28)` images and output `(B, 10)` class logits.

### MLP — [`models/mlp.py`](models/mlp.py)

A shallow multi-layer perceptron, used as the simplest baseline.

- Flattens the 28×28 image into a 784-length vector
- `Linear(784 → 128)` → `ReLU` → `Dropout(0.2)` → `Linear(128 → 10)`
- One hidden layer only ("shallow"), ~102K parameters
- Ignores spatial structure entirely — every pixel is treated as an independent input feature

### CNN — [`models/cnn.py`](models/cnn.py)

A small convolutional network that exploits the 2D structure of the image.

- Block 1: `Conv2d(1 → 32, kernel 3×3, padding 1)` → `ReLU` → `MaxPool2d(2)` (28×28 → 14×14)
- Block 2: `Conv2d(32 → 64, kernel 3×3, padding 1)` → `ReLU` → `MaxPool2d(2)` (14×14 → 7×7)
- Classifier head: flatten (64×7×7) → `Linear(→ 128)` → `ReLU` → `Dropout(0.25)` → `Linear(128 → 10)`
- ~422K parameters
- Convolution + pooling let it learn local, translation-invariant features (edges, strokes, loops) rather than raw pixel positions

### Transformer Encoder — [`models/transformer.py`](models/transformer.py)

A ViT-style ("Vision Transformer") encoder-only transformer.

- **Patch embedding**: the 28×28 image is split into 4×4 patches (7×7 = 49 patches total), each linearly projected to a 64-dim embedding via a `Conv2d(kernel=4, stride=4)` (equivalent to a per-patch linear layer)
- **Class token**: a learnable `[CLS]` embedding is prepended to the sequence of 49 patch embeddings (50 tokens total), following the BERT/ViT convention — its output representation is used for classification
- **Position embedding**: a learnable positional embedding is added to all 50 tokens so the model can distinguish patch locations, since self-attention itself is position-agnostic
- **Encoder**: 4 stacked `TransformerEncoderLayer`s (`nn.TransformerEncoder`), each with 4 attention heads, embedding dim 64, feed-forward dim 128, GELU activation, and dropout 0.1
- **Head**: `LayerNorm` → `Linear(64 → 10)` applied to the final `[CLS]` token representation
- ~139K parameters
- Self-attention lets every patch attend to every other patch directly, giving it a global receptive field from the first layer (unlike the CNN, which builds up receptive field gradually through pooling)

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
