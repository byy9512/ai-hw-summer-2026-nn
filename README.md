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

### Revised variants

Each base model has a `*_revised.py` counterpart applying one targeted improvement:

- **[`mlp_revised.py`](models/mlp_revised.py)** — adds `BatchNorm1d` and a second hidden layer (784→256→128→10)
- **[`cnn_revised.py`](models/cnn_revised.py)** — adds `BatchNorm2d` after each convolution
- **[`transformer_revised.py`](models/transformer_revised.py)** — mean-pools all patch tokens instead of using a `[CLS]` token

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision

# train, e.g.:
python3 train.py cnn --epochs 5

# test, e.g.:
python3 test.py cnn
```

## Results

**Preliminary — 1 epoch**, batch size 256, Adam (lr=1e-3), no augmentation. These are a sanity check that the pipeline works end-to-end, not final numbers; full multi-epoch results will replace this table.

| Model | Params | Train loss | Train acc | Test acc |
|---|---|---|---|---|
| MLP | 101,770 | 0.4010 | 88.30% | 94.34% |
| CNN | 421,642 | 0.2560 | 92.08% | 98.01% |
| Transformer | 139,018 | 1.2831 | 53.92% | 77.52% |
| MLP Revised | 235,914 | 0.3423 | 91.31% | 96.60% |
| CNN Revised | 421,834 | 0.2028 | 93.78% | 98.31% |
| Transformer Revised | 138,890 | 1.2528 | 54.67% | 82.08% |

### Analysis

- **CNN wins clearly, even after just one epoch.** Its convolutional inductive bias — local connectivity and weight sharing — matches the structure of image data directly, so it needs very little training to start recognizing strokes and edges. It reaches 98%+ test accuracy in a single pass.
- **MLP is a reasonable baseline but behind the CNN.** It flattens the image and treats every pixel as an independent feature, discarding spatial relationships entirely. It still learns fast (simple architecture, few parameters) but tops out well below the CNN's accuracy.
- **Transformer lags substantially in this single-epoch snapshot** (77.52% vs. 98.01%/94.34%). Self-attention has no built-in spatial locality — every patch attends to every other patch from layer one, with no inherent notion of "nearby" — so the model has to learn spatial structure entirely from the position embeddings and data itself. This takes more training than a CNN, especially in epoch 1. Transformers are also known to be more data/compute-hungry than CNNs; expect the gap to narrow substantially with more epochs.
- **All three revised variants improved over their base version at 1 epoch** — BatchNorm speeds up and stabilizes early convergence for MLP (+2.26 pts) and CNN (+0.30 pts), and mean-pooling gave the transformer a solid early boost (+4.56 pts) over `[CLS]`-token pooling, though it's still far behind the other two architectures at this stage.
- **Note on train vs. test accuracy**: train accuracy is a running average computed batch-by-batch *during* the epoch (including early batches when weights were still near-random), while test accuracy is measured once, after training, using the final weights — so it's expected and not a sign of unusually good generalization that test accuracy comes out higher than train accuracy here.

## Full Training Results (15 Epochs)

**Full run** — 15 epochs, batch size 128, Adam (lr=1e-3), no augmentation.

| Model | Params | Final train loss | Final train acc | Test acc |
|---|---|---|---|---|
| MLP | 101,770 | 0.0392 | 98.70% | 97.92% |
| CNN | 421,642 | 0.0062 | 99.79% | 99.14% |
| Transformer | 139,018 | 0.0630 | 97.98% | 98.32% |
| MLP Revised | 235,914 | 0.0257 | 99.12% | 98.35% |
| CNN Revised | 421,834 | 0.0124 | 99.58% | 99.15% |
| Transformer Revised | 138,890 | 0.0539 | 98.23% | 98.25% |

### Analysis

- **CNN remains the top performer** (99.14%/99.15%), but its lead over the other architectures has shrunk drastically compared to epoch 1. Given enough training, all three architectures converge toward MNIST's practical ceiling (~99%+) — the gap that mattered most in the 1-epoch snapshot was mostly a *convergence speed* gap, not a hard ceiling on what each architecture can ultimately learn.
- **Transformer closed almost its entire deficit.** From 77.52% at epoch 1 to 98.32% at epoch 15 — it now beats the plain MLP and sits within ~0.8 points of the CNN. This confirms the earlier prediction: self-attention lacks a built-in spatial locality prior, so it needs more gradient steps to learn from data what convolution gets "for free" architecturally, but once it has enough training it's very competitive on a small, well-behaved dataset like MNIST.
- **MLP is now clearly the weakest of the three base architectures** (97.92%), a reversal from the 1-epoch table where it beat the transformer. With enough training, the architectures that model spatial structure (CNN, and eventually the transformer via learned position embeddings) pull ahead of the one that never sees spatial structure at all.
- **Revised vs. base, at full training, is a mixed bag** — MLP Revised (+0.43 pts) and CNN Revised (+0.01 pts, i.e. essentially tied) still edge out their base versions, but Transformer Revised (98.25%) actually ends up marginally *below* base Transformer (98.32%), reversing its 1-epoch advantage. Mean-pooling gave the transformer a faster start, but by epoch 15 the `[CLS]`-token variant caught up and edged slightly ahead — the difference (0.07 pts) is well within run-to-run noise, so this is best read as "roughly equivalent at convergence" rather than a real regression.

## Comparison: 1 Epoch vs. Full Training (15 Epochs)

| Model | 1-Epoch Test Acc | 15-Epoch Test Acc | Δ |
|---|---|---|---|
| MLP | 94.34% | 97.92% | +3.58 |
| CNN | 98.01% | 99.14% | +1.13 |
| Transformer | 77.52% | 98.32% | +20.80 |
| MLP Revised | 96.60% | 98.35% | +1.75 |
| CNN Revised | 98.31% | 99.15% | +0.84 |
| Transformer Revised | 82.08% | 98.25% | +16.17 |

- **CNN was already near its ceiling after 1 epoch** (+1.13 / +0.84 pts of headroom left) — its inductive bias gets it to a good solution almost immediately, so extra epochs mainly polish, rather than transform, its performance.
- **MLP improves steadily but moderately** (+3.58 / +1.75 pts) — it's simple enough to converge relatively fast, but it has no architectural advantage left to unlock with more training, so its ceiling is inherently lower than the other two.
- **Transformer improves by far the most** (+20.80 / +16.17 pts) — this is the headline result of the full run. The 1-epoch table made the transformer look like a clearly inferior architecture for this task; the 15-epoch table shows that conclusion was really about training budget, not architecture. Ranking models on a single epoch would have been misleading here — it rewarded architectures with strong inductive biases (CNN) and penalized ones that must learn structure from data (transformer), even though the latter is competitive once given comparable training.
- **Practical takeaway**: convergence speed and final accuracy are different axes, and a fair architecture comparison needs to control for training budget — otherwise a model that "loses" early may simply need more epochs, not a better architecture.

## Image Augmentation Comparison (1 Epoch)

To try to raise test accuracy, `data.py` supports an `augment` flag (`train.py --augment`) that applies random augmentation to the **train** split only — the test split is always left unaugmented so evaluation stays a fixed, fair benchmark. The augmentation pipeline: `RandomRotation(10°)` → `RandomAffine` (±10% translate, 0.9–1.1x scale) → `RandomErasing` (p=0.1, small patches).

**Same settings as the original 1-epoch run** — 1 epoch, batch size 256, Adam (lr=1e-3) — with augmentation turned on, for direct comparison:

| Model | Unaugmented Test Acc (1 Epoch) | Augmented Test Acc (1 Epoch) | Δ |
|---|---|---|---|
| MLP | 94.34% | 92.90% | -1.44 |
| CNN | 98.01% | 97.79% | -0.22 |
| Transformer | 77.52% | 57.62% | -19.90 |
| MLP Revised | 96.60% | 95.57% | -1.03 |
| CNN Revised | 98.31% | 97.90% | -0.41 |
| Transformer Revised | 82.08% | 66.90% | -15.18 |

### Analysis

- **Augmentation hurt every single model at 1 epoch** — an important and slightly counterintuitive result, since the assignment frames augmentation as "a good opportunity to add test %." At this training budget it does the opposite.
- **Why**: augmentation makes each training example harder — a rotated, shifted, partially-erased digit is objectively more difficult to fit than the clean original. Train accuracy dropped sharply across the board (e.g. MLP: 88.30% → 71.15%, transformer: 53.92% → 35.35%), confirming the model is spending its one epoch learning from a noisier, harder-to-fit distribution rather than the clean one. Augmentation is fundamentally a regularizer that trades a bit of easy training-set fit for better generalization — but that trade only pays off once the model has had enough epochs to actually learn the invariances (rotation/shift/occlusion robustness) the augmentation is trying to teach. One epoch isn't enough time for that payoff to materialize; all you see here is the added difficulty.
- **The transformer suffered by far the most** (-19.90 / -15.18 pts). It was already the slowest-converging architecture at 1 epoch (see the earlier comparison — it needed ~15 epochs to close its gap with the CNN). Stacking augmentation-induced difficulty on top of an already-too-small training budget compounds the problem: it now has even less signal per epoch to learn both "what a digit is" and "what a digit looks like under distortion."
- **CNN was hurt the least** (-0.22 / -0.41 pts). Its convolutional weight-sharing already gives it some built-in translation invariance — part of what augmentation is trying to teach the other architectures from scratch, the CNN already gets architecturally, so the extra example variability is comparatively less "new" for it to absorb.
- **Expected next result**: based on the 1-epoch-vs-15-epoch pattern seen with the un-augmented models, augmentation's benefit is likely a *longer-horizon* effect. The natural follow-up is running augmented training for the full 15 epochs and checking whether it now beats the un-augmented 15-epoch results (the actual question the assignment is asking) — 1 epoch is too short a budget to fairly judge whether augmentation helps.

## Full Augmented Training Results (15 Epochs)

Same augmentation pipeline as above (`RandomRotation(10°)` → `RandomAffine` → `RandomErasing`, train split only), now run for the full 15 epochs, batch size 128, Adam (lr=1e-3) — matching the settings of the original [Full Training Results](#full-training-results-15-epochs) exactly, so the two are directly comparable.

| Model | Final train loss | Final train acc | Test acc |
|---|---|---|---|
| MLP | 0.2306 | 92.89% | 97.76% |
| CNN | 0.0570 | 98.19% | 99.42% |
| Transformer | 0.1413 | 95.42% | 98.72% |
| MLP Revised | 0.1632 | 94.81% | 98.47% |
| CNN Revised | 0.0696 | 97.84% | 99.44% |
| Transformer Revised | 0.1188 | 96.22% | 98.43% |

## Comparison: Unaugmented vs. Augmented (15 Epochs)

| Model | Unaugmented Test Acc | Augmented Test Acc | Δ |
|---|---|---|---|
| MLP | 97.92% | 97.76% | -0.16 |
| CNN | 99.14% | 99.42% | +0.28 |
| Transformer | 98.32% | 98.72% | +0.40 |
| MLP Revised | 98.35% | 98.47% | +0.12 |
| CNN Revised | 99.15% | 99.44% | +0.29 |
| Transformer Revised | 98.25% | 98.43% | +0.18 |

### Analysis

- **The predicted reversal happened.** At 1 epoch, augmentation hurt every model (up to -19.90 pts for the transformer). At 15 epochs, it now *helps* 5 of the 6 models. This confirms augmentation is a longer-horizon regularizer: it needs enough training time for the model to actually learn the invariances (rotation/shift/occlusion robustness) the distorted examples are teaching, rather than just seeing them as harder noise.
- **CNN Revised (augmented) is the best model in the entire project so far — 99.44% test accuracy**, edging out augmented CNN (99.42%) and beating every unaugmented variant. This directly answers the assignment's premise that augmentation is "a good opportunity to add test %" — it was, but only once given a training budget long enough to pay off.
- **Transformer and Transformer Revised gained the most from augmentation** (+0.40 / +0.18 pts) among the models that improved, on top of already being the architectures that gained the most from more epochs in general (see the earlier 1-epoch-vs-15-epoch comparison). Interpretation: architectures with weaker built-in inductive biases have more "invariances" left to learn from data, so they have the most to gain from both extra epochs and extra augmented variety — the two effects compound in the same direction for the transformer.
- **Plain MLP is the one model that did *not* benefit** (-0.16 pts, essentially flat/noise-level). It has no spatial structure modeling at all — no convolutional translation invariance, no attention-based flexible spatial reasoning — so it has the hardest time using augmented examples to learn a genuinely more robust representation; it likely just needs a rotated/shifted image to look sufficiently "different" from the original in raw pixel space, without any architectural bias to help it recognize them as the same underlying digit. Notably, MLP Revised (more capacity + BatchNorm) *did* benefit slightly (+0.12), suggesting that once you add capacity, the model can start extracting a small amount of value from augmentation even without spatial architecture.
- **CNN's gain from augmentation, while real, is modest** (+0.28 / +0.29 pts) — consistent with the 1-epoch finding that the CNN was hurt least by augmentation to begin with. Convolution already encodes much of what augmentation teaches (some translation robustness), so there's simply less headroom left for augmentation to unlock; most of the CNN's ceiling was already being reached architecturally, not through data variety.
- **Overall conclusion for this assignment**: augmentation is a net positive at a full training budget (best model overall uses it), but the "1-epoch snapshot" would have wrongly suggested the opposite — the same training-budget caveat that applied to the architecture comparison applies here too.
