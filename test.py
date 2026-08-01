import argparse
from pathlib import Path

import torch
import torch.nn as nn

from data import get_dataloaders
from registry import MODEL_REGISTRY
from train import get_device

CHECKPOINT_DIR = Path("checkpoints")


def evaluate_model(model: nn.Module, test_loader, device: torch.device) -> float:
    """Returns test accuracy as a percentage. Assumes model is already on device."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total


def main():
    parser = argparse.ArgumentParser(description="Test an MNIST model on the test split")
    parser.add_argument("model", choices=MODEL_REGISTRY.keys())
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    device = get_device()
    print(f"using device: {device}")

    _, test_loader = get_dataloaders(batch_size=args.batch_size)
    model = MODEL_REGISTRY[args.model]().to(device)

    ckpt_path = Path(args.checkpoint) if args.checkpoint else CHECKPOINT_DIR / f"{args.model}.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    accuracy = evaluate_model(model, test_loader, device)
    print(f"test accuracy for {args.model}: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
