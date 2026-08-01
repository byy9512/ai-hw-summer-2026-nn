import argparse
from pathlib import Path

import torch
import torch.nn as nn

from data import get_dataloaders
from registry import MODEL_REGISTRY

CHECKPOINT_DIR = Path("checkpoints")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_model(model: nn.Module, train_loader, device: torch.device, epochs: int = 5, lr: float = 1e-3) -> nn.Module:
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        print(f"epoch {epoch}/{epochs} - loss: {running_loss / total:.4f} - train acc: {100 * correct / total:.2f}%")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train an MNIST model")
    parser.add_argument("model", choices=MODEL_REGISTRY.keys())
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = get_device()
    print(f"using device: {device}")

    train_loader, _ = get_dataloaders(batch_size=args.batch_size)
    model = MODEL_REGISTRY[args.model]()

    train_model(model, train_loader, device, epochs=args.epochs, lr=args.lr)

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / f"{args.model}.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
