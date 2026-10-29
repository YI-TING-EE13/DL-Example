"""Classification loops, model selection, and evaluation checkpoints."""

import csv
import math
from pathlib import Path

import torch


class Trainer:
    """Train a device-placed classifier with a sample-mean cross-entropy loss."""

    def __init__(self, model, optimizer, device, output_dir: Path):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.output_dir = output_dir
        self.criterion = torch.nn.CrossEntropyLoss()

    def _epoch(self, loader, training: bool) -> dict:
        self.model.train(training)
        loss_sum, correct, count = 0.0, 0, 0
        with torch.set_grad_enabled(training):
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(inputs)
                loss = self.criterion(logits, labels)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite loss encountered")
                if training:
                    loss.backward()
                    self.optimizer.step()
                batch_size = labels.size(0)
                loss_sum += loss.item() * batch_size
                correct += (logits.argmax(1) == labels).sum().item()
                count += batch_size
        if count == 0:
            raise ValueError("Cannot compute metrics for an empty data loader")
        return {"loss": loss_sum / count, "accuracy": correct / count, "samples": count}

    def evaluate(self, loader) -> dict:
        return self._epoch(loader, training=False)

    def fit(self, train_loader, validation_loader, config, scheduler=None, tracker=None) -> dict:
        best_accuracy = -math.inf
        best_epoch, epochs_without_improvement = 0, 0
        with (self.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "epoch",
                    "train_loss",
                    "train_accuracy",
                    "validation_loss",
                    "validation_accuracy",
                    "learning_rate",
                ],
            )
            writer.writeheader()
            for epoch in range(1, config.epochs + 1):
                learning_rate = self.optimizer.param_groups[0]["lr"]
                train = self._epoch(train_loader, training=True)
                validation = self.evaluate(validation_loader)
                if scheduler is not None:
                    scheduler.step()
                row = {
                    "epoch": epoch,
                    "train_loss": train["loss"],
                    "train_accuracy": train["accuracy"],
                    "validation_loss": validation["loss"],
                    "validation_accuracy": validation["accuracy"],
                    "learning_rate": learning_rate,
                }
                writer.writerow(row)
                handle.flush()
                if tracker is not None:
                    tracker.log(row, step=epoch)
                improved = validation["accuracy"] > best_accuracy
                if improved:
                    best_accuracy, best_epoch = validation["accuracy"], epoch
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                payload = {
                    "format_version": 1,
                    "epoch": epoch,
                    "config": config.to_dict(),
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
                    "best_validation_accuracy": best_accuracy,
                    "best_epoch": best_epoch,
                }
                torch.save(payload, self.output_dir / "last.pt")
                if improved:
                    torch.save(payload, self.output_dir / "best.pt")
                print(
                    f"Epoch {epoch}/{config.epochs}: train_loss={train['loss']:.4f} "
                    f"val_accuracy={validation['accuracy']:.4f}",
                    flush=True,
                )
                if (
                    config.early_stop_patience
                    and epochs_without_improvement >= config.early_stop_patience
                ):
                    break
        return {
            "epochs_completed": epoch,
            "best_epoch": best_epoch,
            "best_validation_accuracy": best_accuracy,
        }


def load_checkpoint(model, path: Path, device: torch.device) -> dict:
    """Load a locally produced evaluation checkpoint; training resumption is not implemented."""
    payload = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    model.load_state_dict(payload["model_state_dict"])
    return payload
