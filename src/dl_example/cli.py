"""Commands for environment checks, training, and held-out evaluation."""

import argparse
import json
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import torch

from dl_example.config import ExperimentConfig, load_config
from dl_example.data import prepare_data
from dl_example.models import MnistCnn, ShakeShakeResNet
from dl_example.reproducibility import environment_record, select_device, set_seed, write_json
from dl_example.trainer import Trainer, load_checkpoint


def build_model(dataset: str):
    return MnistCnn() if dataset == "mnist" else ShakeShakeResNet()


def check_environment(requested: str) -> dict:
    """Verify a real forward/backward/update step without downloading a dataset."""
    set_seed(42)
    device = select_device(requested)
    model = MnistCnn().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    inputs = torch.randn(4, 1, 28, 28, device=device)
    labels = torch.arange(4, device=device)
    previous = model.fc2.weight.detach().clone()
    loss = torch.nn.functional.cross_entropy(model(inputs), labels)
    loss.backward()
    optimizer.step()
    if not torch.isfinite(loss) or torch.equal(previous, model.fc2.weight):
        raise RuntimeError("The synthetic training check did not update the model")
    return {
        "status": "pass",
        "device": str(device),
        "torch": torch.__version__,
        "synthetic_loss": loss.item(),
        "dataset_downloaded": False,
    }


def train(config: ExperimentConfig) -> Path:
    config.validate()
    set_seed(config.seed)
    device = select_device(config.device)
    # A unique directory prevents a repeated classroom command from overwriting prior work.
    run_id = f"{config.dataset}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    output = Path(config.output_dir) / run_id
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config.to_dict())
    record = {"run_id": run_id, "status": "running", "environment": environment_record(device)}
    write_json(output / "run.json", record)
    started = time.perf_counter()
    tracker = None
    try:
        if config.tracking:
            try:
                import wandb
            except ImportError as error:
                raise ValueError("Tracking requires installation with --extra tracking") from error
            tracker = wandb.init(
                project="dl-example", mode="offline", dir=str(output), config=config.to_dict()
            )
        training, validation, testing, manifest = prepare_data(config, device)
        write_json(output / "data.json", manifest)
        model = build_model(config.dataset).to(device)
        # Move parameters before constructing the optimizer.
        if config.dataset == "mnist":
            optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
            scheduler = None
        else:
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=config.lr,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
        trainer = Trainer(model, optimizer, device, output)
        summary = trainer.fit(training, validation, config, scheduler, tracker)
        load_checkpoint(model, output / "best.pt", device)
        summary["test"] = trainer.evaluate(testing)
        summary["elapsed_seconds"] = time.perf_counter() - started
        summary["scope"] = (
            "smoke" if config.max_train_samples or config.max_eval_samples else "full"
        )
        write_json(output / "summary.json", summary)
        record["status"] = "complete"
        print(json.dumps({"run_dir": str(output), **summary}, indent=2), flush=True)
    except KeyboardInterrupt:
        record.update(status="interrupted")
        raise
    except Exception as error:
        record.update(status="failed", error=str(error))
        raise
    finally:
        write_json(output / "run.json", record)
        if tracker is not None:
            tracker.finish()
    return output


def evaluate(path: Path, requested: str) -> dict:
    # Only load checkpoints produced by a trusted local training run.
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    config = replace(ExperimentConfig(**payload["config"]), device=requested)
    config.validate()
    set_seed(config.seed)
    device = select_device(requested)
    _, _, testing, _ = prepare_data(config, device)
    model = build_model(config.dataset).to(device)
    model.load_state_dict(payload["model_state_dict"])
    trainer = Trainer(model, None, device, path.parent)
    return {
        "checkpoint_epoch": payload["epoch"],
        "test": trainer.evaluate(testing),
        "device": str(device),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="Run a synthetic device and optimizer check")
    check.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    training = commands.add_parser(
        "train", help="Train, select by validation, then evaluate the test split"
    )
    training.add_argument("--dataset", choices=["mnist", "cifar10"])
    training.add_argument("--config")
    training.add_argument("--device", choices=["auto", "cpu", "cuda"])
    for name in (
        "epochs",
        "batch-size",
        "num-workers",
        "seed",
        "max-train-samples",
        "max-eval-samples",
    ):
        training.add_argument(f"--{name}", type=int)
    training.add_argument("--lr", type=float)
    training.add_argument("--output-dir")
    training.add_argument("--data-dir")
    training.add_argument(
        "--smoke", action="store_true", help="Use one epoch and 1024/256/256 examples"
    )
    training.add_argument("--tracking", action="store_true", default=None)
    evaluation = commands.add_parser(
        "evaluate", help="Evaluate a trusted checkpoint on the test split"
    )
    evaluation.add_argument("checkpoint", type=Path)
    evaluation.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args(argv)
    try:
        if args.command == "check":
            print(json.dumps(check_environment(args.device), indent=2))
        elif args.command == "evaluate":
            print(json.dumps(evaluate(args.checkpoint, args.device), indent=2))
        else:
            config = load_config(args.config, args.dataset)
            overrides = {
                name: getattr(args, name)
                for name in config.to_dict()
                if hasattr(args, name) and getattr(args, name) is not None
            }
            if args.smoke:
                overrides.update(epochs=1, max_train_samples=1024, max_eval_samples=256)
            train(replace(config, **overrides))
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted; the run record preserves its status.\n")
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")
