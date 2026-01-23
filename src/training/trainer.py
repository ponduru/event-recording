"""Training pipeline for event detection across multiple domains."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.core.base_detector import BaseDetectorConfig, save_checkpoint
from src.core.domain import DomainRegistry

# Import domains to trigger registration
import src.domains  # noqa: F401


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Domain
    domain: str = "cricket"  # Domain name (cricket, soccer, warehouse, etc.)

    # Data
    labels_dir: str = "data/labels"
    videos_dir: str = "data/raw"
    window_size: int = 8
    frame_size: tuple[int, int] = (224, 224)
    target_fps: float = 10.0
    negative_ratio: float = 1.0

    # Model
    backbone: str = "resnet18"
    pretrained: bool = True
    freeze_backbone: bool = False
    feature_dim: int = 512
    hidden_dim: int = 256
    num_lstm_layers: int = 2
    dropout: float = 0.3

    # Training
    batch_size: int = 8
    num_workers: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 50
    early_stopping_patience: int = 10
    val_ratio: float = 0.2

    # Output
    output_dir: str = "models"
    experiment_name: str = "event_detector"

    # Logging
    use_wandb: bool = False
    wandb_project: str = "event-detection"

    def to_model_config(self) -> dict:
        """Create model config dictionary from training config."""
        return {
            "backbone": self.backbone,
            "pretrained": self.pretrained,
            "freeze_backbone": self.freeze_backbone,
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
            "num_lstm_layers": self.num_lstm_layers,
            "dropout": self.dropout,
        }


@dataclass
class TrainingMetrics:
    """Metrics tracked during training."""

    epoch: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    train_accuracy: float = 0.0
    val_accuracy: float = 0.0
    train_precision: float = 0.0
    val_precision: float = 0.0
    train_recall: float = 0.0
    val_recall: float = 0.0
    train_f1: float = 0.0
    val_f1: float = 0.0
    learning_rate: float = 0.0

    def to_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "train_accuracy": self.train_accuracy,
            "val_accuracy": self.val_accuracy,
            "train_precision": self.train_precision,
            "val_precision": self.val_precision,
            "train_recall": self.train_recall,
            "val_recall": self.val_recall,
            "train_f1": self.train_f1,
            "val_f1": self.val_f1,
            "learning_rate": self.learning_rate,
        }


def compute_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> dict:
    """Compute classification metrics.

    Args:
        predictions: Predicted probabilities.
        targets: Ground truth labels.
        threshold: Classification threshold.

    Returns:
        Dict with accuracy, precision, recall, f1.
    """
    preds_binary = (predictions >= threshold).float()

    # True positives, false positives, false negatives
    tp = ((preds_binary == 1) & (targets == 1)).sum().float()
    fp = ((preds_binary == 1) & (targets == 0)).sum().float()
    fn = ((preds_binary == 0) & (targets == 1)).sum().float()
    tn = ((preds_binary == 0) & (targets == 0)).sum().float()

    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "accuracy": accuracy.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
    }


class Trainer:
    """Trainer for event detection models across domains."""

    def __init__(
        self,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: Optional[torch.device] = None,
    ):
        """Initialize trainer.

        Args:
            config: Training configuration.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            device: Device to train on.
        """
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Get domain and create model
        domain = DomainRegistry.get(config.domain)
        model_config = config.to_model_config()
        self.model = domain.create_model(model_config).to(self.device)

        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs,
            eta_min=config.learning_rate / 100,
        )

        # Output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Tracking
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.history: list[TrainingMetrics] = []

        # Wandb
        self.wandb_run = None
        if config.use_wandb:
            try:
                import wandb

                self.wandb_run = wandb.init(
                    project=config.wandb_project,
                    name=config.experiment_name,
                    config={
                        "domain": config.domain,
                        "model": model_config,
                        "training": {
                            "batch_size": config.batch_size,
                            "learning_rate": config.learning_rate,
                            "weight_decay": config.weight_decay,
                            "num_epochs": config.num_epochs,
                        },
                    },
                )
            except ImportError:
                print("Warning: wandb not installed, skipping logging")

    def train_epoch(self) -> tuple[float, dict]:
        """Train for one epoch.

        Returns:
            (loss, metrics) tuple
        """
        self.model.train()

        total_loss = 0.0
        all_preds = []
        all_targets = []

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for frames, labels in pbar:
            frames = frames.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(frames)
            loss = self.criterion(logits, labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

            # Track predictions
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                all_preds.append(probs.cpu())
                all_targets.append(labels.cpu())

            pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(self.train_loader)
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        metrics = compute_metrics(all_preds, all_targets)

        return avg_loss, metrics

    @torch.no_grad()
    def validate(self) -> tuple[float, dict]:
        """Validate the model.

        Returns:
            (loss, metrics) tuple
        """
        self.model.eval()

        total_loss = 0.0
        all_preds = []
        all_targets = []

        for frames, labels in tqdm(self.val_loader, desc="Validation", leave=False):
            frames = frames.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(frames)
            loss = self.criterion(logits, labels)

            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu())
            all_targets.append(labels.cpu())

        avg_loss = total_loss / len(self.val_loader)
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        metrics = compute_metrics(all_preds, all_targets)

        return avg_loss, metrics

    def train(self) -> nn.Module:
        """Run full training loop.

        Returns:
            Trained model
        """
        print(f"Training on {self.device}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")

        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")

            # Train
            train_loss, train_metrics = self.train_epoch()

            # Validate
            val_loss, val_metrics = self.validate()

            # Update scheduler
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            # Record metrics
            metrics = TrainingMetrics(
                epoch=epoch + 1,
                train_loss=train_loss,
                val_loss=val_loss,
                train_accuracy=train_metrics["accuracy"],
                val_accuracy=val_metrics["accuracy"],
                train_precision=train_metrics["precision"],
                val_precision=val_metrics["precision"],
                train_recall=train_metrics["recall"],
                val_recall=val_metrics["recall"],
                train_f1=train_metrics["f1"],
                val_f1=val_metrics["f1"],
                learning_rate=current_lr,
            )
            self.history.append(metrics)

            # Print metrics
            print(
                f"  Train - Loss: {train_loss:.4f}, "
                f"Acc: {train_metrics['accuracy']:.4f}, "
                f"F1: {train_metrics['f1']:.4f}"
            )
            print(
                f"  Val   - Loss: {val_loss:.4f}, "
                f"Acc: {val_metrics['accuracy']:.4f}, "
                f"F1: {val_metrics['f1']:.4f}"
            )

            # Log to wandb
            if self.wandb_run:
                import wandb

                wandb.log(metrics.to_dict())

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0

                checkpoint_path = self.output_dir / f"{self.config.experiment_name}_best.pt"
                # Create a BaseDetectorConfig for saving
                detector_config = BaseDetectorConfig(**self.config.to_model_config())
                save_checkpoint(
                    self.model,
                    detector_config,
                    str(checkpoint_path),
                    self.optimizer,
                    epoch + 1,
                    metrics.to_dict(),
                    domain=self.config.domain,
                )
                print(f"  Saved best model to {checkpoint_path}")
            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                print(f"\nEarly stopping after {epoch + 1} epochs")
                break

        # Save final model
        final_path = self.output_dir / f"{self.config.experiment_name}_final.pt"
        detector_config = BaseDetectorConfig(**self.config.to_model_config())
        save_checkpoint(
            self.model,
            detector_config,
            str(final_path),
            self.optimizer,
            self.config.num_epochs,
            domain=self.config.domain,
        )

        if self.wandb_run:
            self.wandb_run.finish()

        return self.model


def train_model(config: TrainingConfig) -> nn.Module:
    """Train an event detector model.

    Args:
        config: Training configuration.

    Returns:
        Trained model.
    """
    # Get domain
    domain = DomainRegistry.get(config.domain)

    # Create datasets using domain
    from src.data.dataset import create_train_val_split

    train_dataset, val_dataset = create_train_val_split(
        domain=domain,
        labels_dir=config.labels_dir,
        videos_dir=config.videos_dir,
        val_ratio=config.val_ratio,
        window_size=config.window_size,
        frame_size=config.frame_size,
        target_fps=config.target_fps,
        negative_ratio=config.negative_ratio,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # Train
    trainer = Trainer(config, train_loader, val_loader)
    model = trainer.train()

    return model
