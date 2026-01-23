"""CLI script for training event detection models across domains."""

import click
import yaml
from pathlib import Path

from src.training.trainer import TrainingConfig, train_model
from src.core.domain import DomainRegistry

# Import domains to trigger registration
import src.domains  # noqa: F401


@click.command()
@click.option(
    "--domain",
    "-d",
    type=click.Choice(["cricket", "soccer", "warehouse"]),
    default="cricket",
    help="Domain for event detection (cricket, soccer, warehouse)",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to YAML config file",
)
@click.option(
    "--labels-dir",
    type=click.Path(exists=True),
    default="data/labels",
    help="Directory containing label JSON files",
)
@click.option(
    "--videos-dir",
    type=click.Path(exists=True),
    default="data/raw",
    help="Directory containing video files",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default="models",
    help="Output directory for checkpoints",
)
@click.option(
    "--experiment-name",
    "-n",
    default="delivery_detector",
    help="Name for this experiment",
)
@click.option(
    "--backbone",
    type=click.Choice(["resnet18", "resnet34", "resnet50"]),
    default="resnet18",
    help="CNN backbone architecture",
)
@click.option(
    "--batch-size",
    "-b",
    type=int,
    default=8,
    help="Training batch size",
)
@click.option(
    "--epochs",
    "-e",
    type=int,
    default=50,
    help="Number of training epochs",
)
@click.option(
    "--learning-rate",
    "-lr",
    type=float,
    default=1e-4,
    help="Learning rate",
)
@click.option(
    "--window-size",
    "-w",
    type=int,
    default=8,
    help="Number of frames per sample window",
)
@click.option(
    "--target-fps",
    type=float,
    default=10.0,
    help="Target FPS for frame extraction",
)
@click.option(
    "--freeze-backbone/--no-freeze-backbone",
    default=False,
    help="Freeze CNN backbone weights",
)
@click.option(
    "--use-wandb/--no-wandb",
    default=False,
    help="Enable Weights & Biases logging",
)
@click.option(
    "--wandb-project",
    default="cricket-delivery-detection",
    help="W&B project name",
)
def main(
    domain: str,
    config: str,
    labels_dir: str,
    videos_dir: str,
    output_dir: str,
    experiment_name: str,
    backbone: str,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    window_size: int,
    target_fps: float,
    freeze_backbone: bool,
    use_wandb: bool,
    wandb_project: str,
):
    """Train an event detection model for a specific domain.

    Example usage:

        python -m scripts.train --domain cricket --labels-dir data/labels --videos-dir data/raw

        python -m scripts.train --domain soccer --config configs/domains/soccer.yaml
    """
    # Load config from file if provided
    if config:
        with open(config) as f:
            config_dict = yaml.safe_load(f)
        training_config = TrainingConfig(**config_dict)
    else:
        # Create config from CLI arguments
        training_config = TrainingConfig(
            domain=domain,
            labels_dir=labels_dir,
            videos_dir=videos_dir,
            output_dir=output_dir,
            experiment_name=experiment_name,
            backbone=backbone,
            batch_size=batch_size,
            num_epochs=epochs,
            learning_rate=learning_rate,
            window_size=window_size,
            target_fps=target_fps,
            freeze_backbone=freeze_backbone,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
        )

    # Validate paths
    if not Path(training_config.labels_dir).exists():
        raise click.ClickException(
            f"Labels directory not found: {training_config.labels_dir}"
        )
    if not Path(training_config.videos_dir).exists():
        raise click.ClickException(
            f"Videos directory not found: {training_config.videos_dir}"
        )

    # Check for label files
    label_files = list(Path(training_config.labels_dir).glob("*.json"))
    if not label_files:
        raise click.ClickException(
            f"No label files found in {training_config.labels_dir}. "
            "Use 'python -m scripts.label' to create labels first."
        )

    click.echo(f"Domain: {training_config.domain}")
    click.echo(f"Found {len(label_files)} label file(s)")
    click.echo(f"Output directory: {training_config.output_dir}")
    click.echo(f"Experiment name: {training_config.experiment_name}")
    click.echo()

    # Train
    model = train_model(training_config)

    click.echo("\nTraining complete!")
    click.echo(f"Best model saved to: {training_config.output_dir}/{training_config.experiment_name}_best.pt")


if __name__ == "__main__":
    main()
