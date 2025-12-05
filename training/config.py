"""Default training configuration for flow matching policy.

Usage:
    from training.config import TrainingConfig, get_config

    # Get default config
    config = get_config()

    # Override specific values
    config = get_config(batch_size=256, lr=4e-4)

    # Use with argparse
    parser = argparse.ArgumentParser()
    config.add_argparse_args(parser)
    args = parser.parse_args()
    config = config.from_argparse(args)
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional
import argparse


@dataclass
class TrainingConfig:
    """Training configuration with sensible defaults for RTX 4090."""

    # Data
    libero_dir: str = "/workspace/LIBERO/datasets"
    suites: List[str] = field(default_factory=lambda: ["libero_spatial", "libero_object"])
    chunk_size: int = 16
    temporal_stride: int = 1

    # Model
    action_dim: int = 7
    hidden_dim: int = 256
    proprio_dim: int = 8
    goal_dim: int = 6  # pick_pos(3) + place_pos(3)
    pretrained_vision: bool = True
    dropout: float = 0.1  # Dropout for regularization

    # Training
    batch_size: int = 256
    lr: float = 4e-4
    weight_decay: float = 1e-4
    epochs: int = 100
    grad_clip: float = 1.0

    # Optimization
    freeze_vision_epochs: int = 100  # Keep vision encoder frozen (prevents overfitting)
    use_amp: bool = True  # Mixed precision training
    augment: bool = True

    # Dataloader
    num_workers: int = 16
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4

    # Device
    device: str = "cuda"
    seed: int = 42

    # Output
    output_dir: Optional[str] = None  # Defaults to training/checkpoints
    save_every: int = 10  # Save checkpoint every N epochs
    log_interval: int = 200  # Log every N batches

    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 10  # Stop if no improvement for N epochs
    early_stopping_min_delta: float = 0.001  # Minimum improvement to reset patience

    # W&B logging
    wandb: bool = False
    wandb_project: str = "yubot-flow-policy"
    wandb_name: Optional[str] = None
    wandb_tags: Optional[List[str]] = None

    def __post_init__(self):
        """Set default output directory if not specified."""
        if self.output_dir is None:
            self.output_dir = str(Path(__file__).parent / "checkpoints")

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def add_argparse_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add config fields as argparse arguments."""
        defaults = cls()

        # Data args
        parser.add_argument("--libero_dir", type=str, default=defaults.libero_dir,
                          help="Path to LIBERO datasets directory")
        parser.add_argument("--chunk_size", type=int, default=defaults.chunk_size,
                          help="Action chunk size")
        parser.add_argument("--temporal_stride", type=int, default=defaults.temporal_stride,
                          help="Sample every Nth timestep")

        # Model args
        parser.add_argument("--hidden_dim", type=int, default=defaults.hidden_dim,
                          help="Hidden dimension for policy network")

        # Training args
        parser.add_argument("--batch_size", type=int, default=defaults.batch_size,
                          help="Batch size")
        parser.add_argument("--lr", type=float, default=defaults.lr,
                          help="Learning rate")
        parser.add_argument("--epochs", type=int, default=defaults.epochs,
                          help="Number of epochs")
        parser.add_argument("--grad_clip", type=float, default=defaults.grad_clip,
                          help="Gradient clipping max norm")

        # Optimization args
        parser.add_argument("--freeze_vision_epochs", type=int, default=defaults.freeze_vision_epochs,
                          help="Freeze vision encoder for first N epochs")
        parser.add_argument("--no_amp", action="store_true",
                          help="Disable mixed precision training")
        parser.add_argument("--augment", action="store_true", default=defaults.augment,
                          help="Enable data augmentation")
        parser.add_argument("--no_augment", action="store_true",
                          help="Disable data augmentation")

        # Dataloader args
        parser.add_argument("--num_workers", type=int, default=defaults.num_workers,
                          help="Dataloader workers")

        # Device args
        parser.add_argument("--device", type=str, default=defaults.device,
                          help="Device to use")
        parser.add_argument("--seed", type=int, default=defaults.seed,
                          help="Random seed")

        # Output args
        parser.add_argument("--output_dir", type=str, default=None,
                          help="Directory to save checkpoints")
        parser.add_argument("--save_every", type=int, default=defaults.save_every,
                          help="Save checkpoint every N epochs")
        parser.add_argument("--log_interval", type=int, default=defaults.log_interval,
                          help="Log every N batches")

        # W&B args
        parser.add_argument("--wandb", action="store_true",
                          help="Enable W&B logging")
        parser.add_argument("--wandb_project", type=str, default=defaults.wandb_project,
                          help="W&B project name")
        parser.add_argument("--wandb_name", type=str, default=None,
                          help="W&B run name")
        parser.add_argument("--wandb_tags", type=str, nargs="*", default=None,
                          help="W&B tags")

        return parser

    @classmethod
    def from_argparse(cls, args: argparse.Namespace) -> "TrainingConfig":
        """Create config from parsed arguments."""
        config = cls(
            libero_dir=args.libero_dir,
            chunk_size=args.chunk_size,
            temporal_stride=args.temporal_stride,
            hidden_dim=args.hidden_dim,
            batch_size=args.batch_size,
            lr=args.lr,
            epochs=args.epochs,
            grad_clip=args.grad_clip,
            freeze_vision_epochs=args.freeze_vision_epochs,
            use_amp=not args.no_amp,
            augment=args.augment and not args.no_augment,
            num_workers=args.num_workers,
            device=args.device,
            seed=args.seed,
            output_dir=args.output_dir,
            save_every=args.save_every,
            log_interval=args.log_interval,
            wandb=args.wandb,
            wandb_project=args.wandb_project,
            wandb_name=args.wandb_name,
            wandb_tags=args.wandb_tags,
        )
        return config

    def __str__(self) -> str:
        """Pretty print config."""
        lines = ["TrainingConfig:"]
        for key, value in self.to_dict().items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)


def get_config(**overrides) -> TrainingConfig:
    """Get training config with optional overrides.

    Args:
        **overrides: Config values to override

    Returns:
        TrainingConfig instance
    """
    return TrainingConfig(**overrides)


# Preset configurations for different scenarios
CONFIGS = {
    "default": TrainingConfig(),

    "fast": TrainingConfig(
        batch_size=256,
        lr=4e-4,
        epochs=50,
        freeze_vision_epochs=3,
        num_workers=16,
    ),

    "debug": TrainingConfig(
        batch_size=32,
        lr=1e-4,
        epochs=5,
        freeze_vision_epochs=0,
        num_workers=4,
        use_amp=False,
        augment=False,
    ),

    "full": TrainingConfig(
        batch_size=128,
        lr=2e-4,
        epochs=200,
        freeze_vision_epochs=10,
        num_workers=16,
        wandb=True,
    ),
}


def get_preset(name: str) -> TrainingConfig:
    """Get a preset configuration by name.

    Args:
        name: One of "default", "fast", "debug", "full"

    Returns:
        TrainingConfig instance
    """
    if name not in CONFIGS:
        raise ValueError(f"Unknown preset: {name}. Available: {list(CONFIGS.keys())}")
    return CONFIGS[name]
