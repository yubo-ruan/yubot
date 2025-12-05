"""Weights & Biases logging utilities for training.

Usage:
    from training.wandb_logger import WandbLogger

    logger = WandbLogger(
        enabled=args.wandb,
        project="yubot-flow-policy",
        config=vars(args),
    )
    logger.watch(model)

    # In training loop
    logger.log({"train/loss": loss.item()})
    logger.log_image("samples/image", image_tensor)

    # End of training
    logger.finish()
"""

from typing import Any, Dict, Optional

import numpy as np
import torch

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class WandbLogger:
    """Wrapper for W&B logging with optional enable/disable.

    When disabled, all methods are no-ops for clean integration.
    """

    def __init__(
        self,
        enabled: bool = True,
        project: str = "yubot-flow-policy",
        config: Optional[Dict[str, Any]] = None,
        run_name: Optional[str] = None,
        tags: Optional[list] = None,
    ):
        """Initialize W&B logger.

        Args:
            enabled: Whether to enable logging
            project: W&B project name
            config: Hyperparameters dict (usually vars(args))
            run_name: Optional run name (auto-generated if None)
            tags: Optional list of tags for the run
        """
        self.enabled = enabled and HAS_WANDB

        if enabled and not HAS_WANDB:
            print("Warning: wandb not installed. Install with: pip install wandb")
            print("Continuing without W&B logging...")

        if self.enabled:
            wandb.init(
                project=project,
                config=config,
                name=run_name,
                tags=tags,
            )
            print(f"W&B logging enabled: {wandb.run.url}")

    def watch(
        self,
        model: torch.nn.Module,
        log: str = "gradients",
        log_freq: int = 100,
    ):
        """Watch model for gradient/parameter logging.

        Args:
            model: PyTorch model to watch
            log: What to log ("gradients", "parameters", "all", or None)
            log_freq: How often to log (in batches)
        """
        if self.enabled:
            wandb.watch(model, log=log, log_freq=log_freq)

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to W&B.

        Args:
            metrics: Dict of metric name -> value
            step: Optional global step (auto-incremented if None)
        """
        if self.enabled:
            wandb.log(metrics, step=step)

    def log_image(
        self,
        key: str,
        image: torch.Tensor,
        caption: Optional[str] = None,
    ):
        """Log an image to W&B.

        Args:
            key: Metric name for the image
            image: Image tensor (C, H, W) in [0, 1] range
            caption: Optional caption
        """
        if self.enabled:
            # Convert from (C, H, W) to (H, W, C) numpy
            if image.dim() == 3:
                img_np = image.cpu().permute(1, 2, 0).numpy()
            else:
                img_np = image.cpu().numpy()

            # Ensure [0, 255] range
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)

            wandb.log({key: wandb.Image(img_np, caption=caption)})

    def log_histogram(
        self,
        key: str,
        values: np.ndarray,
        num_bins: int = 64,
    ):
        """Log a histogram to W&B.

        Args:
            key: Metric name for the histogram
            values: Array of values
            num_bins: Number of bins
        """
        if self.enabled:
            wandb.log({key: wandb.Histogram(values, num_bins=num_bins)})

    def log_video(
        self,
        key: str,
        video_path: str,
        fps: int = 30,
        caption: Optional[str] = None,
    ):
        """Log a video file to W&B.

        Args:
            key: Metric name for the video
            video_path: Path to video file
            fps: Frames per second
            caption: Optional caption
        """
        if self.enabled:
            wandb.log({key: wandb.Video(video_path, fps=fps, caption=caption)})

    def log_artifact(
        self,
        name: str,
        artifact_type: str,
        file_path: str,
        metadata: Optional[Dict] = None,
    ):
        """Log a file as a W&B artifact.

        Args:
            name: Artifact name
            artifact_type: Type (e.g., "model", "dataset")
            file_path: Path to file
            metadata: Optional metadata dict
        """
        if self.enabled:
            artifact = wandb.Artifact(name, type=artifact_type, metadata=metadata)
            artifact.add_file(file_path)
            wandb.log_artifact(artifact)

    def set_summary(self, key: str, value: Any):
        """Set a summary metric (shown in run overview).

        Args:
            key: Metric name
            value: Metric value
        """
        if self.enabled:
            wandb.run.summary[key] = value

    def finish(self):
        """Finish the W&B run."""
        if self.enabled:
            wandb.finish()

    @property
    def run_url(self) -> Optional[str]:
        """Get the W&B run URL."""
        if self.enabled:
            return wandb.run.url
        return None

    @property
    def run_name(self) -> Optional[str]:
        """Get the W&B run name."""
        if self.enabled:
            return wandb.run.name
        return None
