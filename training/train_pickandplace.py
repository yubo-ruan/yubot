#!/usr/bin/env python3
"""Train PickAndPlace flow matching policy on LIBERO demos.

Usage:
    # Run with default config (optimized for RTX 4090):
    python training/train_pickandplace.py

    # Override specific parameters:
    python training/train_pickandplace.py --batch_size 128 --lr 2e-4

    # Use a preset config:
    python training/train_pickandplace.py --preset fast

    # With W&B logging:
    python training/train_pickandplace.py --wandb

The script expects LIBERO demo files in:
    {libero_dir}/libero_spatial/*.hdf5
    {libero_dir}/libero_object/*.hdf5

Outputs:
    - training/checkpoints/: Model checkpoints
    - W&B dashboard (if --wandb enabled)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# Get the training directory (where this script lives)
TRAINING_DIR = Path(__file__).parent.resolve()
YUBOT_DIR = TRAINING_DIR.parent

# Add src to path
sys.path.insert(0, str(YUBOT_DIR))

from src.policy.flow_policy import PickAndPlaceFlowPolicy, flow_matching_loss
from training.data.libero_dataset import get_libero_dataloaders
from training.wandb_logger import WandbLogger
from training.config import TrainingConfig, get_preset, CONFIGS


def freeze_vision_encoder(policy: nn.Module):
    """Freeze vision encoder parameters."""
    for param in policy.img_encoder.parameters():
        param.requires_grad = False


def unfreeze_vision_encoder(policy: nn.Module):
    """Unfreeze vision encoder parameters."""
    for param in policy.img_encoder.parameters():
        param.requires_grad = True


def train_epoch(
    policy: nn.Module,
    train_loader,
    optimizer,
    device: str,
    logger: WandbLogger,
    epoch: int,
    global_step: int,
    scaler: GradScaler = None,
    use_amp: bool = True,
    log_interval: int = 50,
) -> tuple:
    """Train for one epoch with optional mixed precision.

    Returns:
        (avg_loss, global_step)
    """
    policy.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        image = batch["image"].to(device)
        proprio = batch["proprio"].to(device)
        goal = batch["goal"].to(device)
        actions = batch["actions"].to(device)

        optimizer.zero_grad()

        if use_amp and scaler is not None:
            # Mixed precision forward pass
            with autocast():
                loss = flow_matching_loss(policy, image, proprio, goal, actions)
            # Scaled backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard forward/backward
            loss = flow_matching_loss(policy, image, proprio, goal, actions)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()

        loss_val = loss.item()
        total_loss += loss_val
        num_batches += 1
        global_step += 1
        pbar.set_postfix({"loss": f"{loss_val:.4f}"})

        # Log to W&B
        if batch_idx % log_interval == 0:
            logger.log({
                "train/loss": loss_val,
                "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                "train/epoch": epoch,
                "train/step": global_step,
            })

            # Sample logging disabled for training speed
            # if batch_idx % (log_interval * 4) == 0:
            #     logger.log_image("samples/input_image", image[0])
            #     logger.log_histogram("samples/actions", actions.cpu().numpy().flatten())
            #     logger.log_histogram("samples/goal", goal.cpu().numpy().flatten())

    return total_loss / num_batches, global_step


@torch.no_grad()
def validate(
    policy: nn.Module,
    val_loader,
    device: str,
) -> float:
    """Validate on held-out data."""
    policy.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(val_loader, desc="Validating"):
        image = batch["image"].to(device)
        proprio = batch["proprio"].to(device)
        goal = batch["goal"].to(device)
        actions = batch["actions"].to(device)

        loss = flow_matching_loss(policy, image, proprio, goal, actions)
        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def main():
    # Build parser with config defaults
    parser = argparse.ArgumentParser(
        description="Train PickAndPlace flow policy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Add preset option
    parser.add_argument(
        "--preset",
        type=str,
        choices=list(CONFIGS.keys()),
        default=None,
        help="Use a preset configuration (default, fast, debug, full)",
    )

    # Add resume option
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from latest checkpoint",
    )

    # Add all config arguments with defaults
    TrainingConfig.add_argparse_args(parser)

    args = parser.parse_args()

    # Load config: start with preset or default, then apply CLI overrides
    if args.preset:
        config = get_preset(args.preset)
        print(f"Using preset config: {args.preset}")
    else:
        config = TrainingConfig.from_argparse(args)

    # Print config
    print("\n" + "=" * 50)
    print(config)
    print("=" * 50 + "\n")

    # Set seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check device
    device = config.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    print(f"Using device: {device}")

    # Initialize W&B logger
    logger = WandbLogger(
        enabled=config.wandb,
        project=config.wandb_project,
        config=config.to_dict(),
        run_name=config.wandb_name,
        tags=config.wandb_tags,
    )

    # Create dataloaders
    print("Loading LIBERO demos...")
    train_loader, val_loader = get_libero_dataloaders(
        libero_dir=config.libero_dir,
        suites=config.suites,
        chunk_size=config.chunk_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        temporal_stride=config.temporal_stride,
        augment=config.augment,
    )

    # Get proprio_dim from first batch
    sample_batch = next(iter(train_loader))
    proprio_dim = sample_batch["proprio"].shape[-1]
    print(f"Proprio dim: {proprio_dim}")

    # Create policy
    policy = PickAndPlaceFlowPolicy(
        action_dim=config.action_dim,
        chunk_size=config.chunk_size,
        hidden_dim=config.hidden_dim,
        proprio_dim=proprio_dim,
        goal_dim=config.goal_dim,
        pretrained_vision=config.pretrained_vision,
        dropout=config.dropout,
    ).to(device)

    num_params = sum(p.numel() for p in policy.parameters())
    print(f"Policy parameters: {num_params:,}")

    # Watch model with W&B (disabled gradient logging for speed)
    # logger.watch(policy, log="gradients", log_freq=100)

    # Optimizer and scheduler
    optimizer = AdamW(policy.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)

    # Mixed precision scaler
    use_amp = config.use_amp and device == "cuda"
    scaler = GradScaler() if use_amp else None
    if use_amp:
        print("Using mixed precision training (AMP)")

    # Frozen encoder setup
    vision_frozen = False
    if config.freeze_vision_epochs > 0:
        freeze_vision_encoder(policy)
        vision_frozen = True
        trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        print(f"Vision encoder frozen for first {config.freeze_vision_epochs} epochs")
        print(f"Trainable parameters: {trainable_params:,} (was {num_params:,})")

    # Training loop
    best_val_loss = float("inf")
    global_step = 0
    start_epoch = 0
    epochs_without_improvement = 0

    # Resume from checkpoint if requested
    checkpoint_path = output_dir / "pickandplace_latest.pt"
    if args.resume and checkpoint_path.exists():
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        policy.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint.get("val_loss", float("inf"))
        # Advance scheduler to correct position
        for _ in range(start_epoch):
            scheduler.step()
        print(f"Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.4f}")

    print(f"\nStarting training for {config.epochs} epochs...")
    print(f"Checkpoints: {output_dir}")
    if config.early_stopping:
        print(f"Early stopping: patience={config.early_stopping_patience}, min_delta={config.early_stopping_min_delta}")

    for epoch in range(start_epoch, config.epochs):
        # Unfreeze vision encoder after N epochs
        if vision_frozen and epoch >= config.freeze_vision_epochs:
            unfreeze_vision_encoder(policy)
            vision_frozen = False
            print(f"Vision encoder unfrozen at epoch {epoch + 1}")

        print(f"\n=== Epoch {epoch + 1}/{config.epochs} ===")

        # Train
        train_loss, global_step = train_epoch(
            policy, train_loader, optimizer, device,
            logger, epoch + 1, global_step, scaler, use_amp,
            log_interval=config.log_interval,
        )
        print(f"Train loss: {train_loss:.4f}")

        # Validate
        val_loss = validate(policy, val_loader, device)
        print(f"Val loss: {val_loss:.4f}")

        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"LR: {current_lr:.6f}")

        # Log epoch metrics to W&B
        logger.log({
            "epoch/train_loss": train_loss,
            "epoch/val_loss": val_loss,
            "epoch/lr": current_lr,
            "epoch/epoch": epoch + 1,
        })

        # Save checkpoint
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "config": config.to_dict(),
        }

        # Save latest
        torch.save(checkpoint, output_dir / "pickandplace_latest.pt")

        # Save best and check early stopping
        if val_loss < best_val_loss - config.early_stopping_min_delta:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(checkpoint, output_dir / "pickandplace_best.pt")
            print(f"New best model! Val loss: {val_loss:.4f}")
            logger.set_summary("best_val_loss", best_val_loss)
            logger.set_summary("best_epoch", epoch + 1)
        else:
            epochs_without_improvement += 1
            if config.early_stopping:
                print(f"No improvement for {epochs_without_improvement}/{config.early_stopping_patience} epochs")

        # Early stopping check
        if config.early_stopping and epochs_without_improvement >= config.early_stopping_patience:
            print(f"\nEarly stopping triggered! No improvement for {config.early_stopping_patience} epochs.")
            break

        # Save periodic checkpoint
        if (epoch + 1) % config.save_every == 0:
            torch.save(checkpoint, output_dir / f"pickandplace_epoch{epoch + 1}.pt")

    # Finish W&B run
    logger.set_summary("final_train_loss", train_loss)
    logger.set_summary("final_val_loss", val_loss)
    logger.finish()

    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    main()
