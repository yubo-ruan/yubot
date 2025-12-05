#!/usr/bin/env python3
"""Train PickAndPlace flow matching policy on LIBERO demos.

Usage:
    python training/train_pickandplace.py --libero_dir /path/to/libero/datasets

    # With W&B logging:
    python training/train_pickandplace.py --libero_dir /path/to/libero/datasets --wandb

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

            # Log sample image every 200 steps
            if batch_idx % (log_interval * 4) == 0:
                logger.log_image("samples/input_image", image[0])
                logger.log_histogram("samples/actions", actions.cpu().numpy().flatten())
                logger.log_histogram("samples/goal", goal.cpu().numpy().flatten())

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
    parser = argparse.ArgumentParser(description="Train PickAndPlace flow policy")
    parser.add_argument(
        "--libero_dir",
        type=str,
        required=True,
        help="Path to LIBERO datasets directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,  # Will default to training/checkpoints
        help="Directory to save checkpoints (default: training/checkpoints)",
    )
    parser.add_argument("--chunk_size", type=int, default=16, help="Action chunk size")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--num_workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision training")
    parser.add_argument("--freeze_vision_epochs", type=int, default=0, help="Freeze vision encoder for first N epochs")
    parser.add_argument("--temporal_stride", type=int, default=1, help="Sample every Nth timestep")
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation")

    # W&B arguments
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb_project", type=str, default="yubot-flow-policy", help="W&B project name")
    parser.add_argument("--wandb_name", type=str, default=None, help="W&B run name")
    parser.add_argument("--wandb_tags", type=str, nargs="*", default=None, help="W&B tags")

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory (default to training/checkpoints)
    if args.output_dir is None:
        output_dir = TRAINING_DIR / "checkpoints"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    print(f"Using device: {args.device}")

    # Initialize W&B logger
    logger = WandbLogger(
        enabled=args.wandb,
        project=args.wandb_project,
        config=vars(args),
        run_name=args.wandb_name,
        tags=args.wandb_tags,
    )

    # Create dataloaders
    print("Loading LIBERO demos...")
    train_loader, val_loader = get_libero_dataloaders(
        libero_dir=args.libero_dir,
        suites=["libero_spatial", "libero_object"],
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        temporal_stride=args.temporal_stride,
        augment=args.augment,
    )

    # Get proprio_dim from first batch
    sample_batch = next(iter(train_loader))
    proprio_dim = sample_batch["proprio"].shape[-1]
    print(f"Proprio dim: {proprio_dim}")

    # Create policy
    policy = PickAndPlaceFlowPolicy(
        action_dim=7,
        chunk_size=args.chunk_size,
        hidden_dim=256,
        proprio_dim=proprio_dim,
        goal_dim=6,
        pretrained_vision=True,
    ).to(args.device)

    num_params = sum(p.numel() for p in policy.parameters())
    print(f"Policy parameters: {num_params:,}")

    # Watch model with W&B (logs gradients)
    logger.watch(policy, log="gradients", log_freq=100)

    # Optimizer and scheduler
    optimizer = AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Mixed precision scaler
    use_amp = not args.no_amp and args.device == "cuda"
    scaler = GradScaler() if use_amp else None
    if use_amp:
        print("Using mixed precision training (AMP)")

    # Frozen encoder setup
    vision_frozen = False
    if args.freeze_vision_epochs > 0:
        freeze_vision_encoder(policy)
        vision_frozen = True
        trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        print(f"Vision encoder frozen for first {args.freeze_vision_epochs} epochs")
        print(f"Trainable parameters: {trainable_params:,} (was {num_params:,})")

    # Training loop
    best_val_loss = float("inf")
    global_step = 0

    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Checkpoints: {output_dir}")

    for epoch in range(args.epochs):
        # Unfreeze vision encoder after N epochs
        if vision_frozen and epoch >= args.freeze_vision_epochs:
            unfreeze_vision_encoder(policy)
            vision_frozen = False
            print(f"Vision encoder unfrozen at epoch {epoch + 1}")

        print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")

        # Train
        train_loss, global_step = train_epoch(
            policy, train_loader, optimizer, args.device,
            logger, epoch + 1, global_step, scaler, use_amp
        )
        print(f"Train loss: {train_loss:.4f}")

        # Validate
        val_loss = validate(policy, val_loader, args.device)
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
            "args": vars(args),
        }

        # Save latest
        torch.save(checkpoint, output_dir / "pickandplace_latest.pt")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, output_dir / "pickandplace_best.pt")
            print(f"New best model! Val loss: {val_loss:.4f}")
            logger.set_summary("best_val_loss", best_val_loss)
            logger.set_summary("best_epoch", epoch + 1)

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, output_dir / f"pickandplace_epoch{epoch + 1}.pt")

    # Finish W&B run
    logger.set_summary("final_train_loss", train_loss)
    logger.set_summary("final_val_loss", val_loss)
    logger.finish()

    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    main()
