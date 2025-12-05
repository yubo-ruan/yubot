#!/usr/bin/env python3
"""Train PickAndPlace flow matching policy on LIBERO demos.

Usage:
    python scripts/train_pickandplace.py --libero_dir /path/to/libero/datasets

The script expects LIBERO demo files in:
    {libero_dir}/libero_spatial/*.hdf5
    {libero_dir}/libero_object/*.hdf5
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.policy.flow_policy import PickAndPlaceFlowPolicy, flow_matching_loss
from src.data.libero_dataset import get_libero_dataloaders


def train_epoch(
    policy: nn.Module,
    train_loader,
    optimizer,
    device: str,
) -> float:
    """Train for one epoch."""
    policy.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        # Move to device
        image = batch["image"].to(device)
        proprio = batch["proprio"].to(device)
        goal = batch["goal"].to(device)
        actions = batch["actions"].to(device)

        # Forward pass
        optimizer.zero_grad()
        loss = flow_matching_loss(policy, image, proprio, goal, actions)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / num_batches


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
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument("--chunk_size", type=int, default=16, help="Action chunk size")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--num_workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    print(f"Using device: {args.device}")

    # Create dataloaders
    print("Loading LIBERO demos...")
    train_loader, val_loader = get_libero_dataloaders(
        libero_dir=args.libero_dir,
        suites=["libero_spatial", "libero_object"],
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
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
        goal_dim=3,
        pretrained_vision=True,
    ).to(args.device)

    num_params = sum(p.numel() for p in policy.parameters())
    print(f"Policy parameters: {num_params:,}")

    # Optimizer and scheduler
    optimizer = AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Training loop
    best_val_loss = float("inf")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")

        # Train
        train_loss = train_epoch(policy, train_loader, optimizer, args.device)
        print(f"Train loss: {train_loss:.4f}")

        # Validate
        val_loss = validate(policy, val_loader, args.device)
        print(f"Val loss: {val_loss:.4f}")

        # Step scheduler
        scheduler.step()
        print(f"LR: {scheduler.get_last_lr()[0]:.6f}")

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
        torch.save(checkpoint, output_dir / f"pickandplace_latest.pt")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, output_dir / f"pickandplace_best.pt")
            print(f"New best model! Val loss: {val_loss:.4f}")

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, output_dir / f"pickandplace_epoch{epoch + 1}.pt")

    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    main()
