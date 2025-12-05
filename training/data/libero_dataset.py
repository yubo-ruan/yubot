"""LIBERO demonstration dataset for PickAndPlace skill training.

Loads HDF5 demonstration files from LIBERO-Spatial and LIBERO-Object
benchmarks for behavior cloning with flow matching.

Uses oracle object positions extracted from demonstration trajectories.
Goal format: 12-dim = [pick_pos(3), pick_approach(3), place_pos(3), place_approach(3)]
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader


# Goal dimension: pick_pos(3) + place_pos(3)
GOAL_DIM = 6


def find_gripper_events(gripper: np.ndarray) -> Tuple[int, int]:
    """Find grasp (close) and release (open) timestamps from gripper states.

    Args:
        gripper: (T, 2) gripper finger positions

    Returns:
        (grasp_t, release_t) timestamps
    """
    gripper_width = gripper.sum(axis=1)
    T = len(gripper_width)

    # Find first significant closing (grasp)
    grasp_t = 0
    for t in range(1, T):
        if gripper_width[t] < gripper_width[t - 1] - 0.01:
            grasp_t = t
            break

    # Find last significant opening (release) after grasp
    release_t = T - 1
    for t in range(T - 1, grasp_t, -1):
        if gripper_width[t] > gripper_width[t - 1] + 0.01:
            release_t = t
            break

    return grasp_t, release_t


class LIBERODataset(Dataset):
    """Dataset for loading LIBERO pick-and-place demonstrations.

    Loads action chunks from HDF5 demonstration files and pairs them
    with observations (images, proprioception, goal).

    Goal format (6-dim):
        - pick_pos (3): Position where object was grasped
        - place_pos (3): Position where object was released

    Args:
        demo_paths: List of paths to HDF5 demo files
        chunk_size: Length of action chunks to sample
        img_size: Size to resize images to (H, W)
        camera_name: Which camera to use ('agentview_rgb' or 'eye_in_hand_rgb')
        place_height_offset: Height offset to add to place target position
    """

    def __init__(
        self,
        demo_paths: List[str],
        chunk_size: int = 16,
        img_size: Tuple[int, int] = (128, 128),
        camera_name: str = "agentview_rgb",
        place_height_offset: float = 0.02,
        temporal_stride: int = 1,
        augment: bool = False,
    ):
        self.chunk_size = chunk_size
        self.img_size = img_size
        self.camera_name = camera_name
        self.place_height_offset = place_height_offset
        self.temporal_stride = temporal_stride
        self.augment = augment

        # Data augmentation transforms (applied to normalized [0,1] images)
        if augment:
            self.transform = T.Compose([
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
                T.RandomAffine(degrees=3, translate=(0.03, 0.03), scale=(0.97, 1.03)),
            ])
        else:
            self.transform = None

        # Load all demos into memory
        self.samples = []
        self._load_demos(demo_paths)

    def _load_demos(self, demo_paths: List[str]):
        """Load all demonstrations from HDF5 files."""
        for path in demo_paths:
            if not os.path.exists(path):
                print(f"Warning: {path} not found, skipping")
                continue

            with h5py.File(path, "r") as f:
                data_grp = f["data"]

                # Iterate over demos
                for demo_key in data_grp.keys():
                    if not demo_key.startswith("demo_"):
                        continue

                    demo = data_grp[demo_key]
                    self._process_demo(demo, path)

    def _process_demo(self, demo: h5py.Group, path: str):
        """Extract samples from a single demonstration."""
        obs_grp = demo["obs"]

        # Load data
        images = obs_grp[self.camera_name][:]  # (T, H, W, C)
        actions = demo["actions"][:]           # (T, 7)

        # Proprioception: ee_pos + ee_ori (axis-angle or quat) + gripper
        ee_pos = obs_grp["ee_pos"][:]          # (T, 3)

        # Handle different orientation formats
        if "ee_ori" in obs_grp:
            ee_ori = obs_grp["ee_ori"][:]      # (T, 3) axis-angle or (T, 4) quat
        elif "ee_quat" in obs_grp:
            ee_ori = obs_grp["ee_quat"][:]     # (T, 4)
        else:
            # Use ee_states which contains full state
            ee_states = obs_grp["ee_states"][:]
            ee_ori = ee_states[:, 3:7]         # Assume quat is positions 3-7

        gripper = obs_grp["gripper_states"][:]  # (T, 2)

        T = len(images)

        # Extract 6-dim goal from trajectory
        goal = self._extract_goal_6d(ee_pos, gripper)

        # Sample action chunks with temporal stride
        # We can start a chunk at any timestep where there's enough future steps
        for t in range(0, T - self.chunk_size, self.temporal_stride):
            sample = {
                "image": images[t],                          # (H, W, C)
                "ee_pos": ee_pos[t],                         # (3,)
                "ee_ori": ee_ori[t],                         # (3,) or (4,)
                "gripper": gripper[t],                       # (2,)
                "goal": goal,                                # (6,)
                "actions": actions[t:t + self.chunk_size],   # (chunk_size, 7)
            }
            self.samples.append(sample)

    def _extract_goal_6d(
        self,
        ee_pos: np.ndarray,
        gripper: np.ndarray,
    ) -> np.ndarray:
        """Extract 6-dim goal from trajectory.

        The goal encodes pick and place positions:
        - pick_pos: EE position at grasp moment (where object is)
        - place_pos: EE position at release moment (where object goes)

        Args:
            ee_pos: (T, 3) end-effector positions
            gripper: (T, 2) gripper states

        Returns:
            (6,) goal vector
        """
        # Find grasp and release events
        grasp_t, release_t = find_gripper_events(gripper)

        # Get positions at grasp and release
        pick_pos = ee_pos[grasp_t].copy()
        place_pos = ee_pos[release_t].copy()

        # Add small height offset to place position (object settles slightly lower)
        place_pos[2] += self.place_height_offset

        # Concatenate into 6-dim goal
        goal = np.concatenate([
            pick_pos,   # (3,) where to pick
            place_pos,  # (3,) where to place
        ])

        return goal

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Convert image: (H, W, C) -> (C, H, W), normalize to [0, 1]
        image = torch.from_numpy(sample["image"]).float()
        image = image.permute(2, 0, 1) / 255.0  # (C, H, W)

        # Apply augmentation if enabled
        if self.transform is not None:
            image = self.transform(image)

        # Proprioception
        ee_pos = torch.from_numpy(sample["ee_pos"]).float()
        ee_ori = torch.from_numpy(sample["ee_ori"]).float()
        gripper = torch.from_numpy(sample["gripper"]).float()

        # Concat proprioception
        proprio = torch.cat([ee_pos, ee_ori, gripper], dim=-1)

        # Goal (6-dim)
        goal = torch.from_numpy(sample["goal"]).float()

        # Actions
        actions = torch.from_numpy(sample["actions"]).float()

        return {
            "image": image,      # (C, H, W)
            "proprio": proprio,  # (proprio_dim,)
            "goal": goal,        # (6,)
            "actions": actions,  # (chunk_size, 7)
        }


def find_libero_demos(
    libero_dir: str,
    suites: List[str] = ["libero_spatial", "libero_object"],
) -> List[str]:
    """Find all demo HDF5 files for given LIBERO suites.

    Args:
        libero_dir: Root directory of LIBERO datasets
        suites: List of suite names to include

    Returns:
        List of paths to HDF5 demo files
    """
    demo_paths = []
    libero_path = Path(libero_dir)

    for suite in suites:
        suite_dir = libero_path / suite
        if not suite_dir.exists():
            print(f"Warning: Suite directory {suite_dir} not found")
            continue

        # Find all HDF5 files
        for hdf5_file in suite_dir.glob("*.hdf5"):
            demo_paths.append(str(hdf5_file))

    print(f"Found {len(demo_paths)} demo files across {suites}")
    return demo_paths


def get_libero_dataloaders(
    libero_dir: str,
    suites: List[str] = ["libero_spatial", "libero_object"],
    chunk_size: int = 16,
    batch_size: int = 32,
    val_split: float = 0.1,
    num_workers: int = 4,
    temporal_stride: int = 1,
    augment: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders for LIBERO demos.

    Args:
        libero_dir: Root directory of LIBERO datasets
        suites: List of suite names to include
        chunk_size: Action chunk size
        batch_size: Batch size for training
        val_split: Fraction of data for validation
        num_workers: Number of dataloader workers
        temporal_stride: Sample every Nth timestep (reduces sample count, increases diversity)
        augment: Whether to apply data augmentation to training images

    Returns:
        (train_loader, val_loader) tuple
    """
    # Find demo files
    demo_paths = find_libero_demos(libero_dir, suites)

    if len(demo_paths) == 0:
        raise ValueError(f"No demo files found in {libero_dir} for suites {suites}")

    # Split into train/val
    np.random.shuffle(demo_paths)
    n_val = max(1, int(len(demo_paths) * val_split))
    val_paths = demo_paths[:n_val]
    train_paths = demo_paths[n_val:]

    print(f"Train: {len(train_paths)} files, Val: {len(val_paths)} files")

    # Create datasets (augmentation only for training)
    train_dataset = LIBERODataset(
        train_paths, chunk_size=chunk_size, temporal_stride=temporal_stride, augment=augment
    )
    val_dataset = LIBERODataset(
        val_paths, chunk_size=chunk_size, temporal_stride=temporal_stride, augment=False
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
