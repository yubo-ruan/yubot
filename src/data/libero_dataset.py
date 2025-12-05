"""LIBERO demonstration dataset for PickAndPlace skill training.

Loads HDF5 demonstration files from LIBERO-Spatial and LIBERO-Object
benchmarks for behavior cloning with flow matching.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class LIBERODataset(Dataset):
    """Dataset for loading LIBERO pick-and-place demonstrations.

    Loads action chunks from HDF5 demonstration files and pairs them
    with observations (images, proprioception, goal).

    Args:
        demo_paths: List of paths to HDF5 demo files
        chunk_size: Length of action chunks to sample
        img_size: Size to resize images to (H, W)
        camera_name: Which camera to use ('agentview_rgb' or 'eye_in_hand_rgb')
    """

    def __init__(
        self,
        demo_paths: List[str],
        chunk_size: int = 16,
        img_size: Tuple[int, int] = (128, 128),
        camera_name: str = "agentview_rgb",
    ):
        self.chunk_size = chunk_size
        self.img_size = img_size
        self.camera_name = camera_name

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
                num_demos = data_grp.attrs.get("num_demos", 0)

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

        # Get goal (final object position - use last frame's ee position as proxy)
        # In practice, we'd extract object position from task description
        # For now, use final gripper position when gripper first closes
        goal = self._extract_goal(ee_pos, gripper)

        # Sample action chunks
        # We can start a chunk at any timestep where there's enough future steps
        for t in range(T - self.chunk_size):
            sample = {
                "image": images[t],                          # (H, W, C)
                "ee_pos": ee_pos[t],                         # (3,)
                "ee_ori": ee_ori[t],                         # (3,) or (4,)
                "gripper": gripper[t],                       # (2,)
                "goal": goal,                                # (3,)
                "actions": actions[t:t + self.chunk_size],   # (chunk_size, 7)
            }
            self.samples.append(sample)

    def _extract_goal(
        self,
        ee_pos: np.ndarray,
        gripper: np.ndarray,
    ) -> np.ndarray:
        """Extract goal position from trajectory.

        For pick-and-place, the goal is where the object ends up.
        We approximate this as the gripper position at the last frame
        before gripper opens for release.

        Args:
            ee_pos: (T, 3) end-effector positions
            gripper: (T, 2) gripper states

        Returns:
            (3,) goal position
        """
        # Find when gripper opens at the end (release point)
        gripper_width = gripper.sum(axis=1)
        T = len(gripper_width)

        # Look for last closing→opening transition
        for t in range(T - 1, 0, -1):
            if gripper_width[t] > gripper_width[t - 1] + 0.01:
                # Found opening - goal is position just before
                return ee_pos[t - 1].copy()

        # Fallback: use final position
        return ee_pos[-1].copy()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Convert image: (H, W, C) -> (C, H, W), normalize to [0, 1]
        image = torch.from_numpy(sample["image"]).float()
        image = image.permute(2, 0, 1) / 255.0  # (C, H, W)

        # Proprioception
        ee_pos = torch.from_numpy(sample["ee_pos"]).float()
        ee_ori = torch.from_numpy(sample["ee_ori"]).float()
        gripper = torch.from_numpy(sample["gripper"]).float()

        # Concat proprioception
        proprio = torch.cat([ee_pos, ee_ori, gripper], dim=-1)

        # Goal
        goal = torch.from_numpy(sample["goal"]).float()

        # Actions
        actions = torch.from_numpy(sample["actions"]).float()

        return {
            "image": image,      # (C, H, W)
            "proprio": proprio,  # (proprio_dim,)
            "goal": goal,        # (3,)
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
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders for LIBERO demos.

    Args:
        libero_dir: Root directory of LIBERO datasets
        suites: List of suite names to include
        chunk_size: Action chunk size
        batch_size: Batch size for training
        val_split: Fraction of data for validation
        num_workers: Number of dataloader workers

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

    # Create datasets
    train_dataset = LIBERODataset(train_paths, chunk_size=chunk_size)
    val_dataset = LIBERODataset(val_paths, chunk_size=chunk_size)

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
