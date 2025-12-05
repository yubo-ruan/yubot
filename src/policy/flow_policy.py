"""Flow matching policy for PickAndPlace skill.

Uses conditional flow matching to learn action chunk prediction
from LIBERO demonstrations.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for time conditioning."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ImageEncoder(nn.Module):
    """ResNet-based image encoder."""

    def __init__(self, output_dim: int = 256, pretrained: bool = True):
        super().__init__()
        resnet = models.resnet18(pretrained=pretrained)
        # Remove final FC layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(512, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor, normalized to [0, 1]
        Returns:
            (B, output_dim) feature vector
        """
        # Normalize to ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        feat = self.backbone(x)  # (B, 512, 1, 1)
        feat = feat.flatten(1)   # (B, 512)
        return self.fc(feat)     # (B, output_dim)


class PickAndPlaceFlowPolicy(nn.Module):
    """Flow matching policy for PickAndPlace skill.

    Architecture:
    - Image encoder (ResNet18)
    - Proprioception encoder (MLP)
    - Goal encoder (target position)
    - Time embedding (sinusoidal)
    - Flow network (MLP predicting velocity)

    Predicts action chunks of size `chunk_size`.
    """

    def __init__(
        self,
        action_dim: int = 7,
        chunk_size: int = 16,
        hidden_dim: int = 256,
        proprio_dim: int = 8,   # ee_pos(3) + ee_ori(3) + gripper(2)
        goal_dim: int = 6,      # pick_pos(3) + place_pos(3)
        pretrained_vision: bool = True,
        dropout: float = 0.0,   # Dropout rate for regularization
    ):
        super().__init__()

        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.hidden_dim = hidden_dim

        # Image encoder
        self.img_encoder = ImageEncoder(hidden_dim, pretrained=pretrained_vision)

        # Proprioception encoder
        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprio_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, hidden_dim),
        )

        # Goal encoder (target position)
        self.goal_encoder = nn.Sequential(
            nn.Linear(goal_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, hidden_dim),
        )

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(64),
            nn.Linear(64, hidden_dim),
            nn.SiLU(),
        )

        # Fusion layer
        # Combines: img_feat + proprio_feat + goal_feat + time_feat
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
        )

        # Flow network: predicts velocity field
        # Input: fused features + flattened noisy actions
        flow_input_dim = hidden_dim + action_dim * chunk_size
        self.flow_net = nn.Sequential(
            nn.Linear(flow_input_dim, 512),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(512, action_dim * chunk_size),
        )

    def encode_obs(
        self,
        image: torch.Tensor,
        proprio: torch.Tensor,
        goal: torch.Tensor,
    ) -> torch.Tensor:
        """Encode observations into fused feature.

        Args:
            image: (B, C, H, W) RGB image
            proprio: (B, proprio_dim) proprioception
            goal: (B, goal_dim) target position

        Returns:
            (B, hidden_dim) fused feature
        """
        img_feat = self.img_encoder(image)        # (B, hidden)
        proprio_feat = self.proprio_encoder(proprio)  # (B, hidden)
        goal_feat = self.goal_encoder(goal)       # (B, hidden)

        return img_feat, proprio_feat, goal_feat

    def forward(
        self,
        image: torch.Tensor,
        proprio: torch.Tensor,
        goal: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Predict velocity field at x_t, time t.

        Args:
            image: (B, C, H, W) RGB image
            proprio: (B, proprio_dim) proprioception
            goal: (B, goal_dim) target position
            x_t: (B, chunk_size, action_dim) noisy action chunk
            t: (B,) time values in [0, 1]

        Returns:
            (B, chunk_size, action_dim) predicted velocity
        """
        B = x_t.shape[0]

        # Encode observations
        img_feat, proprio_feat, goal_feat = self.encode_obs(image, proprio, goal)

        # Time embedding
        time_feat = self.time_mlp(t)  # (B, hidden)

        # Fuse all features
        fused = self.fuse(torch.cat([img_feat, proprio_feat, goal_feat, time_feat], dim=-1))

        # Flatten noisy actions
        x_flat = x_t.view(B, -1)  # (B, chunk_size * action_dim)

        # Predict velocity
        flow_input = torch.cat([fused, x_flat], dim=-1)
        v = self.flow_net(flow_input)
        v = v.view(B, self.chunk_size, self.action_dim)

        return v

    @torch.no_grad()
    def sample(
        self,
        image: torch.Tensor,
        proprio: torch.Tensor,
        goal: torch.Tensor,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """Sample action chunk via Euler integration.

        Args:
            image: (B, C, H, W) RGB image
            proprio: (B, proprio_dim) proprioception
            goal: (B, goal_dim) target position
            num_steps: Number of Euler integration steps

        Returns:
            (B, chunk_size, action_dim) sampled action chunk
        """
        B = image.shape[0]
        device = image.device

        # Start from noise
        x = torch.randn(B, self.chunk_size, self.action_dim, device=device)

        # Euler integration from t=0 to t=1
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((B,), i * dt, device=device)
            v = self.forward(image, proprio, goal, x, t)
            x = x + v * dt

        # Clip actions to valid range
        x = torch.clamp(x, -1.0, 1.0)

        return x


def flow_matching_loss(
    policy: PickAndPlaceFlowPolicy,
    image: torch.Tensor,
    proprio: torch.Tensor,
    goal: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    """Compute flow matching loss.

    Args:
        policy: Flow policy network
        image: (B, C, H, W) RGB image
        proprio: (B, proprio_dim) proprioception
        goal: (B, goal_dim) target position
        actions: (B, chunk_size, action_dim) ground truth action chunk

    Returns:
        Scalar loss value
    """
    B = actions.shape[0]
    device = actions.device

    # Sample random time t ~ U(0, 1)
    t = torch.rand(B, device=device)

    # x_0 = noise, x_1 = ground truth actions
    x_0 = torch.randn_like(actions)
    x_1 = actions

    # Linear interpolation: x_t = (1-t)*x_0 + t*x_1
    t_ = t.view(B, 1, 1)  # (B, 1, 1) for broadcasting
    x_t = (1 - t_) * x_0 + t_ * x_1

    # Target velocity: dx/dt = x_1 - x_0 (for linear interpolation)
    v_target = x_1 - x_0

    # Predict velocity
    v_pred = policy(image, proprio, goal, x_t, t)

    # MSE loss on velocity
    loss = F.mse_loss(v_pred, v_target)

    return loss
