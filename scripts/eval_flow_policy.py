#!/usr/bin/env python3
"""Evaluate flow matching policy on LIBERO tasks.

Usage:
    python scripts/eval_flow_policy.py --checkpoint checkpoints/pickandplace_best.pt
    python scripts/eval_flow_policy.py --checkpoint checkpoints/pickandplace_best.pt --task_suite libero_spatial --num_episodes 10
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.policy.flow_policy import PickAndPlaceFlowPolicy


def add_debug_overlay(
    frame: np.ndarray,
    step: int,
    action: np.ndarray,
    ee_pos: np.ndarray,
    goal: np.ndarray,
    chunk_idx: int,
    chunk_size: int,
    reward: float,
    task_description: str = "",
    success: bool = False,
) -> np.ndarray:
    """Add debug information overlay to frame with side panel (similar to debug_video.py style)."""
    if not HAS_CV2:
        return frame

    # Flip image vertically (robosuite uses OpenGL convention with origin at bottom-left)
    frame = np.flipud(frame).copy()

    # Scale up frame for better readability (128x128 -> 512x512)
    scale = 4
    frame = cv2.resize(frame, (frame.shape[1] * scale, frame.shape[0] * scale), interpolation=cv2.INTER_NEAREST)

    # Convert RGB to BGR for OpenCV
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Create overlay panel on the right (like debug_video.py)
    panel_width = 350
    panel = np.zeros((frame.shape[0], panel_width, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)  # Dark gray background

    # Text settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    color = (255, 255, 255)
    line_height = 20
    y_offset = 25

    # Task description
    if task_description:
        cv2.putText(panel, "TASK:", (10, y_offset), font, font_scale, (100, 200, 100), 1)
        y_offset += line_height
        # Word wrap task description
        words = task_description.split()
        line = ""
        for word in words:
            test_line = line + " " + word if line else word
            if len(test_line) * 8 < panel_width - 20:
                line = test_line
            else:
                cv2.putText(panel, line, (10, y_offset), font, font_scale * 0.9, color, 1)
                y_offset += line_height
                line = word
        if line:
            cv2.putText(panel, line, (10, y_offset), font, font_scale * 0.9, color, 1)
            y_offset += line_height
        y_offset += 5

    # Step and chunk info
    cv2.putText(panel, "POLICY STATE:", (10, y_offset), font, font_scale, (100, 200, 100), 1)
    y_offset += line_height
    cv2.putText(panel, f"  Step: {step}", (10, y_offset), font, font_scale * 0.9, color, 1)
    y_offset += line_height
    cv2.putText(panel, f"  Chunk: {chunk_idx + 1}/{chunk_size}", (10, y_offset), font, font_scale * 0.9, color, 1)
    y_offset += line_height + 5

    # Action info
    cv2.putText(panel, "ACTION (OSC):", (10, y_offset), font, font_scale, (100, 200, 100), 1)
    y_offset += line_height
    action_pos = action[:3]
    action_ori = action[3:6] if len(action) > 5 else np.zeros(3)
    gripper = action[6] if len(action) > 6 else 0
    gripper_str = "CLOSE" if gripper < 0 else "OPEN"

    cv2.putText(panel, f"  dPos: [{action_pos[0]:+.3f}, {action_pos[1]:+.3f}, {action_pos[2]:+.3f}]",
                (10, y_offset), font, font_scale * 0.85, (200, 200, 100), 1)
    y_offset += line_height
    cv2.putText(panel, f"  dOri: [{action_ori[0]:+.3f}, {action_ori[1]:+.3f}, {action_ori[2]:+.3f}]",
                (10, y_offset), font, font_scale * 0.85, (200, 200, 100), 1)
    y_offset += line_height
    cv2.putText(panel, f"  Gripper: {gripper_str} ({gripper:+.2f})",
                (10, y_offset), font, font_scale * 0.85, (200, 200, 100), 1)
    y_offset += line_height + 5

    # Robot state
    cv2.putText(panel, "ROBOT STATE:", (10, y_offset), font, font_scale, (100, 200, 100), 1)
    y_offset += line_height
    cv2.putText(panel, f"  EE Pos: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]",
                (10, y_offset), font, font_scale * 0.85, (100, 200, 200), 1)
    y_offset += line_height + 5

    # Goal info
    cv2.putText(panel, "GOAL:", (10, y_offset), font, font_scale, (100, 200, 100), 1)
    y_offset += line_height
    cv2.putText(panel, f"  Target: [{goal[0]:.3f}, {goal[1]:.3f}, {goal[2]:.3f}]",
                (10, y_offset), font, font_scale * 0.85, (255, 150, 100), 1)
    y_offset += line_height
    dist = np.linalg.norm(ee_pos - goal)
    dist_color = (100, 255, 100) if dist < 0.05 else (100, 200, 200)
    cv2.putText(panel, f"  Distance: {dist:.3f}m",
                (10, y_offset), font, font_scale * 0.85, dist_color, 1)
    y_offset += line_height + 5

    # Cumulative reward
    cv2.putText(panel, f"Cumulative Reward: {reward:.2f}",
                (10, y_offset), font, font_scale * 0.9, (150, 150, 255), 1)

    # Combine frame and panel
    combined = np.hstack([frame, panel])

    # Convert back to RGB
    combined = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)

    return combined


def make_libero_env(task_suite: str, task_id: int):
    """Create LIBERO environment."""
    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv

    benchmark = get_benchmark(task_suite)()
    task = benchmark.get_task(task_id)
    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file
    )

    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": 128,
        "camera_widths": 128,
    }

    env = OffScreenRenderEnv(**env_args)
    env.task_description = task.language
    env.task_name = task.name

    return env, task.language


def quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to axis-angle representation."""
    # robosuite uses (x, y, z, w) format
    x, y, z, w = quat

    # Compute angle
    angle = 2.0 * np.arccos(np.clip(w, -1.0, 1.0))

    # Compute axis
    sin_half = np.sin(angle / 2.0)
    if sin_half < 1e-6:
        return np.zeros(3)

    axis = np.array([x, y, z]) / sin_half
    return axis * angle


def get_obs_tensors(obs: dict, device: str):
    """Convert observation dict to tensors for policy."""
    # Image: (H, W, C) -> (1, C, H, W), normalize to [0, 1]
    image = torch.from_numpy(obs["agentview_image"]).float()
    image = image.permute(2, 0, 1).unsqueeze(0) / 255.0  # (1, C, H, W)

    # Proprioception: ee_pos (3) + ee_ori (3) + gripper (2)
    ee_pos = obs["robot0_eef_pos"]  # (3,)

    # Convert quat to axis-angle to match training data format
    ee_quat = obs["robot0_eef_quat"]  # (4,) in robosuite format
    ee_ori = quat_to_axis_angle(ee_quat)  # (3,)

    gripper = obs["robot0_gripper_qpos"]  # (2,)

    proprio = np.concatenate([ee_pos, ee_ori, gripper])
    proprio = torch.from_numpy(proprio).float().unsqueeze(0)  # (1, 8)

    return image.to(device), proprio.to(device)


def extract_goal_from_obs(obs: dict, task_description: str) -> np.ndarray:
    """Extract goal position from observation dict.

    For pick-and-place tasks, the goal is typically the target container position.
    LIBERO obs dict contains object positions directly.
    """
    task_lower = task_description.lower()

    # Common target objects in LIBERO - look for these in obs keys
    target_keywords = ['plate', 'tray', 'basket', 'container', 'bin']

    # Check which target keyword is mentioned in task
    target_type = None
    for keyword in target_keywords:
        if keyword in task_lower:
            target_type = keyword
            break

    # Search obs keys for matching object position
    if target_type:
        for key in obs.keys():
            if target_type in key.lower() and key.endswith('_pos'):
                pos = obs[key].copy()
                # Offset slightly above the container for placing
                pos[2] += 0.08
                return pos

    # Fallback: look for any plate position
    for key in obs.keys():
        if 'plate' in key.lower() and key.endswith('_pos'):
            pos = obs[key].copy()
            pos[2] += 0.08
            return pos

    # Last fallback: use a default position
    return np.array([0.0, 0.0, 0.95])


def run_episode(
    env,
    policy: PickAndPlaceFlowPolicy,
    goal: np.ndarray,
    device: str,
    max_steps: int = 300,
    action_repeat: int = 1,
    num_inference_steps: int = 10,
    debug_video: bool = True,
    task_description: str = "",
) -> dict:
    """Run a single evaluation episode.

    Args:
        env: LIBERO environment
        policy: Flow matching policy
        goal: (3,) target position
        device: torch device
        max_steps: Maximum environment steps
        action_repeat: How many times to repeat each action
        num_inference_steps: Euler integration steps for flow sampling
        debug_video: Whether to add debug overlay to frames

    Returns:
        Episode results dict
    """
    obs = env.reset()

    goal_tensor = torch.from_numpy(goal).float().unsqueeze(0).to(device)  # (1, 3)

    total_reward = 0.0
    success = False
    frames = []
    debug_frames = []
    actions_taken = []

    # Action chunk buffer
    action_chunk = None
    chunk_idx = 0
    chunk_size = policy.chunk_size

    for step in range(max_steps):
        # Get observation tensors
        image, proprio = get_obs_tensors(obs, device)

        # Sample new action chunk if needed
        if action_chunk is None or chunk_idx >= len(action_chunk):
            with torch.no_grad():
                action_chunk = policy.sample(
                    image, proprio, goal_tensor,
                    num_steps=num_inference_steps
                )  # (1, chunk_size, action_dim)
                action_chunk = action_chunk[0].cpu().numpy()  # (chunk_size, action_dim)
            chunk_idx = 0

        # Get current action from chunk
        action = action_chunk[chunk_idx]
        current_chunk_idx = chunk_idx
        chunk_idx += 1

        # Execute action (with optional repeat)
        for _ in range(action_repeat):
            obs, reward, done, info = env.step(action)
            total_reward += reward

            # Store raw frame
            raw_frame = obs["agentview_image"].copy()
            frames.append(raw_frame)
            actions_taken.append(action.copy())

            # Create debug frame with overlay
            if debug_video:
                ee_pos = obs["robot0_eef_pos"]
                debug_frame = add_debug_overlay(
                    frame=raw_frame,
                    step=step,
                    action=action,
                    ee_pos=ee_pos,
                    goal=goal,
                    chunk_idx=current_chunk_idx,
                    chunk_size=chunk_size,
                    reward=total_reward,
                    task_description=task_description,
                )
                debug_frames.append(debug_frame)

            if done:
                break

        # Check success
        if info.get("success", False) or done:
            success = info.get("success", False)
            break

    return {
        "success": success,
        "total_reward": total_reward,
        "steps": step + 1,
        "frames": frames,
        "debug_frames": debug_frames,
        "actions": actions_taken,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate flow policy on LIBERO")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to policy checkpoint",
    )
    parser.add_argument(
        "--task_suite",
        type=str,
        default="libero_spatial",
        help="LIBERO task suite",
    )
    parser.add_argument(
        "--task_id",
        type=int,
        default=0,
        help="Task ID within suite",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=300,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=10,
        help="Euler integration steps for flow sampling",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on",
    )
    parser.add_argument(
        "--save_videos",
        action="store_true",
        help="Save episode videos",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="logs/flow_eval",
        help="Output directory for results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)

    # Get config from checkpoint
    ckpt_args = checkpoint.get("args", {})
    chunk_size = ckpt_args.get("chunk_size", 16)

    # Create policy
    policy = PickAndPlaceFlowPolicy(
        action_dim=7,
        chunk_size=chunk_size,
        hidden_dim=256,
        proprio_dim=8,
        goal_dim=3,
        pretrained_vision=False,  # Will load from checkpoint
    ).to(args.device)

    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()
    print(f"Loaded policy from epoch {checkpoint.get('epoch', 'unknown')}")

    # Create environment
    print(f"Creating environment: {args.task_suite} task {args.task_id}")
    env, task_description = make_libero_env(args.task_suite, args.task_id)
    print(f"Task: {task_description}")

    # Get initial obs to extract goal
    init_obs = env.reset()

    # Extract goal position from obs
    goal = extract_goal_from_obs(init_obs, task_description)
    print(f"Goal position: {goal}")

    # Create output directory
    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"{args.task_suite}_task{args.task_id}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Run evaluation
    results = []
    successes = []

    print(f"\nRunning {args.num_episodes} episodes...")
    for ep in range(args.num_episodes):
        print(f"\n=== Episode {ep + 1}/{args.num_episodes} ===")

        result = run_episode(
            env=env,
            policy=policy,
            goal=goal,
            device=args.device,
            max_steps=args.max_steps,
            num_inference_steps=args.num_inference_steps,
            debug_video=args.save_videos,
            task_description=task_description,
        )

        successes.append(result["success"])
        print(f"Success: {result['success']}, Steps: {result['steps']}, Reward: {result['total_reward']:.2f}")

        # Save debug video with overlay (uses debug_frames with side panel)
        if args.save_videos and len(result["debug_frames"]) > 0:
            try:
                import imageio
                video_path = run_dir / f"episode_{ep:04d}_debug.mp4"
                imageio.mimsave(str(video_path), result["debug_frames"], fps=20)
                print(f"Saved debug video: {video_path}")
            except ImportError:
                print("imageio not installed, skipping video save")

        # Store result (without frames to save memory)
        results.append({
            "episode": ep,
            "success": result["success"],
            "steps": result["steps"],
            "total_reward": result["total_reward"],
        })

    # Summary
    success_rate = np.mean(successes)
    print(f"\n{'='*50}")
    print(f"RESULTS: {args.task_suite} task {args.task_id}")
    print(f"{'='*50}")
    print(f"Success rate: {success_rate:.1%} ({sum(successes)}/{len(successes)})")
    print(f"Avg steps: {np.mean([r['steps'] for r in results]):.1f}")
    print(f"Avg reward: {np.mean([r['total_reward'] for r in results]):.2f}")

    # Save results
    summary = {
        "task_suite": args.task_suite,
        "task_id": args.task_id,
        "task_description": task_description,
        "checkpoint": args.checkpoint,
        "num_episodes": args.num_episodes,
        "success_rate": success_rate,
        "results": results,
        "args": vars(args),
    }

    summary_path = run_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {run_dir}")

    env.close()


if __name__ == "__main__":
    main()
