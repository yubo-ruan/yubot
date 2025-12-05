#!/usr/bin/env python3
"""Evaluate flow matching policy on LIBERO tasks.

Usage:
    python scripts/eval_flow_policy.py --task_id 0
    python scripts/eval_flow_policy.py --task_ids 4 5 8 9 --task_suite libero_spatial
    python scripts/eval_flow_policy.py --task_ids 0 1 2 --num_episodes 5
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
from src.grounding.semantic_grounder import QwenSemanticGrounder
from src.grounding.enriched_object import EnrichedObject, parse_object_description
from typing import List


# Goal dimension: pick_pos(3) + place_pos(3)
GOAL_DIM = 6


def build_objects_from_obs(obs: dict) -> List[EnrichedObject]:
    """Build EnrichedObject list from observation keys.

    Extracts object IDs from observation dict keys ending in '_pos'.
    Uses parse_object_description to get human-readable descriptions.
    """
    objects = []
    seen_ids = set()

    for key in obs.keys():
        if key.endswith("_pos") and not key.startswith("robot"):
            obj_id = key[:-4]  # Remove "_pos"
            if obj_id in seen_ids:
                continue
            seen_ids.add(obj_id)

            # Get human-readable description
            description = parse_object_description(obj_id)

            # Extract type from description (first noun)
            parts = description.split()
            obj_type = parts[-1] if parts else obj_id

            # Get position
            pos = obs[key]

            objects.append(EnrichedObject(
                id=obj_id,
                description=description,
                type=obj_type,
                position=tuple(pos),
            ))

    return objects


def find_object_position(obs: dict, object_name: str) -> np.ndarray:
    """Find object position from observation dict.

    LIBERO observations contain object positions with keys like:
    - 'alphabet_soup_pos', 'plate_pos', etc.
    """
    # Direct match
    key = f"{object_name}_pos"
    if key in obs:
        return obs[key].copy()

    # Try with common prefixes/suffixes
    for obs_key in obs.keys():
        if object_name in obs_key.lower() and obs_key.endswith("_pos"):
            return obs[obs_key].copy()

    # Fallback: search for partial match
    object_words = object_name.replace("_", " ").split()
    for obs_key in obs.keys():
        if obs_key.endswith("_pos"):
            key_lower = obs_key.lower()
            if all(word in key_lower for word in object_words):
                return obs[obs_key].copy()

    # Return current EE position as fallback
    if "robot0_eef_pos" in obs:
        return obs["robot0_eef_pos"].copy()

    return np.array([0.0, 0.0, 0.9])


def extract_goal_6d(
    obs: dict,
    task_description: str,
    grounder: QwenSemanticGrounder,
) -> tuple:
    """Extract 6-dim goal from observation and task description using VLM grounding.

    Goal format:
        - pick_pos (3): Position of object to pick
        - place_pos (3): Position of place target

    Args:
        obs: LIBERO observation dict containing object positions
        task_description: Natural language task description
        grounder: VLM grounder for semantic grounding

    Returns:
        (goal, pick_obj_name, place_target_name) tuple
    """
    # Use VLM grounding
    objects = build_objects_from_obs(obs)
    result = grounder.ground(task_description, objects)

    if not result.valid:
        raise ValueError(f"VLM grounding failed: {result.error}")

    pick_obj_name = result.source_object
    place_target_name = result.target_location
    print(f"  VLM grounding: pick={pick_obj_name}, place={place_target_name}")

    # Get positions from observations
    pick_pos = find_object_position(obs, pick_obj_name)
    place_pos = find_object_position(obs, place_target_name)

    # Add height offset for placement (place slightly above target)
    place_pos[2] += 0.05

    # Concatenate into 6-dim goal
    goal = np.concatenate([
        pick_pos,   # (3,) where to pick
        place_pos,  # (3,) where to place
    ])

    return goal, pick_obj_name, place_target_name


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
    pick_obj_name: str = "",
    place_target_name: str = "",
) -> np.ndarray:
    """Add debug information overlay to frame with side panel."""
    if not HAS_CV2:
        return frame

    # Flip image vertically (robosuite uses OpenGL convention with origin at bottom-left)
    frame = np.flipud(frame).copy()

    # Scale up frame for better readability (128x128 -> 512x512)
    scale = 4
    frame = cv2.resize(frame, (frame.shape[1] * scale, frame.shape[0] * scale), interpolation=cv2.INTER_NEAREST)

    # Convert RGB to BGR for OpenCV
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Create overlay panel on the right
    panel_width = 380
    panel = np.zeros((frame.shape[0], panel_width, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)  # Dark gray background

    # Text settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.42
    color = (255, 255, 255)
    line_height = 18
    y_offset = 20

    # Task description
    if task_description:
        cv2.putText(panel, "TASK:", (10, y_offset), font, font_scale, (100, 200, 100), 1)
        y_offset += line_height
        # Word wrap task description
        words = task_description.split()
        line = ""
        for word in words:
            test_line = line + " " + word if line else word
            if len(test_line) * 7 < panel_width - 20:
                line = test_line
            else:
                cv2.putText(panel, line, (10, y_offset), font, font_scale * 0.9, color, 1)
                y_offset += line_height
                line = word
        if line:
            cv2.putText(panel, line, (10, y_offset), font, font_scale * 0.9, color, 1)
            y_offset += line_height
        y_offset += 5

    # VLM Grounding info
    if pick_obj_name or place_target_name:
        cv2.putText(panel, "VLM GROUNDING:", (10, y_offset), font, font_scale, (100, 200, 100), 1)
        y_offset += line_height
        if pick_obj_name:
            cv2.putText(panel, f"  Pick: {pick_obj_name}", (10, y_offset), font, font_scale * 0.9, (255, 150, 100), 1)
            y_offset += line_height
        if place_target_name:
            cv2.putText(panel, f"  Place: {place_target_name}", (10, y_offset), font, font_scale * 0.9, (100, 150, 255), 1)
            y_offset += line_height
        y_offset += 5

    # Step and chunk info
    cv2.putText(panel, "POLICY STATE:", (10, y_offset), font, font_scale, (100, 200, 100), 1)
    y_offset += line_height
    cv2.putText(panel, f"  Step: {step}", (10, y_offset), font, font_scale * 0.9, color, 1)
    y_offset += line_height
    cv2.putText(panel, f"  Chunk: {chunk_idx + 1}/{chunk_size}", (10, y_offset), font, font_scale * 0.9, color, 1)
    y_offset += line_height + 3

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
    y_offset += line_height + 3

    # Robot state
    cv2.putText(panel, "ROBOT STATE:", (10, y_offset), font, font_scale, (100, 200, 100), 1)
    y_offset += line_height
    cv2.putText(panel, f"  EE Pos: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]",
                (10, y_offset), font, font_scale * 0.85, (100, 200, 200), 1)
    y_offset += line_height + 3

    # Goal info (6-dim: pick_pos + place_pos)
    cv2.putText(panel, "GOAL (6-dim):", (10, y_offset), font, font_scale, (100, 200, 100), 1)
    y_offset += line_height

    pick_pos = goal[:3]
    place_pos = goal[3:6]

    # Pick target
    cv2.putText(panel, f"  Pick:  [{pick_pos[0]:.3f}, {pick_pos[1]:.3f}, {pick_pos[2]:.3f}]",
                (10, y_offset), font, font_scale * 0.85, (255, 150, 100), 1)
    y_offset += line_height
    dist_pick = np.linalg.norm(ee_pos - pick_pos)
    dist_color = (100, 255, 100) if dist_pick < 0.05 else (200, 200, 200)
    cv2.putText(panel, f"    dist: {dist_pick:.3f}m",
                (10, y_offset), font, font_scale * 0.8, dist_color, 1)
    y_offset += line_height

    # Place target
    cv2.putText(panel, f"  Place: [{place_pos[0]:.3f}, {place_pos[1]:.3f}, {place_pos[2]:.3f}]",
                (10, y_offset), font, font_scale * 0.85, (100, 150, 255), 1)
    y_offset += line_height
    dist_place = np.linalg.norm(ee_pos - place_pos)
    dist_color = (100, 255, 100) if dist_place < 0.05 else (200, 200, 200)
    cv2.putText(panel, f"    dist: {dist_place:.3f}m",
                (10, y_offset), font, font_scale * 0.8, dist_color, 1)
    y_offset += line_height + 3

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
    pick_obj_name: str = "",
    place_target_name: str = "",
) -> dict:
    """Run a single evaluation episode.

    Args:
        env: LIBERO environment
        policy: Flow matching policy
        goal: (6,) goal vector [pick_pos, place_pos]
        device: torch device
        max_steps: Maximum environment steps
        action_repeat: How many times to repeat each action
        num_inference_steps: Euler integration steps for flow sampling
        debug_video: Whether to add debug overlay to frames

    Returns:
        Episode results dict
    """
    obs = env.reset()

    goal_tensor = torch.from_numpy(goal).float().unsqueeze(0).to(device)  # (1, 6)

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
                    pick_obj_name=pick_obj_name,
                    place_target_name=place_target_name,
                )
                debug_frames.append(debug_frame)

            if done:
                break

        # Check success - in LIBERO, done = _check_success(), so done implies success
        # Only break on actual success, not on other termination conditions
        if done:
            success = done  # LIBERO sets done = success
            break

    return {
        "success": success,
        "total_reward": total_reward,
        "steps": step + 1,
        "frames": frames,
        "debug_frames": debug_frames,
        "actions": actions_taken,
    }


def run_single_task(task_args: dict) -> dict:
    """Run evaluation for a single task.

    Supports two modes:
    1. Pre-computed grounding: If 'grounding_result' is provided, uses it directly (for parallel execution)
    2. Live grounding: If 'grounder' is provided, uses it for VLM grounding (for sequential execution)
    """
    task_id = task_args["task_id"]
    task_suite = task_args["task_suite"]
    checkpoint = task_args["checkpoint"]
    num_episodes = task_args["num_episodes"]
    max_steps = task_args["max_steps"]
    num_inference_steps = task_args["num_inference_steps"]
    device = task_args["device"]
    save_videos = task_args["save_videos"]
    output_dir = task_args["output_dir"]
    seed = task_args["seed"]
    grounder = task_args.get("grounder", None)  # Optional shared grounder
    grounding_result = task_args.get("grounding_result", None)  # Pre-computed grounding

    # Set seeds
    np.random.seed(seed + task_id)
    torch.manual_seed(seed + task_id)

    # Check device
    if device == "cuda" and not torch.cuda.is_available():
        print(f"[Task {task_id}] CUDA not available, falling back to CPU")
        device = "cpu"

    # Load checkpoint
    print(f"[Task {task_id}] Loading checkpoint: {checkpoint}")
    checkpoint_data = torch.load(checkpoint, map_location=device)

    # Get config from checkpoint
    ckpt_args = checkpoint_data.get("args", {})
    chunk_size = ckpt_args.get("chunk_size", 16)

    # Create policy with 6-dim goal
    policy = PickAndPlaceFlowPolicy(
        action_dim=7,
        chunk_size=chunk_size,
        hidden_dim=256,
        proprio_dim=8,
        goal_dim=GOAL_DIM,  # 6-dim goal (pick_pos + place_pos)
        pretrained_vision=False,
    ).to(device)

    policy.load_state_dict(checkpoint_data["model_state_dict"])
    policy.eval()

    # Create environment
    print(f"[Task {task_id}] Creating environment: {task_suite} task {task_id}")
    env, task_description = make_libero_env(task_suite, task_id)
    print(f"[Task {task_id}] Task: {task_description}")

    # Get initial obs
    init_obs = env.reset()

    # Use pre-computed grounding or compute live
    if grounding_result is not None:
        # Use pre-computed grounding (for parallel execution)
        pick_obj = grounding_result["pick_object"]
        place_target = grounding_result["place_target"]
        # Re-extract positions from current obs (positions may vary per reset)
        pick_pos = find_object_position(init_obs, pick_obj)
        place_pos = find_object_position(init_obs, place_target)
        place_pos[2] += 0.05  # Height offset
        goal = np.concatenate([pick_pos, place_pos])
        print(f"  Using pre-computed grounding: pick={pick_obj}, place={place_target}")
    elif grounder is not None:
        # Use shared grounder (for sequential execution)
        goal, pick_obj, place_target = extract_goal_6d(init_obs, task_description, grounder)
    else:
        # Load VLM (fallback for single-task mode)
        print(f"[Task {task_id}] Loading VLM grounder...")
        grounder = QwenSemanticGrounder()
        grounder.load_model()
        goal, pick_obj, place_target = extract_goal_6d(init_obs, task_description, grounder)

    # Log goal extraction
    print(f"[Task {task_id}] Pick object: {pick_obj}, Place target: {place_target}")
    print(f"[Task {task_id}] Goal - Pick pos: {goal[:3]}, Place pos: {goal[3:6]}")

    # Create output directory
    output_path = Path(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_path / f"{task_suite}_task{task_id}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Run evaluation
    results = []
    successes = []

    for ep in range(num_episodes):
        print(f"[Task {task_id}] Episode {ep + 1}/{num_episodes}")

        result = run_episode(
            env=env,
            policy=policy,
            goal=goal,
            device=device,
            max_steps=max_steps,
            num_inference_steps=num_inference_steps,
            debug_video=save_videos,
            task_description=task_description,
            pick_obj_name=pick_obj,
            place_target_name=place_target,
        )

        successes.append(result["success"])
        print(f"[Task {task_id}] Success: {result['success']}, Steps: {result['steps']}, Reward: {result['total_reward']:.2f}")

        # Save debug video
        if save_videos and len(result["debug_frames"]) > 0:
            try:
                import imageio
                video_path = run_dir / f"episode_{ep:04d}_debug.mp4"
                imageio.mimsave(str(video_path), result["debug_frames"], fps=20)
            except ImportError:
                pass

        results.append({
            "episode": ep,
            "success": result["success"],
            "steps": result["steps"],
            "total_reward": result["total_reward"],
        })

    # Summary
    success_rate = np.mean(successes)
    print(f"[Task {task_id}] Success rate: {success_rate:.1%} ({sum(successes)}/{len(successes)})")

    # Save results (exclude non-serializable objects like grounder)
    serializable_args = {k: v for k, v in task_args.items() if k != "grounder"}
    summary = {
        "task_suite": task_suite,
        "task_id": task_id,
        "task_description": task_description,
        "checkpoint": checkpoint,
        "num_episodes": num_episodes,
        "success_rate": success_rate,
        "goal": {
            "pick_object": pick_obj,
            "place_target": place_target,
            "pick_pos": goal[:3].tolist(),
            "place_pos": goal[3:6].tolist(),
        },
        "results": results,
        "args": serializable_args,
    }

    summary_path = run_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    env.close()

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate flow policy on LIBERO")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/pickandplace_best.pt",
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
        default=None,
        help="Single task ID within suite (use --task_ids for multiple)",
    )
    parser.add_argument(
        "--task_ids",
        type=int,
        nargs="+",
        default=None,
        help="Multiple task IDs to run in parallel (e.g., --task_ids 4 5 8 9)",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1,
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
        default=True,
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
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel workers for policy evaluation (VLM grounding is always sequential)",
    )
    args = parser.parse_args()

    # Determine task IDs to run
    if args.task_ids is not None:
        task_ids = args.task_ids
    elif args.task_id is not None:
        task_ids = [args.task_id]
    else:
        task_ids = [0]  # Default to task 0

    # ========== PHASE 1: VLM Grounding (Sequential, single VLM instance) ==========
    print("=" * 60)
    print("PHASE 1: VLM Grounding (sequential)")
    print("=" * 60)
    print("Loading VLM grounder...")
    grounder = QwenSemanticGrounder()
    grounder.load_model()
    print("VLM grounder ready.\n")

    # Pre-compute grounding for all tasks
    grounding_results = {}
    for tid in task_ids:
        print(f"[Task {tid}] Grounding...")
        # Create temp environment to get task description and initial obs
        env, task_description = make_libero_env(args.task_suite, tid)
        init_obs = env.reset()

        # Build objects and run VLM grounding
        objects = build_objects_from_obs(init_obs)
        result = grounder.ground(task_description, objects)

        if result.valid:
            grounding_results[tid] = {
                "pick_object": result.source_object,
                "place_target": result.target_location,
                "task_description": task_description,
            }
            print(f"  pick={result.source_object}, place={result.target_location}")
        else:
            print(f"  ERROR: {result.error}")
            grounding_results[tid] = None

        env.close()

    # Unload VLM to free GPU memory for parallel policy execution
    print("\nUnloading VLM to free GPU memory...")
    del grounder
    torch.cuda.empty_cache()
    print("VLM unloaded.\n")

    # ========== PHASE 2: Policy Evaluation (Parallel or Sequential) ==========
    print("=" * 60)
    print("PHASE 2: Policy Evaluation")
    print("=" * 60)

    # Build task arguments with pre-computed grounding
    task_args_list = []
    for tid in task_ids:
        if grounding_results[tid] is None:
            print(f"[Task {tid}] Skipping due to grounding failure")
            continue
        task_args_list.append({
            "task_id": tid,
            "task_suite": args.task_suite,
            "checkpoint": args.checkpoint,
            "num_episodes": args.num_episodes,
            "max_steps": args.max_steps,
            "num_inference_steps": args.num_inference_steps,
            "device": args.device,
            "save_videos": args.save_videos,
            "output_dir": args.output_dir,
            "seed": args.seed,
            "grounding_result": grounding_results[tid],  # Pre-computed grounding
        })

    num_workers = min(args.num_workers, len(task_args_list))

    if num_workers > 1:
        # Parallel execution (no VLM needed, just policy + env)
        # Use 'spawn' method to avoid CUDA re-initialization issues in forked processes
        import multiprocessing as mp
        mp.set_start_method('spawn', force=True)
        from multiprocessing import Pool
        print(f"Running {len(task_args_list)} tasks in parallel with {num_workers} workers")
        print(f"Task IDs: {[t['task_id'] for t in task_args_list]}")
        with Pool(processes=num_workers) as pool:
            summaries = pool.map(run_single_task, task_args_list)
    else:
        # Sequential execution
        print(f"Running {len(task_args_list)} tasks sequentially")
        print(f"Task IDs: {[t['task_id'] for t in task_args_list]}")
        summaries = []
        for task_args in task_args_list:
            summaries.append(run_single_task(task_args))

    # Print overall summary
    print(f"\n{'='*60}")
    print(f"OVERALL RESULTS: {args.task_suite}")
    print(f"{'='*60}")
    for s in summaries:
        status = "PASS" if s["success_rate"] > 0 else "FAIL"
        print(f"  Task {s['task_id']:2d}: {s['success_rate']:5.1%} - {s['task_description'][:50]}...")

    overall_rate = np.mean([s["success_rate"] for s in summaries])
    print(f"{'='*60}")
    print(f"Overall success rate: {overall_rate:.1%}")


if __name__ == "__main__":
    main()
