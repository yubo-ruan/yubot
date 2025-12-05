"""ApproachObject skill.

Moves gripper to pre-grasp pose above target object.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np

from .base import Skill, SkillResult
from ..world_model.state import WorldState
from ..config import SkillConfig


def compute_pregrasp_pose(
    object_pose: np.ndarray,
    pregrasp_height: float = 0.10,
    gripper_orientation: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute pre-grasp pose above object."""
    pregrasp = np.zeros(7)
    pregrasp[:3] = object_pose[:3].copy()
    pregrasp[2] += pregrasp_height

    if gripper_orientation is not None:
        pregrasp[3:7] = gripper_orientation
    else:
        # Default: gripper pointing down
        pregrasp[3:7] = np.array([0.0, 0.707107, 0.707107, 0.0])

    return pregrasp


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate of quaternion [w, x, y, z]."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_error(q_target: np.ndarray, q_current: np.ndarray) -> np.ndarray:
    """Compute orientation error as axis-angle."""
    q_err = quat_multiply(q_target, quat_conjugate(q_current))
    if q_err[0] < 0:
        q_err = -q_err
    return 2.0 * q_err[1:4]


def compute_action(
    current_pose: np.ndarray,
    target_pose: np.ndarray,
    gripper: float,
    kp_pos: float = 10.0,
    kp_ori: float = 3.0,
) -> np.ndarray:
    """Compute action: delta * kp, clipped to [-1, 1]."""
    pos_error = target_pose[:3] - current_pose[:3]
    ori_error = quat_error(target_pose[3:7], current_pose[3:7])

    pos_action = np.clip(kp_pos * pos_error, -1, 1)
    ori_action = np.clip(kp_ori * ori_error, -1, 1)

    return np.concatenate([pos_action, ori_action, [np.clip(gripper, -1, 1)]])


class ApproachSkill(Skill):
    """Move gripper to pre-grasp pose above object."""

    name = "ApproachObject"

    def __init__(
        self,
        max_steps: int = 100,
        pos_threshold: float = 0.03,
        pregrasp_height: float = 0.10,
        config: Optional[SkillConfig] = None,
    ):
        super().__init__(max_steps=max_steps, config=config)

        if config:
            self.pos_threshold = config.approach_pos_threshold
            self.pregrasp_height = config.approach_pregrasp_height
            self.max_steps = config.approach_max_steps
            self.kp_pos = config.kp_pos
            self.kp_ori = config.kp_ori
        else:
            self.pos_threshold = pos_threshold
            self.pregrasp_height = pregrasp_height
            self.kp_pos = 10.0
            self.kp_ori = 3.0

    def preconditions(self, world_state: WorldState, args: Dict[str, Any]) -> Tuple[bool, str]:
        obj_name = args.get("obj")

        if obj_name is None:
            return False, "Missing 'obj' argument"

        if obj_name not in world_state.objects:
            return False, f"Object '{obj_name}' not found in world state"

        if world_state.is_holding():
            return False, "Gripper is already holding an object"

        return True, "OK"

    def postconditions(self, world_state: WorldState, args: Dict[str, Any]) -> Tuple[bool, str]:
        obj_name = args.get("obj")

        if world_state.gripper_pose is None:
            return False, "No gripper pose available"

        obj_pose = world_state.get_object_pose(obj_name)
        if obj_pose is None:
            return False, f"Object '{obj_name}' not found"

        target = compute_pregrasp_pose(obj_pose, self.pregrasp_height)
        distance = np.linalg.norm(world_state.gripper_pose[:3] - target[:3])

        if distance > self.pos_threshold:
            return False, f"Gripper not at target: {distance:.3f}m > {self.pos_threshold}m"

        return True, "OK"

    def execute(self, env, world_state: WorldState, args: Dict[str, Any]) -> SkillResult:
        obj_name = args.get("obj")

        obj_pose = world_state.get_object_pose(obj_name)
        if obj_pose is None:
            return SkillResult(
                success=False,
                info={"error_msg": f"Object '{obj_name}' not found", "steps_taken": 0}
            )

        # Get initial pose
        current_pose = world_state.gripper_pose
        if current_pose is None:
            obs, _, _, _ = self._step_env(env, np.zeros(7))
            current_pose = self._get_gripper_pose(env, obs)

        # Compute target (pregrasp pose)
        gripper_ori = current_pose[3:7] if current_pose is not None else None
        target_pose = compute_pregrasp_pose(obj_pose, self.pregrasp_height, gripper_orientation=gripper_ori)

        for step in range(self.max_steps):
            if current_pose is None:
                return SkillResult(
                    success=False,
                    info={"error_msg": "Failed to get gripper pose", "steps_taken": step}
                )

            # Check if at target
            pos_error = np.linalg.norm(target_pose[:3] - current_pose[:3])
            if pos_error < self.pos_threshold:
                return SkillResult(
                    success=True,
                    info={
                        "steps_taken": step + 1,
                        "reached_target": True,
                        "final_pose": current_pose,
                        "final_error": pos_error,
                    }
                )

            # Compute action: delta * kp, clipped
            action = compute_action(current_pose, target_pose, gripper=1.0,
                                   kp_pos=self.kp_pos, kp_ori=self.kp_ori)

            obs, _, _, _ = self._step_env(env, action)
            current_pose = self._get_gripper_pose(env, obs)

        # Timeout
        final_error = np.linalg.norm(target_pose[:3] - current_pose[:3]) if current_pose is not None else float('inf')
        return SkillResult(
            success=False,
            info={
                "error_msg": f"Timeout after {self.max_steps} steps",
                "steps_taken": self.max_steps,
                "timeout": True,
                "final_pose": current_pose,
                "final_error": final_error,
            }
        )

    def update_world_state(
        self,
        world_state: WorldState,
        args: Dict[str, Any],
        result: SkillResult,
    ):
        if "final_pose" in result.info and result.info["final_pose"] is not None:
            world_state.gripper_pose = result.info["final_pose"]
