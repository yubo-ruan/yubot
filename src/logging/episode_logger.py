"""Episode logging for debugging and training data collection.

Logs complete episode traces including skills, world state, and observations.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
import numpy as np

from ..skills.base import SkillResult
from ..world_model.state import WorldState
from ..config import RunConfig
from ..utils.git_info import get_git_info
from ..utils.timing import Timer

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays."""
    
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


@dataclass
class EpisodeLog:
    """Complete episode record for debugging and training.
    
    Captures everything needed to:
    1. Debug skill failures
    2. Analyze Qwen interactions
    3. Train future policies (BC/RL)
    """
    
    # Episode metadata
    task: str
    task_id: int
    episode_idx: int
    timestamp: str
    seed: int
    
    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    code_version: Dict[str, Any] = field(default_factory=dict)
    
    # Execution trace
    skill_sequence: List[Dict[str, Any]] = field(default_factory=list)
    world_state_trace: List[Dict[str, Any]] = field(default_factory=list)
    
    # Observations (for BC/RL later)
    observations: List[Dict[str, Any]] = field(default_factory=list)
    actions: List[List[float]] = field(default_factory=list)
    
    # Outcome
    success: bool = False
    failure_reason: Optional[str] = None
    total_steps: int = 0
    total_time: float = 0.0
    
    # Timing breakdown
    timing: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Qwen interactions (for Phase 2+)
    qwen_prompts: List[str] = field(default_factory=list)
    qwen_responses: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class EpisodeLogger:
    """Logger for episode traces.

    Usage:
        logger = EpisodeLogger("logs/run_001")
        logger.start_episode(task="pick up bowl", task_id=0, episode_idx=0, seed=42)

        # During episode:
        logger.log_skill("ApproachObject", {"obj": "bowl_1"}, result)
        logger.log_world_state(world_state)
        logger.log_observation(obs, step=10)
        logger.log_action(action)

        # End of episode:
        logger.end_episode(success=True)
    """

    def __init__(
        self,
        output_dir: str,
        config: Optional[RunConfig] = None,
        save_observations: bool = True,
        observation_sample_rate: int = 10,
        save_video: bool = True,
        video_fps: int = 20,
        frame_sample_rate: int = 1,
        debug_video: bool = True,
    ):
        """Initialize logger.

        Args:
            output_dir: Directory to save episode logs.
            config: Run configuration to log.
            save_observations: Whether to save observations.
            observation_sample_rate: Save every N steps.
            save_video: Whether to save episode videos.
            video_fps: Frames per second for saved videos.
            frame_sample_rate: Save every N frames (1 = every frame).
            debug_video: Whether to add debug overlay to videos (default True).
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.config = config
        self.save_observations = save_observations
        self.observation_sample_rate = observation_sample_rate
        self.save_video = save_video and HAS_IMAGEIO
        self.video_fps = video_fps
        self.frame_sample_rate = frame_sample_rate
        self.debug_video = debug_video and HAS_CV2

        self.current_episode: Optional[EpisodeLog] = None
        self.timer = Timer()
        self._episode_start_time: float = 0.0
        self._step_count: int = 0
        self._frames: List[np.ndarray] = []
        self._frame_count: int = 0
    
    def start_episode(
        self,
        task: str,
        task_id: int,
        episode_idx: int,
        seed: int,
    ):
        """Start logging a new episode.

        Args:
            task: Task description.
            task_id: Task ID in suite.
            episode_idx: Episode index in run.
            seed: Random seed for this episode.
        """
        self.timer.reset()
        self._episode_start_time = time.time()
        self._step_count = 0
        self._frames = []
        self._frame_count = 0

        self.current_episode = EpisodeLog(
            task=task,
            task_id=task_id,
            episode_idx=episode_idx,
            timestamp=datetime.now().isoformat(),
            seed=seed,
            config=self.config.to_dict() if self.config else {},
            code_version=get_git_info(),
        )
    
    def log_skill(
        self,
        skill_name: str,
        args: Dict[str, Any],
        result: SkillResult,
    ):
        """Log a skill execution.
        
        Args:
            skill_name: Name of the skill.
            args: Skill arguments.
            result: Skill result.
        """
        if self.current_episode is None:
            return
        
        # Convert numpy arrays in result.info
        info = {}
        for k, v in result.info.items():
            if isinstance(v, np.ndarray):
                info[k] = v.tolist()
            else:
                info[k] = v
        
        self.current_episode.skill_sequence.append({
            "skill": skill_name,
            "args": args,
            "success": result.success,
            "info": info,
            "timestamp": time.time() - self._episode_start_time,
        })
    
    def log_world_state(self, world_state: WorldState):
        """Log world state snapshot.
        
        Args:
            world_state: Current world state.
        """
        if self.current_episode is None:
            return
        
        self.current_episode.world_state_trace.append({
            "state": world_state.to_dict(),
            "timestamp": time.time() - self._episode_start_time,
        })
    
    def log_observation(self, obs: Dict[str, Any], step: int):
        """Log observation (sampled).
        
        Args:
            obs: Observation dictionary.
            step: Current step number.
        """
        if self.current_episode is None:
            return
        
        if not self.save_observations:
            return
        
        # Only save at sample rate
        if step % self.observation_sample_rate != 0:
            return
        
        # Extract key info (don't save full images to JSON)
        obs_record = {
            "step": step,
            "timestamp": time.time() - self._episode_start_time,
        }
        
        if 'robot0_eef_pos' in obs:
            obs_record["eef_pos"] = obs['robot0_eef_pos'].tolist() if isinstance(obs['robot0_eef_pos'], np.ndarray) else obs['robot0_eef_pos']
        if 'robot0_eef_quat' in obs:
            obs_record["eef_quat"] = obs['robot0_eef_quat'].tolist() if isinstance(obs['robot0_eef_quat'], np.ndarray) else obs['robot0_eef_quat']
        if 'robot0_gripper_qpos' in obs:
            obs_record["gripper_qpos"] = obs['robot0_gripper_qpos'].tolist() if isinstance(obs['robot0_gripper_qpos'], np.ndarray) else obs['robot0_gripper_qpos']
        
        self.current_episode.observations.append(obs_record)
        self._step_count = max(self._step_count, step)
    
    def log_action(self, action: np.ndarray):
        """Log action taken.
        
        Args:
            action: Action array.
        """
        if self.current_episode is None:
            return
        
        self.current_episode.actions.append(action.tolist() if isinstance(action, np.ndarray) else action)
    
    def log_qwen(self, prompt: str, response: str):
        """Log Qwen interaction.

        Args:
            prompt: Prompt sent to Qwen.
            response: Response from Qwen.
        """
        if self.current_episode is None:
            return

        self.current_episode.qwen_prompts.append(prompt)
        self.current_episode.qwen_responses.append(response)

    def log_frame(self, frame: np.ndarray):
        """Log a video frame.

        Args:
            frame: Image frame as numpy array (H, W, C) in RGB format.
                   The frame will be flipped vertically to correct orientation.
        """
        if not self.save_video or self.current_episode is None:
            return

        self._frame_count += 1

        # Only save at sample rate
        if self._frame_count % self.frame_sample_rate != 0:
            return

        # Flip vertically to correct orientation (LIBERO renders upside down)
        frame_corrected = np.flipud(frame).copy()
        self._frames.append(frame_corrected)
    
    def end_episode(
        self,
        success: bool,
        failure_reason: Optional[str] = None,
    ) -> str:
        """End episode and save log.

        Args:
            success: Whether episode succeeded.
            failure_reason: Reason for failure if any.

        Returns:
            Path to saved log file.
        """
        if self.current_episode is None:
            return ""

        self.current_episode.success = success
        self.current_episode.failure_reason = failure_reason
        self.current_episode.total_time = time.time() - self._episode_start_time
        self.current_episode.total_steps = self._step_count
        self.current_episode.timing = self.timer.summary()

        # Save JSON log to disk
        filename = f"episode_{self.current_episode.episode_idx:04d}_{self.current_episode.timestamp.replace(':', '-')}.json"
        filepath = self.output_dir / filename

        with open(filepath, 'w') as f:
            json.dump(self.current_episode.to_dict(), f, indent=2, cls=NumpyEncoder)

        # Save video if we have frames
        video_path = ""
        if self.save_video and len(self._frames) > 0 and HAS_IMAGEIO:
            video_filename = f"episode_{self.current_episode.episode_idx:04d}_{self.current_episode.timestamp.replace(':', '-')}.mp4"
            video_path = self.output_dir / video_filename

            # Apply debug overlay if enabled
            if self.debug_video:
                total_frames = len(self._frames)
                frames_to_save = [
                    self._create_debug_frame(frame, i, total_frames)
                    for i, frame in enumerate(self._frames)
                ]
            else:
                frames_to_save = self._frames

            try:
                imageio.mimsave(
                    str(video_path),
                    frames_to_save,
                    fps=self.video_fps,
                    codec='libx264',
                    quality=8,
                )
            except Exception as e:
                print(f"Warning: Failed to save video: {e}")
                video_path = ""

        # Reset
        episode = self.current_episode
        self.current_episode = None
        self._frames = []
        self._frame_count = 0

        return str(filepath)
    
    def get_timer(self) -> Timer:
        """Get timer for external timing measurements."""
        return self.timer

    def _create_debug_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        total_frames: int,
    ) -> np.ndarray:
        """Add debug overlay to a video frame.

        Args:
            frame: Original frame (H, W, C) in RGB format.
            frame_idx: Current frame index.
            total_frames: Total number of frames.

        Returns:
            Frame with debug overlay panel on the right.
        """
        if not HAS_CV2 or self.current_episode is None:
            return frame

        episode_data = self.current_episode.to_dict()

        # Scale up frame for better readability (128x128 -> 512x512)
        scale = 4
        frame = cv2.resize(
            frame,
            (frame.shape[1] * scale, frame.shape[0] * scale),
            interpolation=cv2.INTER_NEAREST,
        )

        # Convert RGB to BGR for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Create overlay panel on the right
        panel_width = 400
        panel = np.zeros((frame.shape[0], panel_width, 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)  # Dark gray background

        # Text settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        color = (255, 255, 255)
        line_height = 20
        y_offset = 25

        # Task info
        task = episode_data.get("task", "Unknown task")
        cv2.putText(panel, "TASK:", (10, y_offset), font, font_scale, (100, 200, 100), 1)
        y_offset += line_height

        # Word wrap task description
        words = task.split()
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

        y_offset += 10

        # Qwen grounding info (if available)
        qwen_responses = episode_data.get("qwen_responses", [])
        if qwen_responses:
            cv2.putText(panel, "VLM GROUNDING:", (10, y_offset), font, font_scale, (100, 200, 100), 1)
            y_offset += line_height

            try:
                grounding = json.loads(qwen_responses[0])
                source = grounding.get("source_object", "?")
                target = grounding.get("target_location", "?")
                confidence = grounding.get("confidence", "?")

                cv2.putText(panel, f"Source: {source[:35]}", (10, y_offset), font, font_scale * 0.9, (200, 200, 100), 1)
                y_offset += line_height
                cv2.putText(panel, f"Target: {target[:35]}", (10, y_offset), font, font_scale * 0.9, (100, 200, 200), 1)
                y_offset += line_height
                cv2.putText(panel, f"Confidence: {confidence}", (10, y_offset), font, font_scale * 0.9, color, 1)
                y_offset += line_height
            except Exception:
                cv2.putText(panel, "Parse error", (10, y_offset), font, font_scale, (100, 100, 200), 1)
                y_offset += line_height

        y_offset += 10

        # Skill sequence
        skill_sequence = episode_data.get("skill_sequence", [])
        cv2.putText(panel, "SKILL SEQUENCE:", (10, y_offset), font, font_scale, (100, 200, 100), 1)
        y_offset += line_height

        # Estimate which skill is executing based on frame index
        total_steps = sum(s.get("info", {}).get("steps_taken", 0) for s in skill_sequence)
        steps_per_frame = total_steps / max(total_frames, 1)
        current_step = int(frame_idx * steps_per_frame)

        cumulative_steps = 0
        for skill in skill_sequence:
            skill_name = skill.get("skill", "?")
            skill_steps = skill.get("info", {}).get("steps_taken", 0)
            skill_success = skill.get("success", False)

            # Determine skill status
            skill_end_step = cumulative_steps + skill_steps
            if current_step < cumulative_steps:
                status_color = (128, 128, 128)  # Gray - not started
                prefix = "  "
            elif current_step < skill_end_step:
                status_color = (100, 200, 255)  # Yellow - in progress
                prefix = "> "
            else:
                if skill_success:
                    status_color = (100, 255, 100)  # Green - success
                    prefix = "# "  # Checkmark
                else:
                    status_color = (100, 100, 255)  # Red - failed
                    prefix = "X "

            text = f"{prefix}{skill_name} ({skill_steps} steps)"
            cv2.putText(panel, text, (10, y_offset), font, font_scale * 0.9, status_color, 1)
            y_offset += line_height
            cumulative_steps = skill_end_step

        y_offset += 10

        # Episode result
        success = episode_data.get("success", False)
        failure_reason = episode_data.get("failure_reason", None)

        result_color = (100, 255, 100) if success else (100, 100, 255)
        result_text = "SUCCESS" if success else "FAILURE"
        cv2.putText(panel, f"RESULT: {result_text}", (10, y_offset), font, font_scale, result_color, 1)
        y_offset += line_height

        if failure_reason:
            cv2.putText(panel, "Reason:", (10, y_offset), font, font_scale * 0.8, (180, 180, 180), 1)
            y_offset += line_height
            # Word wrap failure reason
            words = failure_reason.split()
            line = ""
            for word in words:
                test_line = line + " " + word if line else word
                if len(test_line) * 7 < panel_width - 20:
                    line = test_line
                else:
                    cv2.putText(panel, line, (10, y_offset), font, font_scale * 0.8, (180, 180, 180), 1)
                    y_offset += int(line_height * 0.9)
                    line = word
            if line:
                cv2.putText(panel, line, (10, y_offset), font, font_scale * 0.8, (180, 180, 180), 1)

        # Frame counter at bottom
        cv2.putText(
            panel,
            f"Frame: {frame_idx + 1}/{total_frames}",
            (10, panel.shape[0] - 10),
            font,
            font_scale * 0.8,
            (150, 150, 150),
            1,
        )

        # Combine frame and panel
        combined = np.hstack([frame, panel])

        # Convert back to RGB
        combined = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)

        return combined


class RunSummary:
    """Summary of a full run (multiple episodes)."""
    
    def __init__(self, output_dir: str, config: Optional[RunConfig] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        
        self.episodes: List[Dict[str, Any]] = []
        self.start_time = time.time()
    
    def add_episode(self, episode_log: EpisodeLog):
        """Add episode result to summary."""
        self.episodes.append({
            "episode_idx": episode_log.episode_idx,
            "task": episode_log.task,
            "success": episode_log.success,
            "failure_reason": episode_log.failure_reason,
            "total_steps": episode_log.total_steps,
            "total_time": episode_log.total_time,
            "n_skills": len(episode_log.skill_sequence),
        })
    
    def save(self):
        """Save run summary."""
        n_success = sum(1 for e in self.episodes if e["success"])
        n_total = len(self.episodes)
        
        summary = {
            "config": self.config.to_dict() if self.config else {},
            "code_version": get_git_info(),
            "timestamp": datetime.now().isoformat(),
            "total_episodes": n_total,
            "successful_episodes": n_success,
            "success_rate": n_success / n_total if n_total > 0 else 0.0,
            "total_run_time": time.time() - self.start_time,
            "episodes": self.episodes,
        }
        
        with open(self.output_dir / "run_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
