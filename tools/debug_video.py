#!/usr/bin/env python3
"""Simple video debugging tool that overlays VLM planning info on episode videos.

Usage:
    python tools/debug_video.py <episode_json_path>
    python tools/debug_video.py logs/evaluation/qwen_grounded_*/episode_0000_*.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Warning: opencv-python not installed. Install with: pip install opencv-python")

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    print("Warning: imageio not installed. Install with: pip install imageio imageio-ffmpeg")


def load_episode_data(json_path: str) -> dict:
    """Load episode data from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def get_video_path(json_path: str) -> str:
    """Get video path from JSON path."""
    json_path = Path(json_path)
    video_path = json_path.with_suffix('.mp4')
    if video_path.exists():
        return str(video_path)
    return None


def create_debug_frame(frame: np.ndarray, episode_data: dict, frame_idx: int, total_frames: int) -> np.ndarray:
    """Add debug overlay to a video frame."""
    if not HAS_CV2:
        return frame

    # Scale up frame for better readability (128x128 -> 512x512)
    scale = 4
    frame = cv2.resize(frame, (frame.shape[1] * scale, frame.shape[0] * scale), interpolation=cv2.INTER_NEAREST)

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
        except:
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
    for i, skill in enumerate(skill_sequence):
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
                prefix = "✓ "
            else:
                status_color = (100, 100, 255)  # Red - failed
                prefix = "✗ "

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
        # Word wrap failure reason
        cv2.putText(panel, "Reason:", (10, y_offset), font, font_scale * 0.8, (180, 180, 180), 1)
        y_offset += line_height
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
    cv2.putText(panel, f"Frame: {frame_idx+1}/{total_frames}", (10, panel.shape[0] - 10),
                font, font_scale * 0.8, (150, 150, 150), 1)

    # Combine frame and panel
    combined = np.hstack([frame, panel])

    # Convert back to RGB
    combined = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)

    return combined


def create_debug_video(json_path: str, output_path: str = None):
    """Create debug video with overlay from episode JSON and video."""
    if not HAS_IMAGEIO:
        print("Error: imageio required for video processing")
        return False

    if not HAS_CV2:
        print("Error: opencv-python required for overlay creation")
        return False

    # Load episode data
    episode_data = load_episode_data(json_path)

    # Find video file
    video_path = get_video_path(json_path)
    if not video_path:
        print(f"Error: No video file found for {json_path}")
        return False

    print(f"Loading video: {video_path}")

    # Read video
    reader = imageio.get_reader(video_path)
    frames = [frame for frame in reader]
    reader.close()

    print(f"Processing {len(frames)} frames...")

    # Create debug frames
    debug_frames = []
    for i, frame in enumerate(frames):
        debug_frame = create_debug_frame(frame, episode_data, i, len(frames))
        debug_frames.append(debug_frame)

    # Output path
    if output_path is None:
        json_p = Path(json_path)
        output_path = str(json_p.parent / f"{json_p.stem}_debug.mp4")

    # Save debug video
    print(f"Saving debug video: {output_path}")
    imageio.mimsave(output_path, debug_frames, fps=20, codec='libx264', quality=8)

    print(f"Done! Debug video saved to: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Create debug video with VLM planning overlay")
    parser.add_argument("json_path", help="Path to episode JSON file")
    parser.add_argument("-o", "--output", help="Output video path (default: <input>_debug.mp4)")
    args = parser.parse_args()

    if not Path(args.json_path).exists():
        print(f"Error: File not found: {args.json_path}")
        sys.exit(1)

    success = create_debug_video(args.json_path, args.output)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
