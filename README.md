# Yubot

VLM-guided micro-policy for LIBERO benchmark.

## Installation

```bash
# Clone the repository
git clone https://github.com/yubo-ruan/brain-robot.git
cd brain-robot

# Install dependencies (includes LIBERO)
pip install -r requirements.txt

# Download Qwen2.5-VL-7B model (~16GB, required for VLM features)
python scripts/download_models.py
```

> **Note**: Model weights are not included in the repo. Run the download script after cloning.

## Dependencies

### robosuite (Required)

This project uses **robosuite's Operational Space Controller (OSC)** for low-level robot control. LIBERO environments are built on robosuite, which provides impedance-controlled motion with proper PD gains.

- robosuite is installed as a dependency of LIBERO
- We generate smooth trajectories; OSC handles the actual control
- No custom controller implementation needed - we leverage robosuite's battle-tested OSC

The trajectory generator in `src/control/trajectory.py` produces waypoints that OSC tracks with its internal impedance control (kp=150, kd≈24.5).
