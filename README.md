# Isaac Lab Anymal-C Locomotion with Custom PPO

A from-scratch implementation of Proximal Policy Optimization (PPO) for quadruped robot locomotion, achieving **96% performance** compared to the production-grade RSL-RL library.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-0.47.7-green.svg)](https://isaac-sim.github.io/IsaacLab/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<p align="center">
  <img src="docs/images/anymal_walking.gif" alt="Anymal-C Walking" width="600"/>
</p>

## Results

### Training Comparison (10,000 iterations)

<p align="center">
  <img src="comparison_plots/reward_comparison_10000.png" alt="Training Comparison" width="800"/>
</p>

| Metric | RSL-RL PPO | My PPO | Performance |
|--------|-----------|--------|-------------|
| **Mean Episode Reward** | 27.87 | 26.63 | **96%** |
| **Episode Length** | 999/1000 | 999/1000 | **100%** |
| **Training Speed** | ~73K steps/s | ~77K steps/s | **105%** |
| **Convergence** | ~300 iter | ~400 iter | Comparable |

### Early Convergence (1,000 iterations)

<p align="center">
  <img src="comparison_plots/reward_comparison_1k.png" alt="Early Convergence" width="800"/>
</p>

### Detailed Metrics Comparison

<p align="center">
  <img src="comparison_plots/summary_comparison.png" alt="Summary Comparison" width="800"/>
</p>

## Key Features

- **From-Scratch PPO Implementation**: Complete algorithmic control with 600+ lines of documented code
- **GPU-Accelerated Training**: Supports 4096+ parallel environments on consumer GPUs
- **Observation Normalization**: Implements Welford's algorithm for stable training
- **Learnable Action STD**: Adaptive exploration without manual decay schedules
- **Keyboard Control**: Interactive testing with real-time velocity commands
- **TensorBoard Logging**: Comprehensive training visualization and comparison tools

## Installation

### Prerequisites

- NVIDIA GPU with CUDA support (RTX 3060+ recommended)
- [Isaac Sim 4.5+](https://developer.nvidia.com/isaac-sim)
- [Isaac Lab 0.47.7+](https://isaac-sim.github.io/IsaacLab/)
- Conda or Miniconda

### Setup

```bash
# Clone this repository
git clone https://github.com/mturan33/isaaclab-anymal-locomotion.git
cd isaaclab-anymal-locomotion

# Navigate to Isaac Lab directory
cd /path/to/IsaacLab

# Activate Isaac Lab environment
conda activate env_isaaclab

# Copy project files to Isaac Lab
cp -r isaaclab-anymal-locomotion/source/* source/
cp -r isaaclab-anymal-locomotion/scripts/* scripts/
```

## Quick Start

### Training

#### My PPO (From Scratch)

```bash
# Train with 4096 parallel environments (headless mode)
./isaaclab.bat -p scripts/train_anymal_custom.py \
    --task Isaac-MyAnymal-Flat-v0 \
    --num_envs 4096 \
    --headless \
    --max_iterations 10000
```

#### RSL-RL PPO (Baseline)

```bash
# Train using RSL-RL library
./isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-Anymal-C-Direct-v0 \
    --num_envs 4096 \
    --headless \
    --max_iterations 10000
```

### Inference

#### Visualize Trained Policy

```bash
# Run inference with 64 environments
./isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-MyAnymal-Flat-v0 \
    --num_envs 64
```

#### Record Video

```bash
./isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-MyAnymal-Flat-v0 \
    --num_envs 16 \
    --video \
    --video_length 500
```

### Keyboard Control

Interactive control of the trained robot with keyboard inputs:

```bash
# My PPO Model
./isaaclab.bat -p scripts/play_keyboard.py \
    --task Isaac-MyAnymal-Flat-v0 \
    --checkpoint logs/rsl_rl/custom_ppo_v2/2025-11-28_20-39-37/model_best.pt

# RSL-RL Model
./isaaclab.bat -p scripts/play_keyboard.py \
    --task Isaac-Velocity-Flat-Anymal-C-Direct-v0 \
    --checkpoint logs/rsl_rl/anymal_c_flat_direct/2025-11-28_12-15-24/model_9999.pt
```

#### Keyboard Controls

| Key | Action |
|-----|--------|
| `W` / `↑` / `Numpad 8` | Move Forward |
| `S` / `↓` / `Numpad 2` | Move Backward |
| `A` / `←` / `Numpad 4` | Strafe Left |
| `D` / `→` / `Numpad 6` | Strafe Right |
| `Q` / `Numpad 7` | Turn Left |
| `E` / `Numpad 9` | Turn Right |
| `R` | Reset Robot |
| `ESC` | Quit |

### TensorBoard Monitoring

```bash
# Start TensorBoard
tensorboard --logdir logs/rsl_rl --host localhost --port 6006

# Generate comparison plots
python scripts/tensorboard_export.py \
    --rsl_rl_log logs/rsl_rl/anymal_c_flat_direct/2025-11-28_12-15-24 \
    --custom_ppo_log logs/rsl_rl/custom_ppo_v2/2025-11-28_20-39-37 \
    --output comparison_plots \
    --max_iter 10000
```

## Project Structure

```
isaaclab-anymal-locomotion/
├── scripts/
│   ├── train_anymal_custom.py      # Custom PPO training script
│   ├── play_keyboard.py            # Keyboard control for testing
│   └── tensorboard_export.py       # Training curve comparison
├── source/isaaclab_tasks/isaaclab_tasks/direct/
│   └── my_anymal_quadruped/
│       ├── my_anymal_c_env.py      # Custom environment with velocity arrows
│       ├── my_anymal_c_env_cfg.py  # Environment configuration
│       └── __init__.py             # Task registration
├── comparison_plots/               # Generated comparison graphs
│   ├── reward_comparison_1k.png
│   ├── reward_comparison_10000.png
│   ├── summary_comparison.png
│   └── ...
├── logs/                           # Training logs (not tracked)
└── README.md
```

## Technical Details

### PPO Algorithm Implementation

The custom PPO implementation includes:

- **Actor-Critic Network**: Separate MLPs with configurable hidden dimensions
- **GAE (Generalized Advantage Estimation)**: λ=0.95, γ=0.99
- **Clipped Surrogate Objective**: ε=0.2
- **Value Function Clipping**: Prevents large value updates
- **Entropy Bonus**: Encourages exploration (coefficient=0.001)
- **Observation Normalization**: Running mean/std using Welford's algorithm

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Learning Rate | 3e-4 (with decay) |
| Batch Size | 24,576 |
| Mini-batches | 4 |
| Epochs per Update | 5 |
| Discount (γ) | 0.99 |
| GAE Lambda (λ) | 0.95 |
| Clip Range (ε) | 0.2 |
| Entropy Coefficient | 0.001 |

### Reward Components

| Component | Weight | Description |
|-----------|--------|-------------|
| `track_lin_vel_xy_exp` | +1.0 | Track commanded linear velocity |
| `track_ang_vel_z_exp` | +0.5 | Track commanded angular velocity |
| `action_rate_l2` | -0.01 | Penalize rapid action changes |
| `dof_torques_l2` | -1e-4 | Minimize joint torques |
| `dof_acc_l2` | -2.5e-7 | Minimize joint accelerations |
| `feet_air_time` | +0.5 | Encourage proper gait timing |
| `undesired_contacts` | -1.0 | Penalize body contacts |

### Velocity Visualization

The custom environment includes real-time velocity visualization arrows:

| Arrow | Color | Meaning |
|-------|-------|---------|
| 🔴 | **Red** | Commanded Velocity |
| 🟢 | **Green** | Actual Velocity |
| 🔵 | **Cyan** | Heading Direction |

### Tested Configuration

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA RTX 5070 Ti (12GB) |
| CPU | Intel i9-13900HX |
| RAM | 32GB DDR5 |
| OS | Windows 11 Pro |

## Citation

If you use this work, please cite:

```bibtex
@misc{turan2025ppo,
  author = {Mehmet Turan},
  title = {From-Scratch PPO for Quadruped Locomotion in Isaac Lab},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/mturan33/isaaclab-anymal-locomotion}
}
```

## References

- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/)
- [RSL-RL Library](https://github.com/leggedrobotics/rsl_rl)
- [Proximal Policy Optimization (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- [ANYmal Robot by ANYbotics](https://www.anybotics.com/anymal/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Mehmet Turan**
- GitHub: [@mturan33](https://github.com/mturan33)
- Project: [isaaclab-anymal-locomotion](https://github.com/mturan33/isaaclab-anymal-locomotion)

---

<p align="center">
  <b>⭐ If you find this project useful, please consider giving it a star! ⭐</b>
</p>
