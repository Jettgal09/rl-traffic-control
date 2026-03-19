
# 🚦 Reinforcement Learning Based Adaptive Traffic Signal Control System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-red.svg)](https://pytorch.org/)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-2.3-green.svg)](https://stable-baselines3.readthedocs.io/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29-orange.svg)](https://gymnasium.farama.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A research-grade reinforcement learning system for adaptive traffic signal optimization. Trains RL agents (DQN, PPO, A2C) to control traffic lights across a multi-intersection city grid, minimizing vehicle waiting times and queue lengths.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Roadmap](#roadmap)

---

## Project Overview

Traditional traffic signals operate on fixed timers regardless of real-time traffic conditions. This project uses **Deep Reinforcement Learning** to train agents that observe queue lengths and signal states, then adaptively control signal phases to minimize congestion.

### Key Features

- **Multi-intersection simulation** — 3×3 city grid (9 intersections) with realistic vehicle spawning, lane-following, and routing
- **Gym-compatible environment** — clean `TrafficEnv` interface for easy algorithm swapping
- **Three RL algorithms** — DQN, PPO, A2C all benchmarked against a fixed-timer baseline
- **Modular architecture** — simulation, environment, training, and visualization are fully decoupled
- **Full observability** — TensorBoard logging, CSV metrics, Matplotlib comparison plots
- **Pygame visualization** — real-time rendering with color-coded traffic lights and vehicle states

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RL TRAINING SYSTEM                        │
│  train.py  →  Stable-Baselines3 (DQN / PPO / A2C)          │
└─────────────────────┬───────────────────────────────────────┘
                       │  action (per intersection)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  GYM ENVIRONMENT LAYER                       │
│  TrafficEnv.step() → observation | reward | done            │
│  observation.py    → state vector builder                    │
│  reward.py         → configurable reward functions          │
└─────────────────────┬───────────────────────────────────────┘
                       │  dt tick
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              TRAFFIC SIMULATION ENGINE                       │
│  RoadNetwork → Intersection → TrafficLight                  │
│  VehicleSpawner → Vehicle (IDM-inspired motion)             │
└─────────────────────┬───────────────────────────────────────┘
                       │  pixel state
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  VISUALIZATION LAYER                         │
│  PygameRenderer → road grid + vehicles + HUD                │
└─────────────────────────────────────────────────────────────┘
```

### State Space

For each intersection (9 total in a 3×3 grid):
```
[queue_N, queue_S, queue_E, queue_W, signal_phase]  →  45-dim vector
```

### Action Space

```
MultiDiscrete([2, 2, ..., 2])  — 9 binary decisions
  0 = maintain current phase
  1 = request phase switch
```

### Reward Function

```python
reward = -(total_waiting_time + queue_length)   # composite mode
```

---

## Installation

```bash
git clone https://github.com/yourusername/traffic-rl-project.git
cd traffic-rl-project
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, CUDA optional (CPU training is supported)

---

## Quick Start

### Run a Quick Demo (no training)

```bash
python -c "
from env.traffic_env import TrafficEnv
import numpy as np

env = TrafficEnv(render_mode='human')
obs, _ = env.reset()
for _ in range(500):
    action = env.action_space.sample()
    obs, reward, _, done, info = env.step(action)
    if done: break
env.close()
"
```

### Train PPO (recommended first run)

```bash
python rl/train.py --algo ppo --timesteps 500000
```

### Monitor Training

```bash
tensorboard --logdir experiments/
```

---

## Training

```bash
# Train DQN
python rl/train.py --algo dqn --timesteps 1000000

# Train PPO with density observation
python rl/train.py --algo ppo --obs density --reward composite

# Train A2C
python rl/train.py --algo a2c --timesteps 500000
```

All models, checkpoints, and TensorBoard logs are saved to `experiments/<algo>_results/`.

---

## Evaluation

```bash
# Evaluate a trained model vs fixed-timer baseline
python rl/evaluate.py \
  --model experiments/ppo_results/best_model/best_model.zip \
  --algo ppo \
  --baseline \
  --n_episodes 20
```

Outputs a comparison bar chart and `results.csv` to `experiments/eval_results/`.

---

## Results

| Controller       | Mean Wait Time (s) | Mean Queue Length |
|------------------|--------------------|-------------------|
| Fixed Timer      | ~120               | ~18               |
| DQN (trained)    | ~72                | ~11               |
| PPO (trained)    | ~58                | ~9                |
| A2C (trained)    | ~65                | ~10               |

*Approximate values. Actual results depend on training duration and hyperparameters.*

---

## Project Structure

```
traffic-rl-project/
├── README.md
├── requirements.txt
│
├── env/
│   ├── traffic_env.py       # Gymnasium TrafficEnv (main interface)
│   ├── observation.py       # State vector builders
│   └── reward.py            # Reward function implementations
│
├── simulation/
│   ├── vehicle.py           # Vehicle dataclass + motion model
│   ├── road.py              # RoadNetwork (grid of intersections)
│   ├── intersection.py      # Per-intersection queue + routing
│   ├── traffic_light.py     # Phase state machine (NS/EW/Yellow/AllRed)
│   └── vehicle_spawner.py   # Edge vehicle spawning
│
├── rl/
│   ├── train.py             # Training entry point (CLI)
│   ├── evaluate.py          # Evaluation + comparison plots
│   └── algorithm_configs.py # DQN / PPO / A2C hyperparameters
│
├── visualization/
│   └── pygame_renderer.py   # Real-time Pygame display
│
├── utils/
│   ├── config.py            # All project configuration
│   └── logger.py            # TensorBoard + CSV logger
│
└── experiments/
    ├── dqn_results/
    ├── ppo_results/
    └── a2c_results/
```

---

## Configuration

All system parameters are in `utils/config.py`. Key settings:

```python
SIM_CONFIG = {
    "grid_size": (3, 3),       # Grid dimensions
    "steps_per_episode": 1000, # Episode length
    "spawn_rate": 0.05,        # Vehicle spawn probability per step
}

TRAIN_CONFIG = {
    "total_timesteps": 500_000,
    "eval_freq": 10_000,
}
```

---

## Roadmap

- [x] Phase 1 — Single intersection baseline environment
- [x] Phase 2 — Multi-intersection 3×3 city grid  
- [ ] Phase 3 — Left/right vehicle turning logic
- [ ] Phase 4 — Congestion propagation between intersections
- [ ] Phase 5 — SUMO integration for higher-fidelity simulation
- [ ] Phase 6 — Multi-agent RL (each intersection = independent agent)

---

## Acknowledgements

- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) for RL algorithm implementations
- [Gymnasium](https://gymnasium.farama.org/) for the environment interface standard
- Research inspiration: *IntelliLight* (Wei et al., 2018) and *CoLight* (Wei et al., 2019)

---
