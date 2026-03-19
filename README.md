<!-- markdownlint-disable -->
# 🚦 Reinforcement Learning Based Adaptive Traffic Signal Control System

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10-red.svg)](https://pytorch.org/)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-2.7.1-green.svg)](https://stable-baselines3.readthedocs.io/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-1.2.3-orange.svg)](https://gymnasium.farama.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#)

A research-grade reinforcement learning system for adaptive traffic signal optimization. Trains RL agents (DQN, PPO, A2C) to control traffic light phase durations at a 4-way intersection, minimizing vehicle waiting times and queue lengths compared to a fixed-timer baseline.

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

- **Duration-based action space** — agent chooses green phase duration (20/40/60/80 steps) rather than binary switch decisions
- **Gym-compatible environment** — clean TrafficEnv interface for easy algorithm swapping
- **Three RL algorithms** — DQN, PPO, A2C all benchmarked against fixed-timer baseline
- **Reward function iteration** — 5 reward versions developed to solve degenerate policy problems
- **Pygame visualization** — real-time demo with color-coded vehicles and traffic lights
- **Demo mode** — side-by-side Fixed Timer vs RL Agent comparison

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

For each intersection (10-dimensional observation vector):
````
[queue_N, queue_S, queue_E, queue_W,    # Stopped vehicles per direction (normalized)
 count_N, count_S, count_E, count_W,    # Total vehicles per direction (normalized)
 phase_id, phase_timer]                  # Current signal phase and duration
````

### Action Space

```
Discrete(4) — agent selects green phase duration at start of each phase:
  0 → 20 steps green  (short — high opposing traffic)
  1 → 40 steps green  (medium)
  2 → 60 steps green  (long)  
  3 → 80 steps green  (very long — low opposing traffic)
```

### Reward Function
```
reward = -(queue_penalty + 2.0 × imbalance_penalty)
```
Penalizes both total congestion and unequal treatment of directions.
Five reward versions were developed during research (V1→V5).

---

## Installation
```bash
git clone https://github.com/yourusername/rl-traffic-system.git
cd rl-traffic-system
uv sync
```

**Requirements:** Python 3.12, CUDA optional (CPU training supported)

---

## Quick Start

### Run Expo Demo (Fixed Timer vs RL Agent)
```bash
uv run python demo.py
```

### Train an Agent
```bash
uv run python -m rl.train --algo DQN --timesteps 500000
uv run python -m rl.train --algo PPO --timesteps 500000
uv run python -m rl.train --algo A2C --timesteps 500000
```

### Evaluate and Compare All Algorithms
```bash
uv run python -m rl.evaluate --compare --episodes 10
```

### View Learning Curves
```bash
uv run python plot_results.py
```

---

## Training

Models, checkpoints, and logs are saved to `experiments/<algo>_results/`.
```bash
# Quick test run (1k steps)
uv run python -m rl.train --algo DQN --timesteps 1000

# Full training run
uv run python -m rl.train --algo DQN --timesteps 500000

# Heavy traffic experiment
uv run python -m rl.train --algo DQN --spawn-rate 0.10 --timesteps 500000
```

---

## Evaluation
```bash
# Compare all trained algorithms vs fixed-timer baseline
uv run python -m rl.evaluate --compare --episodes 10

# Evaluate single algorithm with visualization
uv run python -m rl.evaluate --algo DQN --render --episodes 5
```

## Results

| Controller  | Avg Waiting Time | Avg Queue | Throughput | vs Baseline |
|-------------|-----------------|-----------|------------|-------------|
| Fixed Timer | 15,248,102      | 33.02     | 898 veh    | baseline    |
| DQN         | 14,048,423      | 31.30     | 924 veh    | -7.9%       |
| PPO         | 13,808,619      | 31.09     | 924 veh    | -9.4%       |
| **A2C**     | **13,663,147**  | **30.76** | **923 veh**| **-10.4%**  |

All three RL algorithms outperform the fixed-timer baseline.
A2C achieved best performance: 10.4% reduction in waiting time.

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

All parameters are centralized in `utils/config.py`:
```python
class SimConfig:
    GRID_SIZE = 1                  # 1 = single intersection
    SPAWN_RATE = 0.05              # Vehicle spawn probability per step
    MIN_GREEN_DURATION = 15        # Minimum steps before phase can change
    MAX_STEPS_PER_EPISODE = 3000   # Episode length

class RLConfig:
    TOTAL_TIMESTEPS = 500_000
    ALGORITHM = "DQN"
    EVAL_FREQUENCY = 10_000
```

---

## Roadmap

- [x] Phase 1 — Single intersection simulation engine
- [x] Phase 1 — Gym-compatible RL environment  
- [x] Phase 1 — DQN, PPO, A2C training and evaluation
- [x] Phase 1 — Pygame visualization and expo demo
- [x] Phase 1 — Reward function iteration (V1→V5)
- [ ] Phase 2 — Multi-intersection 3×3 city grid
- [ ] Phase 3 — Vehicle turning logic
- [ ] Phase 4 — Congestion propagation between intersections
- [ ] Phase 5 — Multi-agent RL coordination

## Acknowledgements

- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) for RL algorithm implementations
- [Gymnasium](https://gymnasium.farama.org/) for the environment interface standard
- Research inspiration: *IntelliLight* (Wei et al., 2018) and *CoLight* (Wei et al., 2019)

---
