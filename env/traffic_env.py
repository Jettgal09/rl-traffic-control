# env/traffic_env.py
#
# The Gym environment — the bridge between our simulation and the RL agent.
#
# This file translates between two worlds:
#
#   SIMULATION WORLD          RL WORLD
#   ─────────────────         ────────────────
#   Vehicle queues      →     Observation vector (numbers)
#   Waiting time        →     Reward signal (negative number)
#   MAX_STEPS reached   →     truncated = True
#   Agent decision      ←     Action (0 or 1)
#
# Any RL algorithm (DQN, PPO, A2C) can plug into this without
# knowing anything about traffic — it just sees numbers in, numbers out.

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from simulation.road import Road
from utils.config import SimConfig


class TrafficEnv(gym.Env):
    """
    Gym compatible environment wrapping the traffic simulation.

    OBSERVATION SPACE:
      9 floats, all normalized between 0 and 1:
      [N_queue, S_queue, E_queue, W_queue,
       N_count, S_count, E_count, W_count,
       current_phase]

    ACTION SPACE:
      Discrete(2):
        0 → do nothing, keep current phase
        1 → request a phase switch

    REWARD:
      -(total_waiting_time + 0.5 * total_queue_length)
      Normalized by MAX_VEHICLES to keep values in a consistent range.
    """

    def __init__(
        self, grid_size: int = 1, render_mode: str = None, spawn_rate: float = None
    ):
        """
        PARAMETERS:
          grid_size   — 1 for single intersection (Phase 1)
          render_mode — "human" to show Pygame window, None for headless training
          spawn_rate  — override default spawn rate for experiments
        """
        super().__init__()

        self.grid_size = grid_size
        self.render_mode = render_mode

        # Build the simulation world
        self.road = Road(grid_size=grid_size)

        # Override spawn rate if specified
        if spawn_rate is not None:
            for spawner in self.road.spawners:
                spawner.set_spawn_rate(spawn_rate)

        # --- OBSERVATION SPACE ---
        # Tell Gymnasium exactly what shape and range our observations have.
        # This is how SB3 knows what size neural network input layer to create.
        # 9 values per intersection, all between 0.0 and 1.0
        obs_size = 10 * (grid_size * grid_size)

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

        # --- ACTION SPACE ---
        # Discrete(2) means two possible actions: 0 or 1
        self.action_space = spaces.Discrete(2)

        # Episode tracking
        self.current_step = 0
        self.max_steps = SimConfig.MAX_STEPS_PER_EPISODE

        # Accumulate metrics across the episode for logging
        self.episode_waiting_time = 0.0
        self.episode_queue_sum = 0.0

        # Renderer — only created if render_mode is set
        self._renderer = None
        self._prev_passed = 0

    def reset(self, seed=None, options=None):
        """
        Start a new episode.

        Called at the very beginning of training and after every
        episode ends (when truncated=True).

        RETURNS:
          observation — initial state as numpy array (all zeros at start)
          info        — empty dict (Gymnasium requires this second return value)
        """
        # This handles random seeding — required by Gymnasium API
        super().reset(seed=seed)

        # Reset the simulation world
        self.road.reset()

        # Reset episode counters
        self.current_step = 0
        self.episode_waiting_time = 0.0
        self.episode_queue_sum = 0.0
        self.prev_passed = 0
        # Get initial observation
        obs = self._get_observation()

        # Gymnasium requires reset() to return (observation, info)
        # info is just an empty dict for now
        info = {}

        return obs, info

    def step(self, action: int):
        """
        Execute one simulation step with the agent's chosen action.

        This is called millions of times during training.
        Every call is one tick of the simulation clock.

        THE LOOP:
          1. Apply agent's action to traffic lights
          2. Advance simulation by one step
          3. Calculate reward
          4. Check if episode is over
          5. Return everything back to the agent

        PARAMETERS:
          action — 0 (keep phase) or 1 (request switch)

        RETURNS:
          observation — new state after action (numpy array)
          reward      — how good or bad this step was (float)
          terminated  — always False in our simulation
          truncated   — True if we hit MAX_STEPS_PER_EPISODE
          info        — dict with extra metrics for logging
        """
        # Step 1 — apply action to simulation
        self.road.apply_action(action, intersection_idx=0)

        # Step 2 — advance the simulation world by one tick
        self.road.step()
        self.current_step += 1

        # Step 3 — calculate reward based on resulting state
        reward = self._calculate_reward()

        # Step 4 — accumulate episode metrics for logging
        metrics = self.road.get_metrics()
        self.episode_waiting_time += metrics["total_waiting_time"]
        self.episode_queue_sum += metrics["total_queue_length"]

        # Step 5 — check if episode is over
        terminated = False
        truncated = self.current_step >= self.max_steps

        # Step 6 — build observation of new state
        obs = self._get_observation()

        # Step 7 — build info dict
        # This is not used by the RL algorithm itself
        # but gets logged for our analysis
        info = {
            "step": self.current_step,
            "waiting_time": metrics["total_waiting_time"],
            "queue_length": metrics["total_queue_length"],
            "total_spawned": metrics["total_spawned"],
        }

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """
        Get current state from simulation and convert to numpy array.

        The RL agent's neural network takes this as input.
        Must be numpy float32 — that's what PyTorch expects.
        """
        obs_list = self.road.get_observation(intersection_idx=0)
        return np.array(obs_list, dtype=np.float32)

    def _calculate_reward(self) -> float:
        """
        Reward function v3 — queue-based with throughput bonus.
        """
        max_v = SimConfig.MAX_VEHICLES
        inter  = self.road.intersections[0]
        queues = inter.get_queue_lengths()

        ns_queue = queues["N"] + queues["S"]
        ew_queue = queues["E"] + queues["W"]

        # Penalty for total queue
        queue_penalty = (ns_queue + ew_queue) / max_v

        # Heavy imbalance penalty
        imbalance_penalty = (abs(ns_queue - ew_queue) / max_v) * 3.0

        # Throughput bonus — vehicles passed THIS step (delta not cumulative)
        current_passed = inter.total_vehicles_passed
        new_passed = current_passed - self._prev_passed
        self._prev_passed = current_passed
        throughput_bonus = min(new_passed / 10.0, 0.5)

        reward = -(queue_penalty + imbalance_penalty) + throughput_bonus

        return float(reward)

    def get_episode_stats(self) -> dict:
        """
        Summary statistics for the completed episode.
        Called after truncated=True to log how well the agent did.
        """
        return {
            "episode_length": self.current_step,
            "total_waiting_time": self.episode_waiting_time,
            "avg_wait_per_step": self.episode_waiting_time / max(self.current_step, 1),
            "total_queue_sum": self.episode_queue_sum,
            "total_spawned": self.road.total_spawned,
        }

    def render(self):
        """
        Render the simulation visually using Pygame.
        Only runs if render_mode="human" was set in __init__.
        During training we never render — it would be too slow.
        """
        if self.render_mode == "human":
            if self._renderer is None:
                from visualization.pygame_renderer import PygameRenderer

                self._renderer = PygameRenderer()
            self._renderer.render(self.road)

    def close(self):
        """Clean up Pygame window when done."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
