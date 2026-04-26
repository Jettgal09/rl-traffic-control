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
#   Agent decision      ←     Action (duration choice per intersection)
#
# Any RL algorithm (DQN, PPO, A2C) can plug into this without
# knowing anything about traffic — it just sees numbers in, numbers out.
#
# PHASE 1: one intersection  — Discrete(4) action, 10-float observation
# PHASE 2: N × N grid        — MultiDiscrete([4]*N²), 10·N²-float observation
#
# IMPORTANT: Phase 1 behavior is byte-preserved. When grid_size=1 the action
# space remains plain Discrete(4) — this is what keeps DQN usable for Phase 1
# (Stable-Baselines3 DQN only supports Discrete, not MultiDiscrete).

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from simulation.road import Road
from utils.config import SimConfig


class TrafficEnv(gym.Env):
    """
    Gym-compatible environment wrapping the traffic simulation.

    OBSERVATION SPACE:
      10 floats per intersection, all normalized between 0 and 1:
        [N_queue, S_queue, E_queue, W_queue,
         N_count, S_count, E_count, W_count,
         current_phase, phase_timer]

      For an N×N grid, we concatenate every intersection's vector into
      a single flat array of length 10·N². The agent sees the whole city
      in one flat observation and decides per-intersection actions from it.

      Intersection ordering matches Road.intersections — row-major,
      so observation[0:10] is (0,0), observation[10:20] is (0,1), etc.

    ACTION SPACE:
      grid_size=1 — Discrete(4): a single duration choice {0..3}
                    (kept as Discrete so Phase 1 can still train with DQN)
      grid_size>1 — MultiDiscrete([4]*N²): one {0..3} per intersection
                    (DQN is NOT compatible with this — use PPO or A2C)

      Each chosen value maps through DURATION_OPTIONS to a green-phase length
      in simulation steps. Only takes effect at the first step of each new
      green phase; intermediate steps' actions are ignored by the light.

    REWARD:
      Per-intersection V5 reward = -(queue + 2·imbalance) / MAX_VEHICLES.
      For multi-intersection we aggregate those per-intersection rewards
      via _aggregate_rewards (defaults to mean — see PENDING_DISCUSSIONS.md #1).
    """

    def __init__(
        self, grid_size: int = 1, render_mode: str = None, spawn_rate: float = None
    ):
        """
        PARAMETERS:
          grid_size   — 1 for single intersection (Phase 1 paper baseline)
                        N for an N×N grid (Phase 2 — requires PPO or A2C)
          render_mode — "human" to show Pygame window, None for headless training
          spawn_rate  — override default spawn rate for experiments
        """
        super().__init__()

        self.grid_size = grid_size
        self.num_intersections = grid_size * grid_size
        self.render_mode = render_mode

        # Build the simulation world
        self.road = Road(grid_size=grid_size)

        # Override spawn rate if specified
        if spawn_rate is not None:
            for spawner in self.road.spawners:
                spawner.set_spawn_rate(spawn_rate)

        # --- OBSERVATION SPACE ---
        # 10 values per intersection, concatenated, all between 0.0 and 1.0.
        # Gymnasium uses this to size the neural network's input layer.
        obs_size = 10 * self.num_intersections
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

        # --- ACTION SPACE ---
        # PHASE 1 (grid=1): plain Discrete(4) so DQN still works.
        # PHASE 2 (grid>1): MultiDiscrete with one 4-way choice per intersection.
        # WHY NOT ALWAYS MULTIDISCRETE?
        #   SB3's DQN is hardcoded to Discrete action spaces — switching Phase 1
        #   to MultiDiscrete([4]) would silently break the phase1-paper-results
        #   training pipeline. The cost of this branch is one if-statement here;
        #   the payoff is that the published Phase 1 results stay reproducible.
        if self.num_intersections == 1:
            self.action_space = spaces.Discrete(4)
        else:
            self.action_space = spaces.MultiDiscrete([4] * self.num_intersections)

        # Episode tracking
        self.current_step = 0
        self.max_steps = SimConfig.MAX_STEPS_PER_EPISODE

        # Accumulate metrics across the episode for logging
        self.episode_waiting_time = 0.0
        self.episode_queue_sum = 0.0

        # Renderer — only created if render_mode is set
        self._renderer = None
        self._prev_passed = 0

        # V6 throughput-bonus state — last step's total_spawned. Δ between
        # consecutive steps is the per-step "cars served" rate that the
        # throughput bonus pays out on. Initialized to 0 here; reset() also
        # zeroes it on every episode boundary so episodes don't bleed.
        self._prev_spawned = 0

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
        # V6: total_spawned restarts at 0 each episode (road.reset() above did
        # that), so the running tally we diff against also has to. Forgetting
        # this was caught in dev — first step of episode 2 would otherwise see
        # delta_spawned = -big_number and crash the reward signal.
        self._prev_spawned = 0

        # Get initial observation
        obs = self._get_observation()

        # Gymnasium requires reset() to return (observation, info)
        info = {}

        return obs, info

    def step(self, action):
        """
        Execute one simulation step with the agent's chosen action(s).

        This is called millions of times during training.
        Every call is one tick of the simulation clock.

        THE LOOP:
          1. Apply agent's action(s) to traffic lights
          2. Advance simulation by one step
          3. Calculate reward
          4. Check if episode is over
          5. Return everything back to the agent

        PARAMETERS:
          action — PHASE 1: a single int 0..3 (Discrete)
                   PHASE 2: an array/list of N² ints 0..3 (MultiDiscrete)

        RETURNS:
          observation — new state after action (numpy array)
          reward      — aggregated per-intersection V5 reward (float)
          terminated  — always False in our simulation
          truncated   — True if we hit MAX_STEPS_PER_EPISODE
          info        — dict with extra metrics for logging
        """
        # Normalize action to a flat array of length N², regardless of whether
        # the space is Discrete (scalar int) or MultiDiscrete (array of ints).
        # This keeps the apply loop below uniform.
        actions = self._normalize_action(action)

        # Apply each action to its corresponding intersection.
        # The traffic light only actually consumes the action at the start of
        # a new green phase — mid-phase steps are silently ignored by the light.
        for idx, a in enumerate(actions):
            self.road.apply_action(int(a), intersection_idx=idx)

        # Advance simulation
        self.road.step()
        self.current_step += 1

        # _calculate_reward returns (aggregated, per_intersection_list) so we
        # can log worst/best intersection reward alongside the scalar the
        # agent trains on — see PENDING_DISCUSSIONS.md #1 on why we didn't
        # just swap aggregation to min. Diagnostic only; the agent optimizes
        # `reward`, not the min/max floats below.
        v5_reward, per_inter_rewards = self._calculate_reward()

        metrics = self.road.get_metrics()
        self.episode_waiting_time += metrics["total_waiting_time"]
        self.episode_queue_sum += metrics["total_queue_length"]

        # --- V6 throughput bonus ---
        # Pay the agent for every car the spawner successfully fired this step.
        # The throttle-inflow exploit (iter2 A2C) gets killed here: when the
        # agent congests entry lanes, the spawner refuses to add cars, so
        # delta_spawned drops to zero and the bonus dries up. Throttling stops
        # being a winning strategy because it costs the agent its bonus stream.
        delta_spawned = metrics["total_spawned"] - self._prev_spawned
        self._prev_spawned = metrics["total_spawned"]
        throughput_bonus = SimConfig.REWARD_THROUGHPUT_WEIGHT * delta_spawned

        # The agent trains on V5 + throughput. We keep them split in info so
        # TensorBoard can show their contributions separately and we can spot
        # if one term is dominating during a future post-mortem.
        reward = v5_reward + throughput_bonus

        terminated = False
        truncated = self.current_step >= self.max_steps
        obs = self._get_observation()

        info = {
            "step": self.current_step,
            "waiting_time": metrics["total_waiting_time"],
            "queue_length": metrics["total_queue_length"],
            "total_spawned": metrics["total_spawned"],
            # --- Diagnostic reward breakdown (not used by agent) ---
            # For grid=1 all three collapse to `v5_reward` — still harmless to log.
            # For grid>1 the gap between reward_mean and reward_min tells us
            # whether a single worst intersection is being ignored by the
            # mean-aggregation optimizer.
            "reward_mean": float(np.mean(per_inter_rewards)) if per_inter_rewards else 0.0,
            "reward_min":  float(np.min(per_inter_rewards))  if per_inter_rewards else 0.0,
            "reward_max":  float(np.max(per_inter_rewards))  if per_inter_rewards else 0.0,
            # V6 components — log each side separately so we can read TensorBoard
            # and tell whether the agent is winning by clearing queues
            # (v5_reward up) or by serving more demand (throughput_bonus up).
            # Mixing them into a single scalar would hide the diagnosis.
            "reward_v5":         float(v5_reward),
            "reward_throughput": float(throughput_bonus),
            "delta_spawned":     int(delta_spawned),
        }

        return obs, reward, terminated, truncated, info

    def _normalize_action(self, action) -> np.ndarray:
        """
        Coerce whatever Gymnasium hands us into a flat length-N² numpy array.

        WHY THIS EXISTS:
          SB3 will pass a plain Python int for Discrete spaces and a numpy
          array for MultiDiscrete. Rather than branching in step(), we
          normalize once here and the apply loop stays simple.
        """
        if self.num_intersections == 1:
            # Discrete → scalar, wrap into length-1 array
            return np.array([int(action)], dtype=np.int64)
        # MultiDiscrete → already array-like, just make sure it's numpy
        return np.asarray(action, dtype=np.int64)

    def _get_observation(self) -> np.ndarray:
        """
        Get current state from simulation and convert to numpy array.

        PHASE 1 (grid=1): returns shape (10,) — exact same bytes as before.
        PHASE 2 (grid>1): returns shape (10·N²,) — every intersection's
                          10-float vector concatenated in row-major order.

        The RL agent's neural network takes this as input.
        Must be numpy float32 — that's what PyTorch expects.
        """
        # Concatenate every intersection's observation.
        # For grid=1 this is a single 10-float vector — byte-identical to
        # the Phase 1 behavior that produced the paper-tagged results.
        parts = [
            self.road.get_observation(intersection_idx=i)
            for i in range(self.num_intersections)
        ]
        flat = [value for inter_obs in parts for value in inter_obs]
        return np.array(flat, dtype=np.float32)

    def _calculate_reward(self) -> tuple:
        """
        Compute the scalar reward for this step PLUS the raw per-intersection
        list used to derive it.

        RETURNS:
          (aggregated_scalar, per_intersection_list)
            aggregated_scalar    — what the agent actually trains on
            per_intersection_list — the N² individual V5 values; used by
                                    step() for diagnostic min/max/mean logging

        For grid=1 the list has one element and aggregation is a no-op, so
        Phase 1 behavior is byte-preserved.

        REWARD V5 (per intersection):
          Since the agent controls phase DURATION rather than moment-to-moment
          switching, we reward based on queue balance. No yellow/all-red
          penalty issue exists anymore because the agent doesn't trigger
          transitions directly.

          r_i = -( queue_total_i + 2.0 · |NS_queue - EW_queue| ) / MAX_VEHICLES

        The imbalance term is weighted 2× so the agent strongly prefers
        balanced phases over leaving one axis starved.
        """
        per_intersection_rewards = [
            self._compute_intersection_reward(inter)
            for inter in self.road.intersections
        ]
        aggregated = float(self._aggregate_rewards(per_intersection_rewards))
        return aggregated, per_intersection_rewards

    def _compute_intersection_reward(self, inter) -> float:
        """
        V5 reward for a single intersection — identical to the Phase 1 formula,
        just extracted so we can call it N² times.
        """
        max_v = SimConfig.MAX_VEHICLES
        queues = inter.get_queue_lengths()

        ns_queue = queues["N"] + queues["S"]
        ew_queue = queues["E"] + queues["W"]

        queue_penalty = (ns_queue + ew_queue) / max_v
        imbalance_penalty = (abs(ns_queue - ew_queue) / max_v) * 2.0

        return -(queue_penalty + imbalance_penalty)

    def _aggregate_rewards(self, rewards: list) -> float:
        """
        Collapse N² per-intersection rewards into a single scalar for the agent.

        ITERATION 2 (active): 0.5 · mean + 0.5 · min.

        Why the switch from pure mean:
          The 2×2 A2C diagnostic (rl/diagnose_policy.py) showed the policy
          collapsed to four intersection-specific FIXED cycles with mean
          action-entropy 0.05 bits — each intersection picked one duration
          100% of the time regardless of queue state. The agent found a
          better-than-baseline hand-tuned fixed-timer set, but NOT a state-
          adaptive controller.

          Root cause: pure mean aggregation dilutes per-intersection credit.
          When intersection i makes a state-adaptive correction worth +ε,
          its contribution to the mean is ε/N², so the gradient signal for
          "that was a good state-dependent choice" is N²× weaker than the
          gradient pressure toward "just pick a good steady-state constant."
          So the agent learned constants, which is exactly what mean-
          averaging rewards.

          Mixing in the min reward forces the agent to attend to the worst
          intersection each step. State-adaptive corrections matter most
          where the min lives — so credit flows back to whichever
          intersection is currently struggling, and constants-only policies
          stop dominating the gradient.

        Why 0.5 / 0.5 split (not a heavier min weight):
          Pure min is extremely sparse gradient — only one intersection
          shapes each update, training gets unstable. 50/50 is the standard
          starting point in potential-based reward shaping; gives the min
          equal voice without starving the other N²−1 intersections.

        Range check:
          mean ∈ [−1, 0], min ∈ [−1, 0], so 0.5·mean + 0.5·min ∈ [−1, 0].
          Same range as pure mean — Phase 1 hyperparameters still transfer,
          no fresh learning-rate sweep needed.

        Phase 1 compatibility:
          For N=1, min == mean, so 0.5·mean + 0.5·min == mean. The
          phase1-paper-results tag is unaffected.

        See RESEARCH_NOTES.md "Phase 2 Iteration 2" for the diagnostic
        evidence and PENDING_DISCUSSIONS.md #1 for alternatives.
        """
        if not rewards:
            return 0.0
        mean_r = float(np.mean(rewards))
        min_r  = float(np.min(rewards))
        return 0.5 * mean_r + 0.5 * min_r

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
