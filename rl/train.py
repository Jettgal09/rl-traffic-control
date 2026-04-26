# rl/train.py
#
# Training script for the Traffic RL agent.
#
# HOW STABLE-BASELINES3 WORKS:
#   You create an algorithm object, pass it your Gym environment,
#   and call model.learn(). SB3 handles the entire training loop:
#     - Collecting experience by calling env.step()
#     - Storing experience in replay buffer (DQN) or rollout buffer (PPO/A2C)
#     - Computing loss and updating neural network weights
#     - Logging metrics to TensorBoard
#
# USAGE:
#   uv run python rl/train.py --algo DQN                    # Phase 1 (1x1)
#   uv run python rl/train.py --algo PPO                    # Phase 1 (1x1)
#   uv run python rl/train.py --algo A2C                    # Phase 1 (1x1)
#   uv run python rl/train.py --algo PPO --grid-size 2      # Phase 2 (2x2 sanity)
#   uv run python rl/train.py --algo A2C --grid-size 3      # Phase 2 (3x3 full)
#
# ALGORITHM × GRID COMPATIBILITY:
#   DQN only works with Discrete action spaces. For grid_size > 1 the env
#   switches to MultiDiscrete, which DQN cannot handle — we raise a clear
#   ValueError up front rather than let SB3 crash mid-training with a
#   cryptic tensor error. Use PPO or A2C for multi-intersection.

import os
import random
import argparse

import numpy as np
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor

from env.traffic_env import TrafficEnv
from rl.callbacks import TensorBoardRewardDiagnostics
from utils.config import RLConfig

# Maps string name to SB3 algorithm class
ALGORITHMS = {
    "DQN": DQN,
    "PPO": PPO,
    "A2C": A2C,
}

def make_env(grid_size: int = 1, spawn_rate: float = None):
    """
    Creates a monitored TrafficEnv.

    WHY A FUNCTION INSTEAD OF JUST CREATING THE ENV DIRECTLY?
      SB3 expects a function that returns an environment.
      This pattern also makes it easy to create multiple
      identical environments later if needed.

    PARAMETERS:
      grid_size  — 1 for Phase 1 single intersection (byte-preserves
                   phase1-paper-results). >1 for Phase 2 multi-intersection.
      spawn_rate — override default spawn rate for experiments.
    """
    env = TrafficEnv(grid_size=grid_size, spawn_rate=spawn_rate)
    env = Monitor(env)   # wrap with Monitor for automatic stat tracking
    return env


def train(algorithm: str = "DQN",
          total_timesteps: int = None,
          spawn_rate: float = None,
          grid_size: int = 1,
          seed: int = None):
    """
    Main training function.

    PARAMETERS:
      algorithm       — "DQN", "PPO", or "A2C"
      total_timesteps — how many steps to train for
      spawn_rate      — traffic density override for experiments
      grid_size       — 1 for Phase 1, >1 for Phase 2. DQN does NOT support
                        grid_size > 1 (see guard below).
      seed            — None for non-reproducible (default), or an int for
                        a reproducible run. When set we seed three sources:
                        Python's stdlib random (used by vehicle_spawner),
                        numpy (used everywhere in obs/reward math), and
                        the SB3 algorithm (network init + rollout sampling).
                        Without all three, "same seed" runs would still
                        diverge because spawn_rate rolls would re-shuffle.
                        Output goes to a seed-namespaced subdir so multi-seed
                        runs don't clobber each other.
    """
    # --- Fail fast on DQN + multi-intersection ---
    # SB3's DQN is hardcoded to Discrete action spaces. TrafficEnv's action
    # space switches to MultiDiscrete when grid_size > 1, which DQN cannot
    # consume — without this guard the user would get a cryptic shape error
    # from deep inside SB3's q-net forward pass 10-30 seconds into training,
    # after the env has already been built and the TensorBoard run started.
    # Better to refuse up front with a message that names the fix.
    if algorithm == "DQN" and grid_size > 1:
        raise ValueError(
            f"DQN does not support grid_size > 1 (got grid_size={grid_size}). "
            f"DQN requires a Discrete action space, but multi-intersection "
            f"uses MultiDiscrete. Use PPO or A2C for Phase 2 training."
        )

    total_timesteps = total_timesteps or RLConfig.TOTAL_TIMESTEPS

    # --- SEED THE THREE RNGs (if --seed was passed) ---
    # We seed BEFORE creating envs/models so every downstream construction
    # — neural net init, env reset, vehicle spawn rolls — sees the same
    # random state. Order doesn't matter, but doing it in one place up
    # front means "did we seed?" is a single grep.
    if seed is not None:
        random.seed(seed)        # vehicle_spawner.py uses random.random()
        np.random.seed(seed)     # belt-and-suspenders; SB3 also seeds via model kwarg

    # --- CREATE OUTPUT DIRECTORIES ---
    # Namespaced by (algorithm, grid_size) so Phase 1 runs (grid1) and
    # Phase 2 runs (grid2, grid3, ...) land in separate directories and
    # don't clobber each other's checkpoints or TensorBoard logs.
    #
    # When --seed is given we add a /seed{N}/ subdir so multi-seed variance
    # runs (the Phase 1 paper variance bars) each get their own home but
    # still group under one parent — point TensorBoard at the parent and
    # all seeds show up as separate runs in one chart.
    base_dir     = f"experiments/{algorithm.lower()}_grid{grid_size}_results"
    results_dir  = f"{base_dir}/seed{seed}" if seed is not None else base_dir
    model_dir    = f"{results_dir}/models"
    log_dir      = f"{results_dir}/logs"
    tb_log_dir   = f"{results_dir}/tensorboard"

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir,   exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)

    print(f"\n{'='*50}")
    print("  Traffic RL Training")
    print(f"  Algorithm  : {algorithm}")
    print(f"  Grid size  : {grid_size}×{grid_size}  ({grid_size*grid_size} intersection{'s' if grid_size*grid_size > 1 else ''})")
    print(f"  Timesteps  : {total_timesteps:,}")
    print(f"  Spawn Rate : {spawn_rate or RLConfig.__dict__.get('SPAWN_RATE', 0.05)}")
    print(f"  Seed       : {seed if seed is not None else 'unseeded (random)'}")
    print(f"  Output dir : {results_dir}")
    print(f"{'='*50}\n")

    # --- CREATE ENVIRONMENTS ---
    # We need two separate environments:
    #   train_env — used during training (agent explores and learns)
    #   eval_env  — used during evaluation (agent acts without exploration)
    #
    # WHY SEPARATE?
    #   During evaluation we want to measure the LEARNED policy only.
    #   If we used the training env, the exploration noise would
    #   corrupt our measurement of how good the agent actually is.
    train_env = make_env(grid_size=grid_size, spawn_rate=spawn_rate)
    eval_env  = make_env(grid_size=grid_size, spawn_rate=spawn_rate)

    # --- CREATE CALLBACKS ---

    # 1. Checkpoint callback
    # Saves the model every SAVE_FREQUENCY steps
    # If training crashes you can resume from the last checkpoint
    checkpoint_cb = CheckpointCallback(
        save_freq=RLConfig.SAVE_FREQUENCY,
        save_path=model_dir,
        name_prefix=f"{algorithm.lower()}_checkpoint",
        verbose=1,
    )

    # 2. Evaluation callback
    # Every EVAL_FREQUENCY steps, runs the agent for EVAL_EPISODES
    # episodes without exploration and logs the mean reward.
    # Also saves the best model seen so far.
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=f"{model_dir}/best",
        log_path=log_dir,
        eval_freq=RLConfig.EVAL_FREQUENCY,
        n_eval_episodes=RLConfig.EVAL_EPISODES,
        deterministic=True,   # no exploration during evaluation
        verbose=1,
    )

    # 3. Reward-diagnostics callback (Phase 2)
    # Forwards the env's per-intersection reward_min/mean/max/gap scalars
    # from info → TensorBoard under the `custom/` namespace. These are how
    # we'll spot the "mean hides a brutally congested intersection" failure
    # mode from PENDING_DISCUSSIONS.md #1 without rerunning any training.
    tb_diagnostics_cb = TensorBoardRewardDiagnostics()

    callbacks = CallbackList([checkpoint_cb, eval_cb, tb_diagnostics_cb])

    # --- CREATE THE RL MODEL ---
    AlgorithmClass = ALGORITHMS[algorithm]

    # Get hyperparameters for this algorithm from config
    hyperparams = _get_hyperparams(algorithm, tb_log_dir)

    # Pass seed to SB3 only when explicitly set — SB3 treats seed=None as
    # "use a fresh random state every time," which is what we want when
    # the user runs without --seed. When set, SB3 seeds (a) the policy
    # network init, (b) the action sampler, and (c) the env via
    # train_env.reset(seed=seed) on the first reset.
    if seed is not None:
        hyperparams = {**hyperparams, "seed": seed}

    # Create the model
    # This builds the neural network but doesn't start training yet
    model = AlgorithmClass(
        env=train_env,
        **hyperparams
    )

    print(f"Observation space : {train_env.observation_space}")
    print(f"Action space      : {train_env.action_space}")
    print(f"Policy            : {hyperparams['policy']}")
    print("\nStarting training...\n")

    # --- TRAIN ---
    # This one call runs the entire training loop.
    # SB3 calls env.step() millions of times, updating the
    # neural network after each batch of experience.
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    # --- SAVE FINAL MODEL ---
    final_path = f"{model_dir}/{algorithm.lower()}_grid{grid_size}_final"
    model.save(final_path)

    print(f"\n{'='*50}")
    print("  Training Complete!")
    print(f"  Final model saved to : {final_path}")
    print("  View training curves : tensorboard --logdir experiments/")
    print(f"{'='*50}\n")

    return model

def _get_hyperparams(algorithm: str, tb_log_dir: str) -> dict:
    """
    Returns hyperparameter dict for the given algorithm.

    WHY HARDCODE HERE INSTEAD OF A SEPARATE FILE?
      Keeps training self-contained — everything needed to
      reproduce a training run is in one place.
      When you write your research paper, you can point
      directly to this function for exact hyperparameters.
    """
    if algorithm == "DQN":
        return {
            "policy"                : "MlpPolicy",
            "learning_rate"         : RLConfig.DQN_LEARNING_RATE,
            "buffer_size"           : RLConfig.DQN_BUFFER_SIZE,
            "learning_starts"       : 1000,
            "batch_size"            : RLConfig.DQN_BATCH_SIZE,
            "gamma"                 : RLConfig.DQN_GAMMA,
            "train_freq"            : 4,
            "exploration_fraction"  : RLConfig.DQN_EXPLORATION_FRACTION,
            "exploration_final_eps" : 0.05,
            "target_update_interval": RLConfig.DQN_TARGET_UPDATE_INTERVAL,
            "verbose"               : 1,
            "tensorboard_log"       : tb_log_dir,
        }
    elif algorithm == "PPO":
        return {
            "policy"        : "MlpPolicy",
            "learning_rate" : RLConfig.PPO_LEARNING_RATE,
            "n_steps"       : RLConfig.PPO_N_STEPS,
            "batch_size"    : RLConfig.PPO_BATCH_SIZE,
            "n_epochs"      : RLConfig.PPO_N_EPOCHS,
            "gamma"         : RLConfig.PPO_GAMMA,
            "clip_range"    : RLConfig.PPO_CLIP_RANGE,
            # ITERATION 2 tweaks (see RESEARCH_NOTES Phase 2):
            #   ent_coef 0.05 → 0.01 — iteration 1 left PPO's policy at 94% of
            #   max entropy after 500k steps because the entropy term was
            #   dominating the policy gradient. 0.01 lets the distribution
            #   actually collapse to preferences. 0.01 is also SB3's default.
            "ent_coef"      : 0.01,
            #   net_arch [128,128] → [256,256] — 2×2 diagnostic showed the
            #   old net only had the capacity to learn four per-intersection
            #   FIXED cycles, not state-adaptive control. Doubling width
            #   gives room to represent action-depends-on-queue policies.
            "policy_kwargs" : {"net_arch": [256, 256]},
            "verbose"       : 1,
            "tensorboard_log": tb_log_dir,
        }
    elif algorithm == "A2C":
        return {
            "policy"        : "MlpPolicy",
            "learning_rate" : RLConfig.A2C_LEARNING_RATE,
            # ITERATION 2: n_steps 5 → 20. A2C's iteration-1 run had
            # explained_variance = −0.887 (critic worse than predicting the
            # mean) because n_steps=5 returns are extremely noisy on the
            # 2×2 reward signal. Longer rollouts give smoother advantage
            # estimates for the critic to fit. Tradeoff: fewer policy
            # updates per timestep, but each is better-informed. At 500k
            # steps we still get 25k updates — plenty.
            "n_steps"       : 20,
            "gamma"         : RLConfig.A2C_GAMMA,
            "ent_coef"      : 0.01,
            # ITERATION 2: explicit [256,256] net. A2C previously used SB3's
            # default (~[64,64]), too small for multi-intersection state-
            # adaptive policies — see PPO note above for the same diagnosis.
            # Matching PPO's new size keeps the two-algo comparison clean.
            "policy_kwargs" : {"net_arch": [256, 256]},
            "verbose"       : 1,
            "tensorboard_log": tb_log_dir,
        }
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def main():
    """Command line interface for training."""
    parser = argparse.ArgumentParser(description="Train Traffic RL agent")

    parser.add_argument(
        "--algo",
        type=str,
        default="DQN",
        choices=["DQN", "PPO", "A2C"],
        help="RL algorithm to use"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Total training timesteps (default from config)"
    )
    parser.add_argument(
        "--spawn-rate",
        type=float,
        default=None,
        help="Vehicle spawn rate: 0.02=light, 0.05=normal, 0.10=heavy"
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=1,
        help="Number of intersections per side (1 = Phase 1, >1 = Phase 2). "
             "DQN only supports grid_size=1; use PPO or A2C for larger grids."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible runs. Seeds Python random, numpy, "
             "and SB3 (network init + sampling). When set, output goes to "
             "experiments/{algo}_grid{N}_results/seed{seed}/ so multi-seed "
             "variance runs don't clobber each other. Omit for non-reproducible."
    )

    args = parser.parse_args()

    train(
        algorithm=args.algo,
        total_timesteps=args.timesteps,
        spawn_rate=args.spawn_rate,
        grid_size=args.grid_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()