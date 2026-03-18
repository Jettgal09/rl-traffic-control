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
#   uv run python rl/train.py --algo DQN
#   uv run python rl/train.py --algo PPO
#   uv run python rl/train.py --algo A2C

import os
import argparse

from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor

from env.traffic_env import TrafficEnv
from utils.config import RLConfig

# Maps string name to SB3 algorithm class
ALGORITHMS = {
    "DQN": DQN,
    "PPO": PPO,
    "A2C": A2C,
}

def make_env(spawn_rate: float = None):
    """
    Creates a monitored TrafficEnv.

    WHY A FUNCTION INSTEAD OF JUST CREATING THE ENV DIRECTLY?
      SB3 expects a function that returns an environment.
      This pattern also makes it easy to create multiple
      identical environments later if needed.
    """
    env = TrafficEnv(grid_size=1, spawn_rate=spawn_rate)
    env = Monitor(env)   # wrap with Monitor for automatic stat tracking
    return env


def train(algorithm: str = "DQN",
          total_timesteps: int = None,
          spawn_rate: float = None):
    """
    Main training function.

    PARAMETERS:
      algorithm       — "DQN", "PPO", or "A2C"
      total_timesteps — how many steps to train for
      spawn_rate      — traffic density override for experiments
    """
    total_timesteps = total_timesteps or RLConfig.TOTAL_TIMESTEPS

    # --- CREATE OUTPUT DIRECTORIES ---
    # Where to save models and logs for this algorithm
    results_dir  = f"experiments/{algorithm.lower()}_results"
    model_dir    = f"{results_dir}/models"
    log_dir      = f"{results_dir}/logs"
    tb_log_dir   = f"{results_dir}/tensorboard"

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir,   exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)

    print(f"\n{'='*50}")
    print("  Traffic RL Training")
    print(f"  Algorithm  : {algorithm}")
    print(f"  Timesteps  : {total_timesteps:,}")
    print(f"  Spawn Rate : {spawn_rate or RLConfig.__dict__.get('SPAWN_RATE', 0.05)}")
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
    train_env = make_env(spawn_rate=spawn_rate)
    eval_env  = make_env(spawn_rate=spawn_rate)

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

    callbacks = CallbackList([checkpoint_cb, eval_cb])

    # --- CREATE THE RL MODEL ---
    AlgorithmClass = ALGORITHMS[algorithm]

    # Get hyperparameters for this algorithm from config
    hyperparams = _get_hyperparams(algorithm, tb_log_dir)

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
    final_path = f"{model_dir}/{algorithm.lower()}_final"
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
            "ent_coef"      : 0.05,    # was 0.01 — forces more exploration
            "policy_kwargs" : {"net_arch": [128, 128]},  # larger network
            "verbose"       : 1,
            "tensorboard_log": tb_log_dir,
        }
    elif algorithm == "A2C":
        return {
            "policy"        : "MlpPolicy",
            "learning_rate" : RLConfig.A2C_LEARNING_RATE,
            "n_steps"       : RLConfig.A2C_N_STEPS,
            "gamma"         : RLConfig.A2C_GAMMA,
            "ent_coef"      : 0.01,
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

    args = parser.parse_args()

    train(
        algorithm=args.algo,
        total_timesteps=args.timesteps,
        spawn_rate=args.spawn_rate,
    )


if __name__ == "__main__":
    main()