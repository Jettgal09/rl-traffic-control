# rl/evaluate.py

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO, A2C

from env.traffic_env import TrafficEnv

ALGORITHMS = {
    "DQN": DQN,
    "PPO": PPO,
    "A2C": A2C,
}


class FixedTimerBaseline:
    """
    Fixed timer baseline — the classical "dumb" traffic light.
    Alternates between action 1 (40-step green) and action 2 (60-step green)
    on every decision point, giving an average 50-step cycle per direction.
    This is the "what if we didn't use RL at all" comparison line the paper
    reports against the learned policy.

    GRID-AWARE (Phase 2):
      For a grid_size=N environment, action space is MultiDiscrete with one
      entry per intersection. The baseline broadcasts the same action to all
      N² intersections — every light in the grid picks the same duration on
      the same step. This is the fairest "dumb" baseline for a grid: no
      coordination, no adaptation, just identical fixed cycles everywhere.
    """

    def __init__(self, switch_interval: int = 30, grid_size: int = 1):
        self.switch_interval = switch_interval
        self.grid_size = grid_size
        self.n_intersections = grid_size * grid_size
        self.step_counter = 0

    def predict(self, observation, deterministic=True):
        self.step_counter += 1
        # Alternate between action 1 (40 steps) and action 2 (60 steps)
        # to simulate a fixed 50-step average cycle
        scalar_action = 1 if self.step_counter % 2 == 0 else 2
        if self.n_intersections == 1:
            action = np.array(scalar_action)  # Discrete
        else:
            # MultiDiscrete: broadcast same action to every intersection
            action = np.array([scalar_action] * self.n_intersections)
        return action, None

    def reset(self):
        self.step_counter = 0


def evaluate_agent(
    model, env: TrafficEnv, n_episodes: int = 10, render: bool = False
) -> dict:
    """
    Run an agent for n_episodes and collect performance metrics.
    Works for both SB3 models and FixedTimerBaseline.

    WHAT WE TRACK (and why):
      - episode_reward — sum of per-step rewards over one episode. This is
        the number EvalCallback reports as `eval/mean_reward` during training
        (e.g. PPO's -3790 on the 2×2 sanity run), so tracking it here lets
        us put the baseline side-by-side with trained agents in one table.
      - waiting_time / queue_length — operational metrics (not reward-shaped),
        what a city planner would actually care about.
      - total_spawned — how many cars even showed up; sanity-check across runs.

    ACTION DISPATCH:
      The action from predict() is passed directly to env.step() — no
      int() cast, because grid_size > 1 returns a MultiDiscrete array and
      casting an array to int raises TypeError. The env unpacks whichever
      shape it gets.
    """
    if hasattr(model, "reset"):
        model.reset()

    all_rewards = []
    all_waiting = []
    all_queue = []
    all_passed = []
    all_lengths = []

    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        ep_waiting = 0
        ep_queue = 0
        ep_steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            # Pass action through as-is. Scalar for Discrete (grid=1),
            # ndarray for MultiDiscrete (grid>1). int(action) would blow up
            # on the latter.
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_reward += float(reward)
            ep_waiting += info["waiting_time"]
            ep_queue += info["queue_length"]
            ep_steps += 1

            if render:
                env.render()

        stats = env.get_episode_stats()
        all_rewards.append(ep_reward)
        all_waiting.append(ep_waiting)
        all_queue.append(ep_queue / max(ep_steps, 1))
        all_passed.append(stats["total_spawned"])
        all_lengths.append(ep_steps)

        print(
            f"  Episode {episode + 1}/{n_episodes} | "
            f"reward={ep_reward:>9.1f} | "
            f"waiting={ep_waiting:>10.0f} | "
            f"avg_queue={ep_queue / max(ep_steps, 1):>5.2f} | "
            f"spawned={stats['total_spawned']:>4.0f}"
        )

    # Summary line (mean ± std of episode reward) — the headline number.
    print(
        f"  → mean_episode_reward = {np.mean(all_rewards):.2f} "
        f"± {np.std(all_rewards):.2f}"
    )

    return {
        "mean_episode_reward": float(np.mean(all_rewards)),
        "std_episode_reward": float(np.std(all_rewards)),
        "avg_total_waiting": float(np.mean(all_waiting)),
        "avg_queue_per_step": float(np.mean(all_queue)),
        "avg_spawned": float(np.mean(all_passed)),
        "avg_ep_length": float(np.mean(all_lengths)),
        "std_waiting": float(np.std(all_waiting)),
    }


def compare_all(
    n_episodes: int = 10,
    grid_size: int = 1,
    output_dir: str = "experiments",
):
    """
    Run evaluation for all trained models + baseline, all on the SAME grid
    size, and produce a comparison table and bar charts for the paper.

    WHY grid_size THREADED THROUGH:
      Each Phase-2 training run is namespaced by grid_size
      (experiments/ppo_grid2_results/... etc). A 3×3 PPO model is NOT
      interchangeable with a 1×1 PPO model — its observation/action shapes
      are different. So `compare_all` has to know which grid it's comparing
      on, load models from the matching dir, and instantiate an env of the
      same shape. Mismatched shapes would surface as an obs-space error the
      moment we try to load the model.

    MODEL PATH CONVENTION:
      experiments/{algo}_grid{N}_results/models/best/best_model.zip
      (matches what rl/train.py writes out.) For grid_size=1 with Phase 1
      legacy runs that predate the `_grid1` suffix, we fall back to the old
      `experiments/{algo}_results/...` path so Phase 1 numbers stay
      reproducible from the phase1-paper-results tag without re-training.
    """
    results = {}
    env = TrafficEnv(grid_size=grid_size)

    # Fixed Timer Baseline — our "dumb" reference point.
    print(f"\n[Baseline] Fixed Timer (alternating 40/60-step cycle, grid={grid_size}×{grid_size})")
    baseline = FixedTimerBaseline(switch_interval=30, grid_size=grid_size)
    results["Fixed Timer"] = evaluate_agent(baseline, env, n_episodes)

    # RL Algorithms — pull from the grid-namespaced experiments dir.
    for algo in ALGORITHMS.keys():
        # Phase 2 path (preferred).
        new_path = f"experiments/{algo.lower()}_grid{grid_size}_results/models/best/best_model"
        # Phase 1 legacy path — only relevant at grid_size=1.
        old_path = f"experiments/{algo.lower()}_results/models/best/best_model"

        path = None
        if os.path.exists(new_path + ".zip"):
            path = new_path
        elif grid_size == 1 and os.path.exists(old_path + ".zip"):
            path = old_path

        if path is None:
            print(f"\n[{algo}] No model found for grid={grid_size} — skipping")
            continue

        print(f"\n[{algo}] Loading from {path}")
        model = ALGORITHMS[algo].load(path, env=env)
        results[algo] = evaluate_agent(model, env, n_episodes)

    # Print Results Table — episode reward is the headline number (it's
    # what the agent optimized), the rest are operational metrics.
    print("\n" + "=" * 85)
    print(
        f"{'Algorithm':<15} {'Ep Reward':>18} "
        f"{'Avg Waiting':>12} {'Avg Queue':>10} {'Spawned':>8}"
    )
    print("-" * 85)
    for name, metrics in results.items():
        reward_str = (
            f"{metrics['mean_episode_reward']:>8.1f} "
            f"± {metrics['std_episode_reward']:.1f}"
        )
        print(
            f"{name:<15} "
            f"{reward_str:>18} "
            f"{metrics['avg_total_waiting']:>12.0f} "
            f"{metrics['avg_queue_per_step']:>10.3f} "
            f"{metrics['avg_spawned']:>8.0f}"
        )
    print("=" * 85)

    _save_results(results, output_dir)
    _plot_comparison(results, output_dir)

    return results


def _save_results(results: dict, output_dir: str):
    """Save results to text file for research notes."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "comparison_results.txt")

    with open(path, "w") as f:
        f.write("Algorithm Comparison Results\n")
        f.write("=" * 50 + "\n\n")
        for name, metrics in results.items():
            f.write(f"{name}:\n")
            for key, val in metrics.items():
                f.write(f"  {key}: {val:.2f}\n")
            f.write("\n")

    print(f"\nResults saved to {path}")


def _plot_comparison(results: dict, output_dir: str):
    """Generate bar charts for research paper."""
    if len(results) < 2:
        print("Need at least 2 results to compare — skipping plot")
        return

    algorithms = list(results.keys())
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"][: len(algorithms)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Traffic RL — Algorithm Comparison", fontsize=14, fontweight="bold")

    metrics_to_plot = [
        ("avg_total_waiting", "Total Waiting Time\n(lower is better)"),
        ("avg_queue_per_step", "Avg Queue Length/Step\n(lower is better)"),
        ("avg_spawned", "Vehicles Spawned\n(higher = more traffic handled)"),
    ]

    for ax, (metric, title) in zip(axes, metrics_to_plot):
        values = [results[a][metric] for a in algorithms]
        bars = ax.bar(
            algorithms,
            values,
            color=colors,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.8,
        )
        ax.set_title(title, fontsize=11, pad=10)
        ax.set_ylabel("Value")
        ax.tick_params(axis="x", rotation=15)

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01,
                f"{val:.0f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    chart_path = os.path.join(output_dir, "comparison_chart.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    print(f"Chart saved to {chart_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate Traffic RL agents")
    parser.add_argument("--algo", type=str, choices=["DQN", "PPO", "A2C"])
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run only the fixed-timer baseline (no model loading). "
             "Use this to get the 'dumb' reference number before your RL "
             "run finishes, so you have something to compare -3790 against.",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--grid-size",
        type=int,
        default=1,
        help="Grid size the model was trained on (1 = Phase 1, >1 = Phase 2). "
             "Must match the trained model's grid — obs/action shapes differ.",
    )
    args = parser.parse_args()

    if args.compare:
        compare_all(n_episodes=args.episodes, grid_size=args.grid_size)

    elif args.baseline:
        # Fixed-timer-only path — skips model loading so we can get the
        # baseline number immediately, without waiting for training to
        # finish or any model to exist on disk.
        env = TrafficEnv(
            grid_size=args.grid_size,
            render_mode="human" if args.render else None,
        )
        print(
            f"\n[Baseline] Fixed Timer — grid={args.grid_size}×{args.grid_size}, "
            f"{args.episodes} episodes"
        )
        baseline = FixedTimerBaseline(grid_size=args.grid_size)
        metrics = evaluate_agent(
            baseline, env, n_episodes=args.episodes, render=args.render
        )
        print("\nFinal Results (Fixed Timer):")
        for key, val in metrics.items():
            print(f"  {key}: {val:.2f}")

    elif args.algo:
        env = TrafficEnv(
            grid_size=args.grid_size,
            render_mode="human" if args.render else None,
        )

        if args.model:
            model_path = args.model
        else:
            # Prefer Phase-2-namespaced path; fall back to legacy path at
            # grid=1 for the phase1-paper-results-tag reproducibility case.
            new_path = (
                f"experiments/{args.algo.lower()}_grid{args.grid_size}_results/"
                f"models/best/best_model"
            )
            old_path = (
                f"experiments/{args.algo.lower()}_results/models/best/best_model"
            )
            if os.path.exists(new_path + ".zip"):
                model_path = new_path
            elif args.grid_size == 1 and os.path.exists(old_path + ".zip"):
                model_path = old_path
            else:
                model_path = new_path  # so the error message points to the right place

        if not os.path.exists(model_path + ".zip"):
            print(f"No model found at {model_path}.zip")
            print(
                f"Train first: uv run python rl/train.py --algo {args.algo} "
                f"--grid-size {args.grid_size}"
            )
            return

        model = ALGORITHMS[args.algo].load(model_path, env=env)
        print(
            f"\nEvaluating {args.algo} (grid={args.grid_size}×{args.grid_size}) "
            f"for {args.episodes} episodes..."
        )
        metrics = evaluate_agent(
            model, env, n_episodes=args.episodes, render=args.render
        )
        print("\nFinal Results:")
        for key, val in metrics.items():
            print(f"  {key}: {val:.2f}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
