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
    Simulates a traditional fixed-timer traffic light controller.
    Switches phase every SWITCH_INTERVAL steps regardless of traffic.
    Implements predict() to match SB3 interface.
    """

    def __init__(self, switch_interval: int = 30):
        self.switch_interval = switch_interval
        self.step_counter = 0

    def predict(self, observation, deterministic=True):
        self.step_counter += 1
        if self.step_counter % self.switch_interval == 0:
            action = np.array(1)
        else:
            action = np.array(0)
        return action, None

    def reset(self):
        self.step_counter = 0


def evaluate_agent(model, env: TrafficEnv,
                   n_episodes: int = 10,
                   render: bool = False) -> dict:
    """
    Run an agent for n_episodes and collect performance metrics.
    Works for both SB3 models and FixedTimerBaseline.
    """
    if hasattr(model, 'reset'):
        model.reset()

    all_waiting = []
    all_queue   = []
    all_passed  = []
    all_lengths = []

    for episode in range(n_episodes):
        obs, info = env.reset()
        done      = False
        ep_waiting = 0
        ep_queue   = 0
        ep_steps   = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated

            ep_waiting += info["waiting_time"]
            ep_queue   += info["queue_length"]
            ep_steps   += 1

            if render:
                env.render()

        stats = env.get_episode_stats()
        all_waiting.append(ep_waiting)
        all_queue.append(ep_queue / max(ep_steps, 1))
        all_passed.append(stats["total_spawned"])
        all_lengths.append(ep_steps)

        print(f"  Episode {episode+1}/{n_episodes} | "
              f"waiting={ep_waiting:>10.0f} | "
              f"avg_queue={ep_queue/max(ep_steps,1):>5.2f} | "
              f"spawned={stats['total_spawned']:>4.0f}")

    return {
        "avg_total_waiting" : float(np.mean(all_waiting)),
        "avg_queue_per_step": float(np.mean(all_queue)),
        "avg_spawned"       : float(np.mean(all_passed)),
        "avg_ep_length"     : float(np.mean(all_lengths)),
        "std_waiting"       : float(np.std(all_waiting)),
    }


def compare_all(n_episodes: int = 10, output_dir: str = "experiments"):
    """
    Run evaluation for all trained models + baseline.
    Generates comparison table and bar charts for research paper.
    """
    results = {}
    env = TrafficEnv(grid_size=1)

    # Fixed Timer Baseline
    print("\n[Baseline] Fixed Timer (switch every 30 steps)")
    baseline = FixedTimerBaseline(switch_interval=30)
    results["Fixed Timer"] = evaluate_agent(baseline, env, n_episodes)

    # RL Algorithms
    model_paths = {
        "DQN": "experiments/dqn_results/models/best/best_model",
        "PPO": "experiments/ppo_results/models/best/best_model",
        "A2C": "experiments/a2c_results/models/best/best_model",
    }

    for algo, path in model_paths.items():
        model_file = path + ".zip"
        if not os.path.exists(model_file):
            print(f"\n[{algo}] No model found at {model_file} — skipping")
            continue

        print(f"\n[{algo}] Loading from {path}")
        model = ALGORITHMS[algo].load(path, env=env)
        results[algo] = evaluate_agent(model, env, n_episodes)

    # Print Results Table
    print("\n" + "="*65)
    print(f"{'Algorithm':<15} {'Avg Waiting':>12} {'Avg Queue':>10} {'Spawned':>8}")
    print("-"*65)
    for name, metrics in results.items():
        print(f"{name:<15} "
              f"{metrics['avg_total_waiting']:>12.0f} "
              f"{metrics['avg_queue_per_step']:>10.3f} "
              f"{metrics['avg_spawned']:>8.0f}")
    print("="*65)

    _save_results(results, output_dir)
    _plot_comparison(results, output_dir)

    return results


def _save_results(results: dict, output_dir: str):
    """Save results to text file for research notes."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "comparison_results.txt")

    with open(path, "w") as f:
        f.write("Algorithm Comparison Results\n")
        f.write("="*50 + "\n\n")
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
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"][:len(algorithms)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Traffic RL — Algorithm Comparison",
        fontsize=14,
        fontweight="bold"
    )

    metrics_to_plot = [
        ("avg_total_waiting",  "Total Waiting Time\n(lower is better)"),
        ("avg_queue_per_step", "Avg Queue Length/Step\n(lower is better)"),
        ("avg_spawned",        "Vehicles Spawned\n(higher = more traffic handled)"),
    ]

    for ax, (metric, title) in zip(axes, metrics_to_plot):
        values = [results[a][metric] for a in algorithms]
        bars = ax.bar(
            algorithms, values,
            color=colors, alpha=0.85,
            edgecolor="black", linewidth=0.8
        )
        ax.set_title(title, fontsize=11, pad=10)
        ax.set_ylabel("Value")
        ax.tick_params(axis="x", rotation=15)

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01,
                f"{val:.0f}",
                ha="center", va="bottom", fontsize=9
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
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    if args.compare:
        compare_all(n_episodes=args.episodes)

    elif args.algo:
        env = TrafficEnv(
            grid_size=1,
            render_mode="human" if args.render else None
        )

        if args.model:
            model_path = args.model
        else:
            model_path = (
                f"experiments/{args.algo.lower()}_results/models/best/best_model"
            )

        if not os.path.exists(model_path + ".zip"):
            print(f"No model found at {model_path}.zip")
            print("Train first: uv run python -m rl.train --algo DQN")
            return

        model = ALGORITHMS[args.algo].load(model_path, env=env)
        print(f"\nEvaluating {args.algo} for {args.episodes} episodes...")
        metrics = evaluate_agent(
            model, env,
            n_episodes=args.episodes,
            render=args.render
        )
        print("\nFinal Results:")
        for key, val in metrics.items():
            print(f"  {key}: {val:.2f}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()