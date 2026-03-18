# plot_results.py
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_learning_curves():
    algorithms = {
        "DQN": "experiments/dqn_results/logs/evaluations.npz",
        "PPO": "experiments/ppo_results/logs/evaluations.npz",
        "A2C": "experiments/a2c_results/logs/evaluations.npz",
    }

    colors = {
        "DQN": "#3498db",
        "PPO": "#2ecc71",
        "A2C": "#e67e22",
    }

    plt.figure(figsize=(12, 6))
    plt.title(
        "Learning Curves — Mean Episode Reward vs Training Steps",
        fontsize=13,
        fontweight="bold"
    )

    for algo, path in algorithms.items():
        if not os.path.exists(path):
            print(f"No log found for {algo} — skipping")
            continue

        data = np.load(path)
        timesteps   = data["timesteps"]
        results     = data["results"]
        mean_rewards = results.mean(axis=1)
        std_rewards  = results.std(axis=1)

        print(f"{algo}:")
        print(f"  First reward: {mean_rewards[0]:>10.0f}")
        print(f"  Best reward:  {mean_rewards.max():>10.0f} "
              f"at step {timesteps[mean_rewards.argmax()]}")
        print(f"  Final reward: {mean_rewards[-1]:>10.0f}")
        print()

        plt.plot(
            timesteps, mean_rewards,
            label=algo,
            color=colors[algo],
            linewidth=2,
            marker="o",
            markersize=3
        )
        plt.fill_between(
            timesteps,
            mean_rewards - std_rewards,
            mean_rewards + std_rewards,
            alpha=0.15,
            color=colors[algo]
        )

    # Fixed timer reference line
    # Average reward per episode for fixed timer
    # Based on evaluation: avg_waiting=15,248,102 over 3000 steps
    # Our reward = -(queue + imbalance) per step ≈ -33/100 * 3000 ≈ -990 per episode
    plt.axhline(
        y=-2785,
        color="red",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="Fixed Timer (approx)"
    )

    plt.xlabel("Training Steps", fontsize=11)
    plt.ylabel("Mean Episode Reward", fontsize=11)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = "experiments/learning_curves.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Learning curve saved to {output_path}")
    plt.show()


if __name__ == "__main__":
    plot_learning_curves()