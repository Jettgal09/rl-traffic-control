# plot_results.py
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_learning_curves():
    """
    Load evaluations.npz files and plot learning curves.
    DQN curve reconstructed from saved terminal output
    due to fragmented log files from interrupted training runs.
    """

    # --- DQN data reconstructed from training terminal output ---
    # These values were recorded from the training logs
    dqn_steps = np.array([10000, 20000, 40000, 70000, 80000, 90000, 100000, 500000])
    dqn_rewards = np.array(
        [-2616529, -2655819, -2602002, -1855362, -1994907, -1871475, -1979302, -451653]
    )

    colors = {
        "DQN": "#3498db",
        "PPO": "#2ecc71",
        "A2C": "#e67e22",
    }

    plt.figure(figsize=(12, 6))
    plt.title(
        "Learning Curves — Mean Episode Reward vs Training Steps",
        fontsize=13,
        fontweight="bold",
    )

    # Plot DQN manually
    plt.plot(
        dqn_steps,
        dqn_rewards,
        label="DQN",
        color=colors["DQN"],
        linewidth=2,
        marker="o",
        markersize=5,
    )

    # Annotate the breakthrough moment
    plt.annotate(
        "Breakthrough\n(-1.86M)",
        xy=(70000, -1855362),
        xytext=(120000, -1500000),
        fontsize=9,
        color=colors["DQN"],
        arrowprops=dict(arrowstyle="->", color=colors["DQN"]),
    )

    # Annotate final reward
    plt.annotate(
        "500k final\n(-451k)",
        xy=(500000, -451653),
        xytext=(380000, -800000),
        fontsize=9,
        color=colors["DQN"],
        arrowprops=dict(arrowstyle="->", color=colors["DQN"]),
    )

    # --- PPO from npz file ---
    ppo_path = "experiments/ppo_results/logs/evaluations.npz"
    if os.path.exists(ppo_path):
        data = np.load(ppo_path)
        timesteps = data["timesteps"]
        results = data["results"]
        mean_rewards = results.mean(axis=1)
        std_rewards = results.std(axis=1)

        plt.plot(timesteps, mean_rewards, label="PPO", color=colors["PPO"], linewidth=2)
        plt.fill_between(
            timesteps,
            mean_rewards - std_rewards,
            mean_rewards + std_rewards,
            alpha=0.2,
            color=colors["PPO"],
        )

        # Annotate catastrophic forgetting
        collapse_idx = mean_rewards.argmin()
        plt.annotate(
            "Catastrophic\nforgetting",
            xy=(timesteps[collapse_idx], mean_rewards[collapse_idx]),
            xytext=(
                timesteps[collapse_idx] - 80000,
                mean_rewards[collapse_idx] - 300000,
            ),
            fontsize=9,
            color=colors["PPO"],
            arrowprops=dict(arrowstyle="->", color=colors["PPO"]),
        )

    # --- A2C from npz file if exists ---
    a2c_path = "experiments/a2c_results/logs/evaluations.npz"
    if os.path.exists(a2c_path):
        data = np.load(a2c_path)
        timesteps = data["timesteps"]
        results = data["results"]
        mean_rewards = results.mean(axis=1)
        std_rewards = results.std(axis=1)

        plt.plot(timesteps, mean_rewards, label="A2C", color=colors["A2C"], linewidth=2)
        plt.fill_between(
            timesteps,
            mean_rewards - std_rewards,
            mean_rewards + std_rewards,
            alpha=0.2,
            color=colors["A2C"],
        )

    # --- Fixed timer baseline reference line ---
    # Average waiting time from evaluation: -23,259,857 normalized
    # We show it as a horizontal reference line
    plt.axhline(
        y=-451653,
        color="red",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="DQN Best (-451k)",
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
