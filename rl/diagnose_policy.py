# rl/diagnose_policy.py
#
# Diagnostic: what did the trained agent actually learn?
#
# WHY THIS EXISTS:
#   Our 2×2 A2C model beat the fixed-timer baseline by only 1.65% on reward
#   and tied on operational metrics. Two possible explanations:
#     (1) The policy is near-random / undertrained — argmax of a near-uniform
#         distribution gives noise-y action selection, eval reward happens to
#         land close to baseline by chance.
#     (2) The policy collapsed to picking one duration everywhere, which would
#         make it essentially a fixed timer at some specific cycle length.
#     (3) The policy actually varies by intersection state intelligently, and
#         it's just weak — in which case the approach works and we need
#         more capacity / better reward, not a different algorithm.
#
# These three have very different fixes, so before we retrain anything we
# want to know which world we're in.
#
# WHAT IT DOES:
#   Load the trained model, roll one deterministic episode, log every action
#   the agent picks at every step. Report:
#     - Overall action distribution (pooled across all N² intersections)
#     - Per-intersection action distribution
#     - Mode action per intersection (what it picks most often)
#     - Shannon entropy of each intersection's action distribution — tells us
#       how "decided" vs "still-exploring" the learned policy is
#
# USAGE:
#   uv run python rl/diagnose_policy.py --algo A2C --grid-size 2

import argparse
import numpy as np
from collections import Counter

from stable_baselines3 import DQN, PPO, A2C
from env.traffic_env import TrafficEnv

ALGORITHMS = {"DQN": DQN, "PPO": PPO, "A2C": A2C}

# Human-readable durations — matches TrafficEnv's action mapping.
ACTION_DURATIONS = {0: 20, 1: 40, 2: 60, 3: 80}


def diagnose(algo: str, grid_size: int):
    # --- Locate model ---
    # Phase-2 namespaced path; fall back to Phase-1 path at grid=1.
    new_path = f"experiments/{algo.lower()}_grid{grid_size}_results/models/best/best_model"
    old_path = f"experiments/{algo.lower()}_results/models/best/best_model"
    import os
    if os.path.exists(new_path + ".zip"):
        model_path = new_path
    elif grid_size == 1 and os.path.exists(old_path + ".zip"):
        model_path = old_path
    else:
        print(f"No model found at {new_path}.zip")
        return

    print(f"Loading {algo} from {model_path}")
    env = TrafficEnv(grid_size=grid_size)
    model = ALGORITHMS[algo].load(model_path, env=env)

    n_intersections = grid_size * grid_size

    # --- Roll one deterministic episode, log every action ---
    obs, info = env.reset()
    # actions_by_inter[i] is a list of the action chosen at intersection i
    # across the full 3000-step episode.
    actions_by_inter = [[] for _ in range(n_intersections)]

    done = False
    ep_reward = 0.0
    steps = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        # action is ndarray shape (N²,) for MultiDiscrete, scalar for Discrete
        if n_intersections == 1:
            actions_by_inter[0].append(int(action))
        else:
            for i, a in enumerate(action):
                actions_by_inter[i].append(int(a))

        obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
        ep_reward += float(reward)
        steps += 1

    print(f"\nEpisode: {steps} steps, reward={ep_reward:.2f}")

    # --- Overall distribution (pooled across intersections) ---
    all_actions = [a for lst in actions_by_inter for a in lst]
    pooled = Counter(all_actions)
    total = sum(pooled.values())
    print(f"\n{'='*60}")
    print(f"OVERALL ACTION DISTRIBUTION (pooled across {n_intersections} intersections)")
    print(f"{'='*60}")
    for action_idx in range(4):
        count = pooled.get(action_idx, 0)
        pct = 100.0 * count / total if total else 0
        bar = "█" * int(pct / 2)  # 50% = 25 chars
        print(
            f"  action {action_idx} ({ACTION_DURATIONS[action_idx]:>3d}-step green): "
            f"{count:>6d} ({pct:5.1f}%) {bar}"
        )

    # --- Per-intersection distribution ---
    print(f"\n{'='*60}")
    print("PER-INTERSECTION ACTION DISTRIBUTION")
    print(f"{'='*60}")
    print(
        f"  {'inter':>6} | "
        f"{'a=0(20s)':>9} {'a=1(40s)':>9} {'a=2(60s)':>9} {'a=3(80s)':>9} | "
        f"{'mode':>5} {'entropy':>8}"
    )
    print("  " + "-" * 68)
    for i, actions in enumerate(actions_by_inter):
        c = Counter(actions)
        t = sum(c.values())
        pcts = [100.0 * c.get(a, 0) / t for a in range(4)]
        mode = max(range(4), key=lambda a: c.get(a, 0))
        # Shannon entropy in bits. Max for 4 actions = 2.0 bits.
        # Near 2.0 = still-exploring / uniform. Near 0.0 = collapsed onto one.
        probs = np.array([c.get(a, 0) / t for a in range(4)])
        probs = probs[probs > 0]  # avoid log(0)
        entropy = -np.sum(probs * np.log2(probs))
        print(
            f"  {i:>6} | "
            f"{pcts[0]:>8.1f}% {pcts[1]:>8.1f}% {pcts[2]:>8.1f}% {pcts[3]:>8.1f}% | "
            f"{mode:>5} {entropy:>8.2f}"
        )

    # --- Interpretation hints ---
    # Pooled entropy tells us the GLOBAL "decidedness" of the policy.
    # If every intersection's entropy is near 2.0, the policy didn't converge.
    # If every intersection's entropy is near 0 AND they all picked the same
    # action, we've got a fixed-timer-in-disguise.
    per_inter_entropies = []
    for actions in actions_by_inter:
        c = Counter(actions)
        t = sum(c.values())
        probs = np.array([c.get(a, 0) / t for a in range(4)])
        probs = probs[probs > 0]
        per_inter_entropies.append(-np.sum(probs * np.log2(probs)))
    mean_entropy = np.mean(per_inter_entropies)

    print(f"\n{'='*60}")
    print("INTERPRETATION")
    print(f"{'='*60}")
    print(f"  Mean per-intersection entropy: {mean_entropy:.2f} bits (max = 2.00)")
    if mean_entropy > 1.8:
        diagnosis = (
            "NEAR-UNIFORM policy — agent didn't learn a clear preference.\n"
            "  Likely causes: ent_coef too high, or policy network too small\n"
            "  to express a useful value function over this obs space."
        )
    elif mean_entropy < 0.3:
        modes = [Counter(a).most_common(1)[0][0] for a in actions_by_inter]
        if len(set(modes)) == 1:
            diagnosis = (
                f"COLLAPSED policy — all intersections deterministically pick\n"
                f"  action {modes[0]} ({ACTION_DURATIONS[modes[0]]}-step green). This is\n"
                f"  effectively a fixed-timer controller at that duration.\n"
                f"  Fix: reward signal isn't rewarding differentiation — maybe\n"
                f"  mean aggregation smoothed away the per-intersection need\n"
                f"  to adapt. mean + λ·worst ablation is the next move."
            )
        else:
            diagnosis = (
                "PARTIALLY COLLAPSED — each intersection picks one action\n"
                "  deterministically but different intersections picked\n"
                "  different actions. Policy is finding per-intersection\n"
                "  fixed cycles — smarter than baseline but not adaptive."
            )
    else:
        diagnosis = (
            "MIXED policy — agent varies actions across the episode.\n"
            "  This is what we want. If reward is still weak vs baseline,\n"
            "  the approach works but needs more capacity or better reward."
        )
    print(f"  {diagnosis}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose a trained policy's action distribution")
    parser.add_argument("--algo", type=str, required=True, choices=["DQN", "PPO", "A2C"])
    parser.add_argument("--grid-size", type=int, default=2)
    args = parser.parse_args()
    diagnose(args.algo, args.grid_size)


if __name__ == "__main__":
    main()
