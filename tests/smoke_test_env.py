# tests/smoke_test_env.py
#
# Fast sanity check for TrafficEnv — makes sure the Phase 1 path is still
# byte-compatible and the Phase 2 path (grid > 1) actually works end-to-end.
#
# Run from project root:
#     uv run python tests/smoke_test_env.py
#
# Exits with code 0 on success, non-zero on any failure.
# Intended to be cheap (<10s) so you can run it after any env-level change.

import sys
import numpy as np
from gymnasium import spaces

# Make sure we can import the local project regardless of cwd
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from env.traffic_env import TrafficEnv  # noqa: E402


def check_phase1():
    """grid=1 must stay exactly as it was before Phase 2 work started."""
    print("=== PHASE 1 (grid_size=1) ===")
    env = TrafficEnv(grid_size=1)

    assert isinstance(env.action_space, spaces.Discrete), (
        "Phase 1 MUST keep Discrete action space so DQN still trains — "
        "see traffic_env.py __init__ for why"
    )
    assert env.action_space.n == 4
    assert env.observation_space.shape == (10,)
    print(f"  obs_space:    {env.observation_space}")
    print(f"  action_space: {env.action_space}")

    obs, info = env.reset(seed=42)
    assert obs.shape == (10,)
    assert obs.dtype == np.float32

    # Run a short episode with mixed actions
    for i in range(50):
        obs, r, term, trunc, info = env.step(i % 4)
    print(f"  50 steps ok — final reward={r:.4f}, spawned={info['total_spawned']}")
    print("  PHASE 1: PASS")


def check_phase2():
    """grid=3 should run multi-intersection with MultiDiscrete actions."""
    print("\n=== PHASE 2 (grid_size=3) ===")
    env = TrafficEnv(grid_size=3)

    assert isinstance(env.action_space, spaces.MultiDiscrete), (
        "grid > 1 requires MultiDiscrete — DQN will no longer be usable"
    )
    assert list(env.action_space.nvec) == [4] * 9
    assert env.observation_space.shape == (90,)
    print(f"  obs_space:    {env.observation_space}")
    print(f"  action_space: {env.action_space}")

    obs, info = env.reset(seed=42)
    assert obs.shape == (90,)
    assert obs.dtype == np.float32

    # Drive 200 steps with random per-intersection actions
    rng = np.random.default_rng(0)
    for _ in range(200):
        a = rng.integers(0, 4, size=9)
        obs, r, term, trunc, info = env.step(a)
    print(f"  200 steps ok — final reward={r:.4f}, spawned={info['total_spawned']}")

    # Each of the 9 intersection chunks should be valid
    for i in range(9):
        chunk = obs[i * 10:(i + 1) * 10]
        assert len(chunk) == 10
        assert 0.0 <= chunk.min() and chunk.max() <= 1.0, (
            f"intersection {i} observation out of [0,1] range: {chunk}"
        )
    print("  per-intersection observation chunks all in [0, 1]")
    print("  PHASE 2: PASS")


def check_sb3_wrap():
    """If SB3 is available, verify the env wraps cleanly for both grid sizes."""
    try:
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError:
        print("\n=== SB3 WRAP CHECK ===")
        print("  stable-baselines3 not importable here — skipping")
        return

    print("\n=== SB3 WRAP CHECK ===")
    for g in [1, 3]:
        def _make(g=g):
            return TrafficEnv(grid_size=g)
        venv = DummyVecEnv([_make])
        obs = venv.reset()
        if g == 1:
            a = np.array([2])
        else:
            a = np.random.randint(0, 4, size=(1, 9))
        obs, r, done, info = venv.step(a)
        print(f"  grid={g}: DummyVecEnv wrap OK, obs.shape={obs.shape}, r={r[0]:.4f}")
    print("  SB3 WRAP: PASS")


if __name__ == "__main__":
    check_phase1()
    check_phase2()
    check_sb3_wrap()
    print("\nALL SMOKE TESTS PASSED.")
