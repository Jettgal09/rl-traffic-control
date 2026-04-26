# visual_sandbox.py
#
# Quick visual check for Phase 2 grid geometry.
#
# This script doesn't touch the RL env at all — it just spins up a raw
# Road + PygameRenderer, applies a dumb fixed-timer cycling action to
# EVERY intersection, and lets you watch cars spawn and queue.
#
# WHY THIS EXISTS:
#   We refactored VehicleSpawner to be grid-aware (Phase 2, Option 1).
#   Before we touch handoff, training, or the env, we want to eyeball
#   that the spawn geometry actually lands where we expect on a 2x2 and
#   3x3 grid. Numbers passing an assert in a headless test is one thing,
#   watching the sim actually look right is another.
#
# USAGE:
#   uv run python visual_sandbox.py                       # default 2x2
#   uv run python visual_sandbox.py 3                     # 3x3
#   uv run python visual_sandbox.py 1                     # 1x1 (sanity — should look like Phase 1)
#   uv run python visual_sandbox.py 3 --spawn-rate 0.02   # 3x3 in DEMO MODE (light traffic)
#   uv run python visual_sandbox.py 3 --spawn-rate 0.10   # 3x3 under heavy stress
#
# DEMO MODE:
#   --spawn-rate overrides SimConfig.SPAWN_RATE for this session only. Useful
#   for presentations where you want sparse, legible traffic (0.02) rather
#   than the training-default firehose (0.05). Doesn't touch the config file,
#   so training runs next to you are unaffected.
#
# Press ESC or close the window to exit.

import argparse
import sys

from simulation.road import Road
from visualization.pygame_renderer import PygameRenderer


# --- HOW THE DUMB CONTROLLER CYCLES EACH LIGHT ---
# Every intersection gets the same action whenever it's at the start
# of a green phase. Action 1 = 40-step green, which is roughly the
# middle of our 20/40/60/80 duration menu — not too short, not too long.
# We're not trying to optimize here, just keep the lights changing so
# traffic flows and queues empty out periodically.
DEMO_ACTION = 1


def _parse_args():
    """
    CLI parser.

    Positional grid_size is kept for backward compat with the old
    `python visual_sandbox.py 3` invocation style — argparse happily
    accepts `nargs="?"` with a default, so old muscle memory still works.

    --spawn-rate is optional; if omitted we fall back to whatever
    SimConfig.SPAWN_RATE is currently set to (0.05 at the time of writing).
    """
    p = argparse.ArgumentParser(
        description="Visual sandbox for Phase 2 grid geometry — "
                    "spin up a Road, apply fixed-timer lights, watch cars flow."
    )
    p.add_argument(
        "grid_size", nargs="?", type=int, default=2,
        help="Number of intersections per side (default: 2). "
             "Use 1 for a Phase 1 sanity check.",
    )
    p.add_argument(
        "--spawn-rate", type=float, default=None, metavar="RATE",
        help="Override SimConfig.SPAWN_RATE for this run only. "
             "0.02 = light/demo traffic, 0.05 = default (training), "
             "0.10 = heavy/stress. Doesn't touch the config file.",
    )
    return p.parse_args()


def main():
    args = _parse_args()
    grid_size = args.grid_size

    if grid_size < 1:
        print(f"grid_size must be >= 1 (got {grid_size}).")
        sys.exit(1)

    if args.spawn_rate is not None and not (0.0 <= args.spawn_rate <= 1.0):
        print(f"--spawn-rate must be in [0.0, 1.0] (got {args.spawn_rate}).")
        sys.exit(1)

    print()
    print("=" * 50)
    print(f"  VISUAL SANDBOX  —  grid_size = {grid_size}")
    if args.spawn_rate is not None:
        print(f"  DEMO MODE      —  spawn_rate = {args.spawn_rate}")
    print("=" * 50)

    road = Road(grid_size=grid_size)
    renderer = PygameRenderer()

    # --- apply CLI spawn-rate override to every spawner ---
    # We do this AFTER Road() builds its spawners, since each spawner reads
    # SimConfig.SPAWN_RATE in its __init__. Going through set_spawn_rate()
    # (rather than stomping on the config class attribute) keeps the
    # override local to this process — training runs next to you see the
    # unaltered default.
    if args.spawn_rate is not None:
        for spawner in road.spawners:
            spawner.set_spawn_rate(args.spawn_rate)

    # --- log what spawners ended up with, as a sanity reminder in the terminal ---
    total_sps = 0
    for inter, spawner in zip(road.intersections, road.spawners):
        row, col = inter.grid_pos
        n = len(spawner.spawn_points)
        total_sps += n
        print(f"  intersection (r={row}, c={col}) @ ({inter.cx:.0f}, {inter.cy:.0f})  "
              f"→ {n} spawn points")
    print(f"  TOTAL spawn points: {total_sps}")
    print()
    print("Press ESC or close window to exit.")
    print()

    step = 0
    try:
        while True:
            # --- drive every intersection on its own schedule ---
            # apply_action is a no-op unless the light is at the start of a
            # new green phase, so we can call it every step on every
            # intersection without fear — it'll only actually do anything
            # when the intersection is ready for a new duration decision.
            for idx in range(len(road.intersections)):
                road.apply_action(DEMO_ACTION, intersection_idx=idx)

            road.step()
            renderer.render(road)
            step += 1
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        renderer.close()


if __name__ == "__main__":
    main()
