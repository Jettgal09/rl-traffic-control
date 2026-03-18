# demo.py
#
# Live demonstration script for expo/hackathon presentations.
#
# Shows two modes back to back:
#   1. Fixed Timer controller (baseline)
#   2. A2C RL agent (best performing algorithm)
#
# Judges can visually see the difference in traffic flow.
#
# USAGE:
#   uv run python demo.py

import time
import sys
import pygame
import numpy as np
from stable_baselines3 import A2C, DQN, PPO

from env.traffic_env import TrafficEnv
from visualization.pygame_renderer import PygameRenderer
from utils.config import VisualizationConfig as VC, SimConfig


# --- DEMO CONFIGURATION ---
STEPS_PER_MODE = 2500  # Steps to run each mode (1500 = ~50 seconds at 30fps)
DEMO_FPS = 1000  # Frames per second
SHOW_COMPARISON = True  # Show fixed timer first, then RL agent


class DemoRunner:
    """
    Runs the traffic simulation demo for expo presentation.
    Shows fixed timer vs RL agent comparison.
    """

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((VC.WINDOW_WIDTH, VC.WINDOW_HEIGHT))
        pygame.display.set_caption("Traffic RL — Adaptive Signal Control Demo")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 18)
        self.big_font = pygame.font.SysFont("monospace", 28, bold=True)
        self.small_font = pygame.font.SysFont("monospace", 14)

    def run(self):
        """Main demo loop — runs fixed timer then RL agent."""

        if SHOW_COMPARISON:
            print("\n" + "=" * 50)
            print("  TRAFFIC RL DEMO")
            print("  Phase 1: Fixed Timer Baseline")
            print("=" * 50)
            fixed_stats = self._run_fixed_timer()

            print("\n" + "=" * 50)
            print("  Phase 2: A2C RL Agent")
            print("=" * 50)
            rl_stats = self._run_rl_agent()

            self._show_results_screen(fixed_stats, rl_stats)
        else:
            self._run_rl_agent()

        pygame.quit()

    def _run_fixed_timer(self) -> dict:
        """Run fixed timer baseline and return stats."""
        env = TrafficEnv(grid_size=1)
        obs, info = env.reset()

        total_waiting = 0
        total_queue = 0
        step_counter = 0
        vehicles_spawned = 0

        for step in range(STEPS_PER_MODE):
            # Handle quit events
            if self._check_quit():
                pygame.quit()
                sys.exit()

            # Fixed timer: switch every 40 steps (action 1)
            action = 1 if step % 80 < 40 else 2

            obs, reward, terminated, truncated, info = env.step(action)
            total_waiting += info["waiting_time"]
            total_queue += info["queue_length"]
            vehicles_spawned = info["total_spawned"]
            step_counter += 1

            # Render
            self._render_frame(
                env=env,
                mode_label="FIXED TIMER (Baseline)",
                mode_color=(220, 80, 80),  # Red label
                step=step,
                total_steps=STEPS_PER_MODE,
                total_waiting=total_waiting,
                avg_queue=total_queue / max(step_counter, 1),
                spawned=vehicles_spawned,
            )
            self.clock.tick(DEMO_FPS)

        return {
            "total_waiting": total_waiting,
            "avg_queue": total_queue / max(step_counter, 1),
            "spawned": vehicles_spawned,
        }

    def _run_rl_agent(self) -> dict:
        """Run best RL agent (A2C) and return stats."""
        env = TrafficEnv(grid_size=1)

        # Try loading A2C, fall back to PPO then DQN
        model = None
        algo_name = ""
        for algo, path in [
            ("A2C", "experiments/a2c_results/models/best/best_model"),
            ("PPO", "experiments/ppo_results/models/best/best_model"),
            ("DQN", "experiments/dqn_results/models/best/best_model"),
        ]:
            try:
                if algo == "A2C":
                    model = A2C.load(path, env=env)
                elif algo == "PPO":
                    model = PPO.load(path, env=env)
                else:
                    model = DQN.load(path, env=env)
                algo_name = algo
                print(f"Loaded {algo} model from {path}")
                break
            except Exception as e:
                print(f"Could not load {algo}: {e}")
                continue

        if model is None:
            print("No trained model found! Run training first.")
            pygame.quit()
            sys.exit()

        obs, info = env.reset()
        total_waiting = 0
        total_queue = 0
        step_counter = 0
        vehicles_spawned = 0

        for step in range(STEPS_PER_MODE):
            if self._check_quit():
                pygame.quit()
                sys.exit()

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_waiting += info["waiting_time"]
            total_queue += info["queue_length"]
            vehicles_spawned = info["total_spawned"]
            step_counter += 1

            self._render_frame(
                env=env,
                mode_label=f"{algo_name} RL AGENT (Adaptive)",
                mode_color=(80, 200, 120),  # Green label
                step=step,
                total_steps=STEPS_PER_MODE,
                total_waiting=total_waiting,
                avg_queue=total_queue / max(step_counter, 1),
                spawned=vehicles_spawned,
            )
            self.clock.tick(DEMO_FPS)

        return {
            "total_waiting": total_waiting,
            "avg_queue": total_queue / max(step_counter, 1),
            "spawned": vehicles_spawned,
        }

    def _render_frame(
        self,
        env,
        mode_label,
        mode_color,
        step,
        total_steps,
        total_waiting,
        avg_queue,
        spawned,
    ):
        """Render one frame with simulation + overlay."""
        from visualization.pygame_renderer import PygameRenderer

        # Draw simulation
        self.screen.fill(VC.COLOR_BACKGROUND)
        road = env.road
        inter = road.intersections[0]

        # Draw roads
        self._draw_roads(inter)
        self._draw_intersection(inter)
        self._draw_lights(inter)
        self._draw_vehicles(inter)

        # Draw mode label banner at top
        self._draw_banner(mode_label, mode_color)

        # Draw metrics panel
        self._draw_metrics(step, total_steps, total_waiting, avg_queue, spawned, inter)

        # Draw progress bar
        self._draw_progress_bar(step, total_steps)

        pygame.display.flip()

    def _draw_banner(self, label, color):
        """Draw mode label at top of screen."""
        banner_rect = pygame.Rect(0, 0, VC.WINDOW_WIDTH, 50)
        pygame.draw.rect(self.screen, (20, 20, 20), banner_rect)
        pygame.draw.rect(self.screen, color, pygame.Rect(0, 0, 6, 50))

        text = self.big_font.render(label, True, color)
        self.screen.blit(text, (20, 12))

    def _draw_metrics(
        self, step, total_steps, total_waiting, avg_queue, spawned, inter
    ):
        """Draw metrics panel in bottom left."""
        queues = inter.get_queue_lengths()
        phase = inter.traffic_light.phase.name

        lines = [
            f"Step     : {step:>5} / {total_steps}",
            f"Phase    : {phase}",
            f"Waiting  : {total_waiting:>10,.0f}",
            f"Avg Queue: {avg_queue:>8.1f} vehicles",
            f"Spawned  : {spawned:>5}",
            "",
            f"N:{queues['N']:>3}  S:{queues['S']:>3}",
            f"E:{queues['E']:>3}  W:{queues['W']:>3}",
        ]

        panel_h = len(lines) * 22 + 16
        panel_w = 240
        panel_y = VC.WINDOW_HEIGHT - panel_h - 10

        # Background
        surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        surf.fill((20, 20, 20, 200))
        self.screen.blit(surf, (10, panel_y))

        for i, line in enumerate(lines):
            text = self.font.render(line, True, VC.COLOR_TEXT)
            self.screen.blit(text, (18, panel_y + 8 + i * 22))

    def _draw_progress_bar(self, step, total_steps):
        """Draw progress bar at bottom of screen."""
        bar_h = 8
        bar_y = VC.WINDOW_HEIGHT - bar_h
        progress = step / total_steps
        bar_w = int(VC.WINDOW_WIDTH * progress)

        pygame.draw.rect(
            self.screen, (40, 40, 40), pygame.Rect(0, bar_y, VC.WINDOW_WIDTH, bar_h)
        )
        pygame.draw.rect(
            self.screen, (100, 200, 100), pygame.Rect(0, bar_y, bar_w, bar_h)
        )

    def _show_results_screen(self, fixed_stats, rl_stats):
        """Show final comparison screen after both modes complete."""
        waiting = True
        while waiting:
            if self._check_quit():
                break

            self.screen.fill((15, 15, 25))

            # Title
            title = self.big_font.render(
                "RESULTS — RL vs Fixed Timer", True, (255, 255, 255)
            )
            self.screen.blit(title, (VC.WINDOW_WIDTH // 2 - title.get_width() // 2, 60))

            # Calculate improvement
            wait_improvement = (
                (fixed_stats["total_waiting"] - rl_stats["total_waiting"])
                / fixed_stats["total_waiting"]
                * 100
            )
            queue_improvement = (
                (fixed_stats["avg_queue"] - rl_stats["avg_queue"])
                / fixed_stats["avg_queue"]
                * 100
            )

            # Results table
            rows = [
                ("Metric", "Fixed Timer", "RL Agent", "Improvement"),
                ("─" * 20, "─" * 12, "─" * 12, "─" * 12),
                (
                    "Total Waiting",
                    f"{fixed_stats['total_waiting']:,.0f}",
                    f"{rl_stats['total_waiting']:,.0f}",
                    f"{wait_improvement:.1f}% ↓",
                ),
                (
                    "Avg Queue",
                    f"{fixed_stats['avg_queue']:.1f}",
                    f"{rl_stats['avg_queue']:.1f}",
                    f"{queue_improvement:.1f}% ↓",
                ),
                (
                    "Vehicles Spawned",
                    f"{fixed_stats['spawned']}",
                    f"{rl_stats['spawned']}",
                    f"+{rl_stats['spawned'] - fixed_stats['spawned']}",
                ),
            ]

            colors_row = [
                (200, 200, 200),
                (80, 80, 80),
                (100, 200, 100),
                (100, 200, 100),
                (100, 200, 100),
            ]

            y = 150
            for i, (row, color) in enumerate(zip(rows, colors_row)):
                x_positions = [80, 300, 520, 720]
                for j, cell in enumerate(row):
                    # Color improvement column green
                    cell_color = color
                    if j == 3 and i > 1:
                        cell_color = (100, 255, 120)
                    text = self.font.render(str(cell), True, cell_color)
                    self.screen.blit(text, (x_positions[j], y))
                y += 40

            # Press any key message
            msg = self.small_font.render(
                "Press any key or close window to exit", True, (120, 120, 120)
            )
            self.screen.blit(
                msg,
                (VC.WINDOW_WIDTH // 2 - msg.get_width() // 2, VC.WINDOW_HEIGHT - 40),
            )

            pygame.display.flip()
            self.clock.tick(30)

            # Wait for keypress
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False
                if event.type == pygame.KEYDOWN:
                    waiting = False

    def _check_quit(self) -> bool:
        """Check if user wants to quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return True
        return False

    # ------------------------------------------------------------------ #
    #  DRAWING HELPERS (simplified from pygame_renderer)
    # ------------------------------------------------------------------ #

    def _draw_roads(self, intersection):
        cx = int(intersection.cx)
        cy = int(intersection.cy)
        road_w = SimConfig.LANE_WIDTH * SimConfig.LANES_PER_DIRECTION * 2

        pygame.draw.rect(
            self.screen,
            VC.COLOR_ROAD,
            pygame.Rect(cx - road_w // 2, 50, road_w, VC.WINDOW_HEIGHT),
        )
        pygame.draw.rect(
            self.screen,
            VC.COLOR_ROAD,
            pygame.Rect(0, cy - road_w // 2, VC.WINDOW_WIDTH, road_w),
        )

        # Center dashes
        dash, gap = 15, 10
        y = 50
        while y < VC.WINDOW_HEIGHT:
            pygame.draw.line(
                self.screen,
                VC.COLOR_LANE_MARKING,
                (cx, y),
                (cx, min(y + dash, VC.WINDOW_HEIGHT)),
                2,
            )
            y += dash + gap
        x = 0
        while x < VC.WINDOW_WIDTH:
            pygame.draw.line(
                self.screen,
                VC.COLOR_LANE_MARKING,
                (x, cy),
                (min(x + dash, VC.WINDOW_WIDTH), cy),
                2,
            )
            x += dash + gap

    def _draw_intersection(self, intersection):
        cx = int(intersection.cx)
        cy = int(intersection.cy)
        half = intersection.box_size // 2
        pygame.draw.rect(
            self.screen,
            VC.COLOR_INTERSECTION,
            pygame.Rect(
                cx - half, cy - half, intersection.box_size, intersection.box_size
            ),
        )

    def _draw_lights(self, intersection):
        from simulation.traffic_light import Phase

        cx = int(intersection.cx)
        cy = int(intersection.cy)
        phase = intersection.traffic_light.phase
        offset = intersection.box_size // 2 + 20
        radius = 10

        if phase == Phase.GREEN_NS:
            ns_col, ew_col = VC.COLOR_LIGHT_GREEN, VC.COLOR_LIGHT_RED
        elif phase == Phase.GREEN_EW:
            ns_col, ew_col = VC.COLOR_LIGHT_RED, VC.COLOR_LIGHT_GREEN
        elif phase == Phase.YELLOW_NS:
            ns_col, ew_col = VC.COLOR_LIGHT_YELLOW, VC.COLOR_LIGHT_RED
        elif phase == Phase.YELLOW_EW:
            ns_col, ew_col = VC.COLOR_LIGHT_RED, VC.COLOR_LIGHT_YELLOW
        else:
            ns_col = ew_col = VC.COLOR_LIGHT_RED

        for pos, col in [
            ((cx, cy - offset), ns_col),
            ((cx, cy + offset), ns_col),
            ((cx + offset, cy), ew_col),
            ((cx - offset, cy), ew_col),
        ]:
            pygame.draw.circle(self.screen, VC.COLOR_LIGHT_OFF, pos, radius + 3)
            pygame.draw.circle(self.screen, col, pos, radius)

    def _draw_vehicles(self, intersection):
        for lane_list in intersection.lanes.values():
            for lane in lane_list:
                for v in lane.vehicles:
                    if v.active:
                        self._draw_one_vehicle(v)
        for v in intersection.crossing_vehicles:
            if v.active:
                self._draw_one_vehicle(v)

    def _draw_one_vehicle(self, vehicle):
        rx, ry, rw, rh = vehicle.get_rect()
        color = (
            VC.COLOR_VEHICLE_NS
            if vehicle.direction in ("N", "S")
            else VC.COLOR_VEHICLE_EW
        )
        if vehicle.is_stopped:
            color = tuple(max(0, c - 50) for c in color)
        pygame.draw.rect(
            self.screen, color, (int(rx), int(ry), max(2, int(rw)), max(2, int(rh)))
        )


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  TRAFFIC RL — EXPO DEMO")
    print("  Press ESC or close window to exit")
    print("=" * 50 + "\n")

    demo = DemoRunner()
    demo.run()
