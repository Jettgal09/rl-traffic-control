# simulation/road.py
#
# Top level container for the entire simulation world.
#
# This is what the RL environment talks to directly.
# It owns all intersections and spawners, and its step()
# method advances the entire world by one simulation step.
#
# PHASE 1: Road contains 1 intersection
# PHASE 2: Road contains 9 intersections in a 3x3 grid
#
# The RL environment never talks to intersections directly —
# it always goes through Road. This means switching from
# Phase 1 to Phase 2 only requires changing Road, not the env.

from simulation.intersection import Intersection
from simulation.vehicle_spawner import VehicleSpawner
from utils.config import SimConfig, VisualizationConfig


class Road:
    """
    The complete simulation world.

    Contains all intersections and manages the step-by-step loop.
    """

    def __init__(self, grid_size: int = None):
        """
        PARAMETERS:
          grid_size — override SimConfig.GRID_SIZE if needed
                      useful for running experiments with different sizes
        """
        self.grid_size = grid_size or SimConfig.GRID_SIZE
        self.map_width = VisualizationConfig.WINDOW_WIDTH
        self.map_height = VisualizationConfig.WINDOW_HEIGHT

        # Create all intersections in a grid layout
        self.intersections = self._create_intersections()

        # One spawner per intersection
        self.spawners = [
            VehicleSpawner(inter, self.map_width, self.map_height)
            for inter in self.intersections
        ]

        self.step_count = 0
        self.total_spawned = 0

    def _create_intersections(self) -> list:
        """
        Create intersections arranged in a grid.

        For GRID_SIZE=1: one intersection at the center of the map.

        For GRID_SIZE=3: 9 intersections evenly spaced.
        The whole grid is centered on the map.

        We store them as a flat list.
        To get intersection at row r, col c:
          intersections[r * grid_size + c]
        """
        intersections = []
        g = self.grid_size

        # Calculate top-left starting position so grid is centered
        total_width = (g - 1) * SimConfig.CELL_SIZE
        total_height = (g - 1) * SimConfig.CELL_SIZE
        start_x = (self.map_width - total_width) / 2
        start_y = (self.map_height - total_height) / 2

        for row in range(g):
            for col in range(g):
                cx = start_x + col * SimConfig.CELL_SIZE
                cy = start_y + row * SimConfig.CELL_SIZE
                inter = Intersection(cx, cy, grid_pos=(row, col))
                intersections.append(inter)

        return intersections

    def step(self):
        """
        Advance the entire simulation by one time step.

        ORDER MATTERS:
          1. Spawn first — new vehicles enter the lanes
          2. Then update — lights tick, vehicles move

        Why spawn before update?
          If we updated first, newly spawned vehicles would miss
          their first movement step. Spawning first means every
          vehicle gets updated on the same step it appears.
        """
        self.step_count += 1

        # Try to spawn vehicles at each intersection's edges
        for spawner in self.spawners:
            new_vehicles = spawner.update()
            self.total_spawned += len(new_vehicles)

        # Update all intersections
        for inter in self.intersections:
            inter.update()

    def apply_action(self, action: int, intersection_idx: int = 0):
        """
        Apply RL agent's action to one intersection.

        ACTION SPACE:
          0 → set green duration to 20 steps (short — high traffic)
          1 → set green duration to 40 steps (medium)
          2 → set green duration to 60 steps (long)
          3 → set green duration to 80 steps (very long — low traffic)

        Agent sets duration at start of each green phase.
        Light automatically switches after that duration.
        """
        from simulation.traffic_light import DURATION_OPTIONS

        if intersection_idx < len(self.intersections):
            inter = self.intersections[intersection_idx]
            if inter.is_start_of_green_phase():
                duration = DURATION_OPTIONS[action]
                inter.set_phase_duration(duration)

    def reset(self):
        """
        Reset the entire simulation to initial state.
        Called at the start of every new RL episode.
        """
        for inter in self.intersections:
            inter.reset()
        for spawner in self.spawners:
            spawner.reset()
        self.step_count = 0
        self.total_spawned = 0

    def get_observation(self, intersection_idx: int = 0) -> list:
        """Get observation vector for one intersection."""
        return self.intersections[intersection_idx].get_observation_vector()

    def get_total_waiting_time(self) -> float:
        """Total waiting time across all vehicles in all intersections."""
        return sum(inter.get_total_waiting_time() for inter in self.intersections)

    def get_total_queue_length(self) -> int:
        """Total stopped vehicles across all intersections."""
        return sum(inter.get_total_queue_length() for inter in self.intersections)

    def get_metrics(self) -> dict:
        """
        All key metrics in one dictionary.
        Used by the RL environment to calculate reward and log results.
        """
        return {
            "step": self.step_count,
            "total_waiting_time": self.get_total_waiting_time(),
            "total_queue_length": self.get_total_queue_length(),
            "total_spawned": self.total_spawned,
        }

    def __repr__(self):
        return (
            f"Road(grid={self.grid_size}x{self.grid_size}, "
            f"intersections={len(self.intersections)}, "
            f"step={self.step_count})"
        )
