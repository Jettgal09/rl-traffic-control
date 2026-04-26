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

        # One spawner per intersection.
        # We pass grid_size so each spawner can tell which of its sides
        # face the outer boundary and therefore need actual spawn points.
        self.spawners = [
            VehicleSpawner(inter, self.map_width, self.map_height, grid_size=self.grid_size)
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
          2. Then update — lights tick, vehicles move, newly-crossed vehicles
             join their intersection's crossing_vehicles
          3. THEN handoff — vehicles that finished crossing an intersection
             get transplanted into the next intersection's approach lane
             (PHASE 2 only; for grid_size=1 this is a no-op)

        WHY SPAWN BEFORE UPDATE?
          If we updated first, newly spawned vehicles would miss their first
          movement step. Spawning first means every vehicle gets updated on
          the same step it appears.

        WHY HANDOFF AFTER UPDATE (NOT DURING)?
          If handoff ran inside an intersection's own update, a vehicle that
          just crossed A could immediately start moving through B's lane
          during the SAME step — that's two movements in one step, which
          desynchronizes the simulation. Running handoff after every
          intersection has updated guarantees each car moves exactly once
          per step.
        """
        self.step_count += 1

        # Try to spawn vehicles at each intersection's edges
        for spawner in self.spawners:
            new_vehicles = spawner.update()
            self.total_spawned += len(new_vehicles)

        # Update all intersections
        for inter in self.intersections:
            inter.update()

        # Hand off vehicles that finished crossing an intersection
        self._process_handoffs()

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

    def _get_neighbor(self, inter, direction: str):
        """
        Return the intersection that sits immediately in `direction` from `inter`,
        or None if we're on the grid boundary facing that way.

        PARAMETERS:
          inter     — the source intersection we're leaving from
          direction — "N"/"S"/"E"/"W", the direction a vehicle is traveling

        RETURNS:
          The neighboring Intersection object, or None if no neighbor exists
          (which means the vehicle is about to exit the grid entirely —
          the out-of-bounds cleanup in intersection.update() handles it from there).

        GRID LAYOUT REMINDER:
          row grows downward (row 0 is top), col grows rightward (col 0 is left).
          So a vehicle going NORTH ends up at row-1; SOUTH at row+1; etc.
        """
        row, col = inter.grid_pos
        if direction == "N":
            target_row, target_col = row - 1, col
        elif direction == "S":
            target_row, target_col = row + 1, col
        elif direction == "E":
            target_row, target_col = row, col + 1
        elif direction == "W":
            target_row, target_col = row, col - 1
        else:
            return None

        # Bounds-check against the grid
        if 0 <= target_row < self.grid_size and 0 <= target_col < self.grid_size:
            return self.intersections[target_row * self.grid_size + target_col]
        return None

    def _process_handoffs(self):
        """
        Transfer vehicles that have finished crossing one intersection into
        the next intersection's approach lane.

        WHY THIS EXISTS:
          Without handoff, a vehicle that crosses (0,0) eastbound on a green
          light just drifts east through (0,1) and (0,2) without either one
          ever noticing it — the "bridge / underpass" bug. Handoff re-registers
          each exiting vehicle into the next intersection's lane so it
          actually interacts with that intersection's signal.

        FOR EACH CROSSING VEHICLE:
          - Still inside the box (hasn't exited yet)
              → keep in the source intersection's crossing_vehicles
          - Exited the box AND a neighbor exists in that direction
              → hand off: append to neighbor's approach lane, remove from source
          - Exited the box BUT no neighbor (grid boundary exit)
              → keep in source crossing_vehicles; intersection.update's
                out-of-bounds cleanup will deactivate it when it finally
                leaves the map edge

        LANE INDEX PRESERVATION:
          A vehicle's (x, y) axis-perpendicular to its direction doesn't change
          as it moves, so its lane position is already exactly where the
          neighbor's same-direction same-lane-index approach expects it to be.
          No repositioning required — just re-registration.
        """
        for inter in self.intersections:
            still_crossing = []
            for vehicle in inter.crossing_vehicles:
                if not inter.has_vehicle_exited_box(vehicle):
                    # still mid-cross, not our problem this step
                    still_crossing.append(vehicle)
                    continue

                neighbor = self._get_neighbor(inter, vehicle.direction)
                if neighbor is None:
                    # we're on the outer boundary — let the car drift off-map
                    # and be cleaned up by the existing out-of-bounds logic
                    still_crossing.append(vehicle)
                    continue

                # handoff: drop the vehicle into the neighbor's approach lane.
                # It keeps its direction and lane_index, and its (x,y) is
                # already in the inter-intersection gap where an approaching
                # car should be for that lane.
                neighbor.add_vehicle_to_lane(vehicle, vehicle.direction, vehicle.lane)

            inter.crossing_vehicles = still_crossing

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
