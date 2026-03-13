# simulation/vehicle_spawner.py
#
# Creates new vehicles at the edges of the map every simulation step.
#
# Think of this as the "tap" that controls traffic flow.
# Higher SPAWN_RATE = more cars entering = heavier traffic.
#
# Each intersection has its own spawner, with spawn points
# positioned to line up with that intersection's lanes.

import random
from simulation.vehicle import Vehicle
from utils.config import SimConfig


class SpawnPoint:
    """
    Defines one location at the map edge where vehicles can appear.

    ATTRIBUTES:
      x, y        — pixel position where vehicle spawns
      direction   — which way the spawned vehicle will travel
      lane_index  — which lane (0 or 1) the vehicle enters
    """

    def __init__(self, x: float, y: float, direction: str, lane_index: int):
        self.x = x
        self.y = y
        self.direction = direction
        self.lane_index = lane_index

class VehicleSpawner:
    """
    Manages vehicle spawning for one intersection.

    Creates spawn points around the intersection's edges and
    randomly generates vehicles at those points each simulation step.
    """

    def __init__(self, intersection, map_width: int, map_height: int):
        """
        PARAMETERS:
          intersection — the Intersection object vehicles will drive toward
          map_width    — total map width in pixels (900)
          map_height   — total map height in pixels (900)
        """
        self.intersection = intersection
        self.map_width = map_width
        self.map_height = map_height
        self.spawn_rate = SimConfig.SPAWN_RATE
        self.total_spawned = 0

        # Create all spawn points for this intersection
        self.spawn_points = self._create_spawn_points()

    def _create_spawn_points(self) -> list:
        """
        Create spawn points at the 4 edges of the map.

        LANE POSITIONING:
          Lanes sit to the right of center from each driver's perspective.
          We calculate lane x/y positions using the intersection center
          and lane width from config.

          For N/S lanes (vertical roads):
            lane 0 sits at cx + 0.5 * lane_width  (right of center)
            lane 1 sits at cx + 1.5 * lane_width  (further right)

          For E/W lanes (horizontal roads):
            same idea but for y coordinate

        WHY SPAWN AT -20 or map_size+20?
          We spawn slightly outside the visible map so vehicles
          smoothly enter the screen rather than popping into existence.
        """
        spawn_points = []
        cx = self.intersection.cx
        cy = self.intersection.cy
        lw = SimConfig.LANE_WIDTH   # shorthand for lane width

        for i in range(SimConfig.LANES_PER_DIRECTION):
            # Each lane sits offset from center by (i + 0.5) * lane_width
            offset = (i + 0.5) * lw

            # --- SOUTH edge → northbound cars (traveling up, y decreasing) ---
            spawn_points.append(SpawnPoint(
                x=cx - offset,          # left of center for northbound
                y=self.map_height + 20, # just below bottom edge
                direction="N",
                lane_index=i
            ))

            # --- NORTH edge → southbound cars (traveling down, y increasing) ---
            spawn_points.append(SpawnPoint(
                x=cx + offset,          # right of center for southbound
                y=-20,                  # just above top edge
                direction="S",
                lane_index=i
            ))

            # --- WEST edge → eastbound cars (traveling right, x increasing) ---
            spawn_points.append(SpawnPoint(
                x=-20,                  # just left of left edge
                y=cy + offset,          # below center for eastbound
                direction="E",
                lane_index=i
            ))

            # --- EAST edge → westbound cars (traveling left, x decreasing) ---
            spawn_points.append(SpawnPoint(
                x=self.map_width + 20,  # just right of right edge
                y=cy - offset,          # above center for westbound
                direction="W",
                lane_index=i
            ))

        return spawn_points
    
    def update(self) -> list:
        """
        Try to spawn a vehicle at each spawn point.

        CALLED every simulation step by Road.

        FOR EACH spawn point:
          Roll a random number 0-1.
          If it's below spawn_rate → create a vehicle there.
          Otherwise → do nothing this step for that point.

        RETURNS:
          List of newly created Vehicle objects.
          May be empty if no vehicles spawned this step.
        """
        new_vehicles = []

        # Don't spawn if simulation is already at max capacity
        if self._count_active_vehicles() >= SimConfig.MAX_VEHICLES:
            return new_vehicles

        for sp in self.spawn_points:
            if random.random() < self.spawn_rate:
                # Create vehicle at this spawn point
                vehicle = Vehicle(
                    x=sp.x,
                    y=sp.y,
                    direction=sp.direction,
                    lane=sp.lane_index
                )

                # Add it to the correct lane in the intersection
                self.intersection.add_vehicle_to_lane(
                    vehicle,
                    sp.direction,
                    sp.lane_index
                )

                new_vehicles.append(vehicle)
                self.total_spawned += 1

        return new_vehicles

    def _count_active_vehicles(self) -> int:
        """
        Count all vehicles currently in the intersection's lanes.
        Used to enforce the MAX_VEHICLES cap.
        """
        total = 0
        for lane_list in self.intersection.lanes.values():
            for lane in lane_list:
                total += lane.total_vehicles()
        return total

    def set_spawn_rate(self, rate: float):
        """
        Change the spawn rate dynamically.

        Used in experiments to test different traffic densities:
          0.02 = light traffic
          0.05 = normal traffic (default)
          0.10 = heavy traffic

        The RL agent gets tested under all conditions to prove
        it learned a general policy, not just memorized one scenario.
        """
        self.spawn_rate = max(0.0, min(1.0, rate))

    def reset(self):
        """Reset spawn counter for a new RL episode."""
        self.total_spawned = 0