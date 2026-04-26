# simulation/vehicle_spawner.py
#
# Creates new vehicles at the OUTER edges of the grid every simulation step.
#
# Think of this as the "tap" that controls traffic flow.
# Higher SPAWN_RATE = more cars entering = heavier traffic.
#
# PHASE 1 (grid_size = 1): the lone intersection IS the whole grid, so all
#                          four sides face outward — every spawner builds
#                          8 spawn points (2 lanes × 4 directions).
# PHASE 2 (grid_size > 1): only intersections on the outer ring spawn cars.
#                          Interior intersections receive their traffic via
#                          handoff from neighbors (handled in road.py), not
#                          via spawning. So an interior spawner ends up
#                          with zero spawn points and is effectively a no-op.
#
# Each intersection has its own spawner, but the spawner now asks
# "which of MY sides face the open world?" before laying down spawn points.

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

    def __init__(self, intersection, map_width: int, map_height: int, grid_size: int = 1):
        """
        PARAMETERS:
          intersection — the Intersection object vehicles will drive toward
          map_width    — total map width in pixels (used for Phase 1 spawn distance)
          map_height   — total map height in pixels (used for Phase 1 spawn distance)
          grid_size    — how many intersections per side of the grid.
                         We need this to figure out which sides of THIS
                         intersection face outward (no neighbor) and therefore
                         deserve a spawn point. Defaults to 1 so existing
                         Phase 1 callers don't need to change.
        """
        self.intersection = intersection
        self.map_width = map_width
        self.map_height = map_height
        self.grid_size = grid_size
        self.spawn_rate = SimConfig.SPAWN_RATE
        self.total_spawned = 0

        # Create all spawn points for this intersection
        self.spawn_points = self._create_spawn_points()

    def _create_spawn_points(self) -> list:
        """
        Build spawn points only on the OUTWARD-FACING sides of this intersection.

        WHY ONLY OUTWARD?
          Cars from a neighboring intersection arrive via handoff in Road,
          not via spawning. Spawning on an inward edge would create duplicate
          cars overlapping with handed-off traffic and wreck the density cap.

        GRID GEOMETRY (row, col) → outward sides:
          row 0           → top of grid    → NORTH side is outward → spawn SOUTHBOUND cars
          row grid_size-1 → bottom of grid → SOUTH side is outward → spawn NORTHBOUND cars
          col 0           → left of grid   → WEST side is outward  → spawn EASTBOUND cars
          col grid_size-1 → right of grid  → EAST side is outward  → spawn WESTBOUND cars

          For grid_size=1, every row/col index is both 0 AND grid_size-1, so all
          four sides count as outward and we end up with all 8 spawn points —
          byte-for-byte the same as Phase 1.

        LANE POSITIONING:
          Lanes sit to the right of center from each driver's perspective.
          We calculate lane x/y positions using the intersection center
          and lane width from config.

          For N/S lanes (vertical roads):
            lane 0 sits at cx + 0.5 * lane_width  (right of center)
            lane 1 sits at cx + 1.5 * lane_width  (further right)

          For E/W lanes (horizontal roads):
            same idea but for y coordinate

        WHY SPAWN AT THE MAP EDGE?
          Cars spawn just outside the visible map (y=-20, y=map_height+20,
          x=-20, x=map_width+20) so they appear to drive IN from off-screen
          rather than popping into existence mid-road.
          This works for ANY grid size because lane positions are tied to
          the specific intersection's (cx, cy) — a car spawned at the west
          map edge with y=cy+offset stays in this intersection's row, and
          the FIRST intersection it encounters along that row is this one
          (since we only spawn here if there's no neighbor to the west,
          i.e. we ARE the leftmost intersection in our row).
          Same reasoning for the other three sides.
        """
        spawn_points = []
        cx = self.intersection.cx
        cy = self.intersection.cy
        lw = SimConfig.LANE_WIDTH   # shorthand for lane width
        row, col = self.intersection.grid_pos
        g = self.grid_size

        # --- figure out which sides face the open world ---
        has_neighbor_north = row > 0
        has_neighbor_south = row < g - 1
        has_neighbor_west  = col > 0
        has_neighbor_east  = col < g - 1

        # --- where each outward spawn sits in pixel space ---
        # Always just outside the map edge, regardless of grid size.
        # For grid_size=1 this is the original Phase 1 behavior, unchanged.
        # For grid_size>1 this gives cars the same long approach distance
        # as Phase 1, which is both visually clean (cars drive in from
        # off-screen) and dynamically consistent with what the agent
        # learned to expect.
        north_spawn_y = -20                   # above top of map
        south_spawn_y = self.map_height + 20  # below bottom of map
        west_spawn_x  = -20                   # left of left edge
        east_spawn_x  = self.map_width + 20   # right of right edge

        for i in range(SimConfig.LANES_PER_DIRECTION):
            # Each lane sits offset from center by (i + 0.5) * lane_width
            offset = (i + 0.5) * lw

            # --- SOUTH edge → northbound cars (traveling up, y decreasing) ---
            if not has_neighbor_south:
                spawn_points.append(SpawnPoint(
                    x=cx - offset,          # left of center for northbound
                    y=south_spawn_y,
                    direction="N",
                    lane_index=i
                ))

            # --- NORTH edge → southbound cars (traveling down, y increasing) ---
            if not has_neighbor_north:
                spawn_points.append(SpawnPoint(
                    x=cx + offset,          # right of center for southbound
                    y=north_spawn_y,
                    direction="S",
                    lane_index=i
                ))

            # --- WEST edge → eastbound cars (traveling right, x increasing) ---
            if not has_neighbor_west:
                spawn_points.append(SpawnPoint(
                    x=west_spawn_x,
                    y=cy + offset,          # below center for eastbound
                    direction="E",
                    lane_index=i
                ))

            # --- EAST edge → westbound cars (traveling left, x decreasing) ---
            if not has_neighbor_east:
                spawn_points.append(SpawnPoint(
                    x=east_spawn_x,
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