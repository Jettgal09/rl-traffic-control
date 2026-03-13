# simulation/intersection.py
#
# Models a complete 4-way intersection.
#
# This is the coordinator class — it owns:
#   - One TrafficLight (controls who gets green)
#   - 8 Lane objects (4 directions × 2 lanes each)
#
# Every simulation step, the intersection:
#   1. Updates the traffic light
#   2. Tells each lane whether it has green or red
#   3. Collects metrics (queue lengths, waiting times) for the RL agent
#
# The RL agent talks TO the intersection (to request phase switches)
# and reads FROM the intersection (to observe traffic state).

from simulation.traffic_light import TrafficLight
from simulation.lane import Lane
from utils.config import SimConfig


class Intersection:
    """
    A 4-way intersection with traffic signals and vehicle queues.

    PARAMETERS:
      cx, cy    — center position of this intersection in pixels
      grid_pos  — (row, col) position in the city grid
                  always (0,0) for Phase 1 single intersection
    """

    def __init__(self, cx: float, cy: float, grid_pos=(0, 0)):
        self.cx = cx  # center x coordinate in pixels
        self.cy = cy  # center y coordinate in pixels
        self.grid_pos = grid_pos

        # The intersection box is the square area where roads cross.
        # Its size depends on how many lanes we have.
        # 2 lanes each way × 2 directions × lane width = box size
        self.box_size = SimConfig.LANE_WIDTH * SimConfig.LANES_PER_DIRECTION * 2

        # One traffic light controls this entire intersection
        self.traffic_light = TrafficLight()

        # Create all 8 lanes (4 directions × 2 lanes each)
        # stored as a dict: {"N": [Lane, Lane], "S": [Lane, Lane], ...}
        self.lanes = self._create_lanes()

        # Vehicles that have passed the stop line and are crossing
        # the intersection box — these always move, never stop
        self.crossing_vehicles = []

        # Track how many vehicles successfully passed through
        self.total_vehicles_passed = 0

    def _create_lanes(self) -> dict:
        """
        Create 8 lanes — 2 for each of the 4 directions.

        Each lane gets a stop_line coordinate — the pixel position
        where vehicles must stop when the light is red.

        Stop lines sit at the edge of the intersection box.
        """
        half_box = self.box_size / 2

        # Stop line for each direction — explained above
        stop_lines = {
            "N": self.cy + half_box,  # northbound stops at south edge
            "S": self.cy - half_box,  # southbound stops at north edge
            "E": self.cx - half_box,  # eastbound stops at west edge
            "W": self.cx + half_box,  # westbound stops at east edge
        }

        lanes = {}
        for direction in ["N", "S", "E", "W"]:
            lanes[direction] = [
                Lane(direction, lane_index=i, stop_line=stop_lines[direction])
                for i in range(SimConfig.LANES_PER_DIRECTION)
            ]

        return lanes

    def update(self):
        """
        Advance the intersection by one simulation step.

        ORDER MATTERS here:
          1. Update the traffic light first — it might change phase
          2. Read the new phase — is NS green or EW green?
          3. Update all lanes with that green/red information
          4. Clean up vehicles that have exited
        """
        # Step 1 — tick the traffic light forward
        self.traffic_light.update()

        # Step 2 — which directions currently have green?
        ns_green = self.traffic_light.is_green_for("N")
        ew_green = self.traffic_light.is_green_for("E")

        # Step 3 — update each lane with its current light state
        for direction, lane_list in self.lanes.items():
            # N and S share the same light state
            # E and W share the same light state
            if direction in ("N", "S"):
                light_is_green = ns_green
            else:
                light_is_green = ew_green

            for lane in lane_list:
                lane.update(light_is_green)

        # Step 4 — clean up vehicles that have left the simulation
        self._cleanup_inactive()

    def _cleanup_inactive(self):
        """
        Remove vehicles that have driven off the map.
        Count each one as successfully passed through.
        """
        for lane_list in self.lanes.values():
            for lane in lane_list:
                lane.cleanup_inactive()

    def get_queue_lengths(self) -> dict:
        """
        Returns how many vehicles are stopped in each direction.
        This is the primary congestion signal for the RL agent.

        Example return: {"N": 3, "S": 0, "E": 7, "W": 1}
        The agent sees this and learns — east is badly congested,
        give east a green light soon.
        """
        return {
            direction: sum(lane.queue_length() for lane in lane_list)
            for direction, lane_list in self.lanes.items()
        }

    def get_total_waiting_time(self) -> float:
        """
        Sum of waiting_time across every vehicle in every lane.
        This feeds directly into the RL reward calculation.
        Higher = more congestion = worse reward for the agent.
        """
        total = 0
        for lane_list in self.lanes.values():
            for lane in lane_list:
                total += lane.total_waiting_time()
        return total

    def get_total_queue_length(self) -> int:
        """Total stopped vehicles across all directions."""
        return sum(self.get_queue_lengths().values())

    def get_observation_vector(self) -> list:
        """
        Builds the list of numbers the RL agent will observe.

        This is what the agent's neural network actually sees —
        not cars, not roads, just a list of normalized numbers.

        We normalize by dividing by MAX_VEHICLES so all values
        stay between 0 and 1. Neural networks work much better
        with small normalized inputs than with raw large numbers.

        Returns a list of 9 floats:
          [N_queue, S_queue, E_queue, W_queue,
           N_total, S_total, E_total, W_total,
           current_phase]
        """
        max_v = SimConfig.MAX_VEHICLES
        queues = self.get_queue_lengths()

        # Total vehicles per direction (stopped + moving)
        counts = {
            direction: sum(lane.total_vehicles() for lane in lane_list)
            for direction, lane_list in self.lanes.items()
        }

        obs = [
            queues["N"] / max_v,
            queues["S"] / max_v,
            queues["E"] / max_v,
            queues["W"] / max_v,
            counts["N"] / max_v,
            counts["S"] / max_v,
            counts["E"] / max_v,
            counts["W"] / max_v,
            self.traffic_light.phase / 5.0,  # normalize 0-5 to 0-1
        ]
        return obs

    def request_phase_switch(self):
        """
        Called by the RL agent when it decides to switch phases.
        Just passes the request down to the traffic light.
        """
        self.traffic_light.request_switch()

    def add_vehicle_to_lane(self, vehicle, direction: str, lane_index: int = 0):
        """
        Add a spawned vehicle into a specific lane.
        Called by VehicleSpawner when a new car appears.
        """
        if direction in self.lanes:
            if lane_index < len(self.lanes[direction]):
                self.lanes[direction][lane_index].add_vehicle(vehicle)

    def reset(self):
        """Reset everything for a new RL episode."""
        self.traffic_light.reset()
        for lane_list in self.lanes.values():
            for lane in lane_list:
                lane.vehicles.clear()
        self.crossing_vehicles.clear()
        self.total_vehicles_passed = 0
