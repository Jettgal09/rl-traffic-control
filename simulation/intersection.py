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
from utils.config import SimConfig, VisualizationConfig


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
        self.traffic_light.update()

        ns_green = self.traffic_light.is_green_for("N")
        ew_green = self.traffic_light.is_green_for("E")

        for direction, lane_list in self.lanes.items():
            if direction in ("N", "S"):
                light_is_green = ns_green
            else:
                light_is_green = ew_green
            for lane in lane_list:
                lane.update(light_is_green)

        # Move vehicles that crossed stop line into crossing list
        self._process_lane_exits()

        # Update crossing vehicles — they always move, never stop
        for vehicle in self.crossing_vehicles:
            vehicle.update(can_move=True, space_ahead=True)

        self._cleanup_inactive()

    def _process_lane_exits(self):
        """
        Check if the front vehicle in any lane has reached the
        stop line with a green light — if so move it into
        crossing_vehicles so it passes through without stopping.
        """
        for direction, lane_list in self.lanes.items():
            light_green = self.traffic_light.is_green_for(direction)
            if not light_green:
                continue

            for lane in lane_list:
                if not lane.vehicles:
                    continue

                front = lane.vehicles[0]
                if self._past_stop_line(front, direction):
                    lane.vehicles.pop(0)
                    self.crossing_vehicles.append(front)

    def _past_stop_line(self, vehicle, direction: str) -> bool:
        """Returns True if vehicle has crossed the stop line."""
        half_box = self.box_size / 2
        if direction == "N":
            return vehicle.y <= self.cy + half_box
        elif direction == "S":
            return vehicle.y >= self.cy - half_box
        elif direction == "E":
            return vehicle.x >= self.cx - half_box
        elif direction == "W":
            return vehicle.x <= self.cx + half_box
        return False

    def _cleanup_inactive(self):
        """Remove vehicles that have driven off the map."""
        # Clean lanes
        for lane_list in self.lanes.values():
            for lane in lane_list:
                lane.cleanup_inactive()

        # Check crossing vehicles against map bounds
        still_crossing = []
        for vehicle in self.crossing_vehicles:
            if vehicle.is_out_of_bounds(
                VisualizationConfig.WINDOW_WIDTH, VisualizationConfig.WINDOW_HEIGHT
            ):
                vehicle.deactivate()
                self.total_vehicles_passed += 1
            else:
                still_crossing.append(vehicle)
        self.crossing_vehicles = still_crossing

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

        # Normalize phase timer — cap at 100 steps
        normalized_timer = min(self.traffic_light.phase_timer / 100.0, 1.0)


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
            normalized_timer,
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
