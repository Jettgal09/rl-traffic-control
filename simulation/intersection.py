# simulation/intersection.py
#
# Models a complete 4-way intersection.

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

        self.box_size = SimConfig.LANE_WIDTH * SimConfig.LANES_PER_DIRECTION * 2

        # One traffic light controls this entire intersection
        self.traffic_light = TrafficLight()
        self.lanes = self._create_lanes()
        self.crossing_vehicles = []
        self.total_vehicles_passed = 0

    def _create_lanes(self) -> dict:
        """
        Create 8 lanes — 2 for each of the 4 directions.
        """
        half_box = self.box_size / 2

        # Stop line for each direction
        stop_lines = {
            "N": self.cy + half_box,  
            "S": self.cy - half_box,  
            "E": self.cx - half_box,  
            "W": self.cx + half_box, 
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

        self._process_lane_exits()

        for vehicle in self.crossing_vehicles:
            vehicle.update(can_move=True, space_ahead=True)

        self._cleanup_inactive()

    def _process_lane_exits(self):
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
        for lane_list in self.lanes.values():
            for lane in lane_list:
                lane.cleanup_inactive()

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
        return {
            direction: sum(lane.queue_length() for lane in lane_list)
            for direction, lane_list in self.lanes.items()
        }

    def get_total_waiting_time(self) -> float:
        total = 0
        for lane_list in self.lanes.values():
            for lane in lane_list:
                total += lane.total_waiting_time()
        return total

    def get_total_queue_length(self) -> int:
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

    def set_phase_duration(self, duration: int):
        """
        Called by RL agent to set how long the current green phase lasts.
        Only has effect at the start of a green phase.
        """
        self.traffic_light.set_green_duration(duration)

    def is_start_of_green_phase(self) -> bool:
        """Returns True on first step of a new green phase."""
        return self.traffic_light.is_start_of_green_phase()

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
