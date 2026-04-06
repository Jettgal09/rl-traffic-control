# simulation/lane.py
#
# Models a single traffic lane approaching an intersection.

from utils.config import SimConfig


class Lane:
    """
    One traffic lane approaching an intersection.

    ATTRIBUTES:
      direction  — which way vehicles in this lane travel ("N","S","E","W")
      lane_index — which lane this is (0 = first lane, 1 = second lane)
      stop_line  — the pixel coordinate where vehicles must stop on red
      vehicles   — ordered list of Vehicle objects (index 0 = front of queue)
    """

    def __init__(self, direction: str, lane_index: int, stop_line: float):
        self.direction = direction
        self.lane_index = lane_index
        self.stop_line = stop_line
        self.vehicles = []

    def add_vehicle(self, vehicle):
        self.vehicles.append(vehicle)

    def remove_vehicle(self, vehicle):
        if vehicle in self.vehicles:
            self.vehicles.remove(vehicle)

    def cleanup_inactive(self):
        self.vehicles = [v for v in self.vehicles if v.active]

    def update(self, light_is_green: bool):
        for i, vehicle in enumerate(self.vehicles):
            if not vehicle.active:
                continue

            space_ahead = self._has_space_ahead(i)

            if i == 0:
                at_stop = self._is_at_stop_line(vehicle)

                if at_stop and not light_is_green:
                    can_move = False
                else:
                    can_move = True
            else:
                can_move = True

            vehicle.update(can_move=can_move, space_ahead=space_ahead)

    def _has_space_ahead(self, vehicle_index: int) -> bool:
        if vehicle_index == 0:
            return True

        vehicle = self.vehicles[vehicle_index]
        leader = self.vehicles[vehicle_index - 1]

        min_gap = SimConfig.VEHICLE_LENGTH * 1.5

        if self.direction == "N":
            gap = vehicle.y - leader.y
        elif self.direction == "S":
            gap = leader.y - vehicle.y
        elif self.direction == "E":
            gap = leader.x - vehicle.x
        elif self.direction == "W":
            gap = vehicle.x - leader.x
        else:
            return True

        return gap > min_gap

    def _is_at_stop_line(self, vehicle) -> bool:
        threshold = 30.0

        if self.direction == "N":
            return vehicle.y <= self.stop_line + threshold
        elif self.direction == "S":
            return vehicle.y >= self.stop_line - threshold
        elif self.direction == "E":
            return vehicle.x >= self.stop_line - threshold
        elif self.direction == "W":
            return vehicle.x <= self.stop_line + threshold
        return False

    def queue_length(self) -> int:
        return sum(1 for v in self.vehicles if v.is_stopped)

    def total_vehicles(self) -> int:
        return len([v for v in self.vehicles if v.active])

    def total_waiting_time(self) -> float:
        return sum(v.waiting_time for v in self.vehicles if v.active)
