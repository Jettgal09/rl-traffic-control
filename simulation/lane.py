# simulation/lane.py
#
# Models a single traffic lane approaching an intersection.
#
# Think of a lane as a queue — vehicles join at the back and
# leave from the front when the light turns green.
# The queue length tells us how congested this lane is,
# which is one of the key things the RL agent observes.

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
        """
        PARAMETERS:
          direction  — "N", "S", "E", or "W"
          lane_index — 0 or 1 (we have 2 lanes per direction)
          stop_line  — pixel position of the stop line.
                       For N/S lanes this is a y coordinate.
                       For E/W lanes this is an x coordinate.
        """
        self.direction = direction
        self.lane_index = lane_index
        self.stop_line = stop_line
        self.vehicles = []  # front of queue is at index 0

    def add_vehicle(self, vehicle):
        """
        Add a vehicle to the back of this lane's queue.
        New vehicles always join at the back — just like a real queue.
        """
        self.vehicles.append(vehicle)

    def remove_vehicle(self, vehicle):
        """Remove a vehicle that has left this lane."""
        if vehicle in self.vehicles:
            self.vehicles.remove(vehicle)

    def cleanup_inactive(self):
        """
        Remove all vehicles that are no longer active.
        Called every simulation step to keep the list clean.
        """
        self.vehicles = [v for v in self.vehicles if v.active]

    def update(self, light_is_green: bool):
        """
        Update all vehicles in this lane for one simulation step.
        """
        for i, vehicle in enumerate(self.vehicles):
            if not vehicle.active:
                continue

            space_ahead = self._has_space_ahead(i)

            if i == 0:
                # Front vehicle — check if it's at the stop line
                at_stop = self._is_at_stop_line(vehicle)

                if at_stop and not light_is_green:
                    # At the stop line AND light is red — must stop
                    can_move = False
                else:
                    # Either not at stop line yet, OR light is green
                    # Either way — can move forward
                    can_move = True
            else:
                # Vehicles behind just follow the car ahead
                # They move freely until they get too close to the car in front
                can_move = True

            vehicle.update(can_move=can_move, space_ahead=space_ahead)

    def _has_space_ahead(self, vehicle_index: int) -> bool:
        """
        Check if there is enough gap between this vehicle and the one ahead.

        The front vehicle (index 0) always has space ahead — nothing is
        blocking it except the traffic light.

        For all other vehicles, we measure the gap to the vehicle in front.
        If the gap is smaller than 1.5x vehicle length, they must stop.
        We use 1.5x so vehicles don't drive into each other's back bumper.
        """
        if vehicle_index == 0:
            return True  # Nothing blocking the front vehicle

        vehicle = self.vehicles[vehicle_index]
        leader = self.vehicles[vehicle_index - 1]  # vehicle directly ahead

        min_gap = SimConfig.VEHICLE_LENGTH * 1.5

        # Calculate gap based on direction of travel
        if self.direction == "N":
            # Both moving up (y decreasing). Leader has smaller y value.
            # Gap = how far apart their y positions are.
            gap = vehicle.y - leader.y
        elif self.direction == "S":
            # Both moving down (y increasing). Leader has larger y value.
            gap = leader.y - vehicle.y
        elif self.direction == "E":
            # Both moving right (x increasing). Leader has larger x value.
            gap = leader.x - vehicle.x
        elif self.direction == "W":
            # Both moving left (x decreasing). Leader has smaller x value.
            gap = vehicle.x - leader.x
        else:
            return True

        return gap > min_gap

    def _is_at_stop_line(self, vehicle) -> bool:
        """
        Returns True if the vehicle is close enough to the stop line
        that it should obey the traffic light.

        We use a 30px threshold — if the vehicle is within 30px of the
        stop line, it's considered "at" the stop line.

        Why 30px and not exactly at the line?
        Because vehicles move 2px per step — if we checked for exact
        position we might miss it entirely and the vehicle would drive through.
        """
        threshold = 30.0

        if self.direction == "N":
            # Northbound vehicles approach from below — stop line is above them
            # Vehicle is "at" the line when its y is close to stop_line
            return vehicle.y <= self.stop_line + threshold
        elif self.direction == "S":
            # Southbound vehicles approach from above
            return vehicle.y >= self.stop_line - threshold
        elif self.direction == "E":
            # Eastbound vehicles approach from the left
            return vehicle.x >= self.stop_line - threshold
        elif self.direction == "W":
            # Westbound vehicles approach from the right
            return vehicle.x <= self.stop_line + threshold
        return False

    def queue_length(self) -> int:
        """
        How many vehicles are currently stopped in this lane.

        This is a direct measure of congestion.
        The RL agent uses this in its observation:
        'north lane has 8 cars stopped, east lane has 1'
        → agent should prefer giving north a green light.
        """
        return sum(1 for v in self.vehicles if v.is_stopped)

    def total_vehicles(self) -> int:
        """Total vehicles in this lane — both moving and stopped."""
        return len([v for v in self.vehicles if v.active])

    def total_waiting_time(self) -> float:
        """
        Sum of waiting_time across all vehicles in this lane.
        Used to calculate the RL reward — higher means more congestion.
        """
        return sum(v.waiting_time for v in self.vehicles if v.active)
