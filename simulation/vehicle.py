# simulation/vehicle.py
#
# Models a single vehicle traveling through the simulation.
# The agent never controls vehicles directly — it controls traffic lights
from utils.config import SimConfig


class Vehicle:
    """
    Represents one car in the simulation.
    """

    def __init__(self, x: float, y: float, direction: str, lane: int = 0):
        # --- Position ---
        self.x = float(x)
        self.y = float(y)
        self.direction = direction
        self.lane = lane

        # --- Movement ---
        self.speed = SimConfig.VEHICLE_SPEED
        self.is_stopped = False

        # --- Metrics ---
        self.waiting_time = 0
        self.travel_time = 0

        # --- Lifecycle ---
        self.active = True

        # --- Size (for rendering) ---
        self.length = SimConfig.VEHICLE_LENGTH
        self.width = SimConfig.VEHICLE_WIDTH

    def update(self, can_move: bool, space_ahead: bool = True):
        """
        Advance this vehicle by one simulation step.
        """
        self.travel_time += 1

        if can_move and space_ahead:
            self._move()
            self.is_stopped = False
        else:
            self.speed = 0
            self.is_stopped = True
            self.waiting_time += 1

    def _move(self):
        """
        Move the vehicle forward one step based on its direction.
        """
        self.speed = SimConfig.VEHICLE_SPEED

        if self.direction == "N":
            self.y -= self.speed
        elif self.direction == "S":
            self.y += self.speed
        elif self.direction == "E":
            self.x += self.speed
        elif self.direction == "W":
            self.x -= self.speed

    def is_out_of_bounds(self, map_width: int, map_height: int) -> bool:
        return self.x < -50 or self.x > map_width + 50 or self.y < -50 or self.y > map_height + 50

    def deactivate(self):
        self.active = False
        
    def get_rect(self):
        if self.direction in ("N", "S"):
            w, h = self.width, self.length
        else:
            w, h = self.length, self.width
        return (self.x - w / 2, self.y - h / 2, w, h)
