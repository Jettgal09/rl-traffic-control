# simulation/vehicle.py
#
# Models a single vehicle traveling through the simulation.
#
# Vehicles are the "objects" the RL agent is indirectly helping.
# The agent never controls vehicles directly — it controls traffic lights,
# which then affect whether vehicles can move or must wait.

from utils.config import SimConfig


class Vehicle:
    """
    Represents one car in the simulation.

    DIRECTION CONVENTION — how directions map to screen coordinates:
      "N" = traveling North = moving UP on screen    = y decreases each step
      "S" = traveling South = moving DOWN on screen  = y increases each step
      "E" = traveling East  = moving RIGHT on screen = x increases each step
      "W" = traveling West  = moving LEFT on screen  = x decreases each step
    """

    def __init__(self, x: float, y: float, direction: str, lane: int = 0):
        """
        PARAMETERS:
          x, y      — starting position in pixels
          direction — which way this vehicle travels: "N", "S", "E", or "W"
          lane      — which lane it's in (0 = first lane, 1 = second lane)
        """
        # --- Position ---
        self.x = float(x)
        self.y = float(y)
        self.direction = direction
        self.lane = lane

        # --- Movement ---
        self.speed = SimConfig.VEHICLE_SPEED  # current speed (0 when stopped)
        self.is_stopped = False  # True when waiting at red light

        # --- Metrics ---
        # waiting_time is the most important metric in this whole project.
        # Every step this vehicle spends NOT moving, this goes up by 1.
        # The RL agent's reward is based on minimizing total waiting time
        # across ALL vehicles in the simulation.
        self.waiting_time = 0

        # How many total steps since this vehicle was spawned
        self.travel_time = 0

        # --- Lifecycle ---
        # When a vehicle drives off the edge of the map, we set this to False
        # and remove it from the simulation
        self.active = True

        # --- Size (for rendering) ---
        self.length = SimConfig.VEHICLE_LENGTH
        self.width = SimConfig.VEHICLE_WIDTH

    def update(self, can_move: bool, space_ahead: bool = True):
        """
        Advance this vehicle by one simulation step.

        PARAMETERS:
          can_move    — True if the traffic light is green for this vehicle
          space_ahead — True if no vehicle is blocking the path ahead

        Both must be True to move. If either is False, vehicle stops
        and waiting_time increases by 1.
        """
        # Always count total time since spawning
        self.travel_time += 1

        if can_move and space_ahead:
            # Both conditions met — vehicle moves forward
            self._move()
            self.is_stopped = False
        else:
            # Either red light or blocked — vehicle stops
            self.speed = 0
            self.is_stopped = True
            self.waiting_time += 1

    def _move(self):
        """
        Move the vehicle forward one step based on its direction.

        Remember the coordinate system:
          North → y decreases  (moving up on screen)
          South → y increases  (moving down on screen)
          East  → x increases  (moving right)
          West  → x decreases  (moving left)
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
        """
        Returns True if this vehicle has driven off the edge of the map.
        When this happens the vehicle gets removed from the simulation
        and counted as successfully completed its journey.

        We use 50px buffer so vehicles fully exit before being removed.
        """
        return (self.x < -50 or self.x > map_width + 50 or
                self.y < -50 or self.y > map_height + 50)

    def deactivate(self):
        """Mark this vehicle as finished — it has exited the map."""
        self.active = False

    def get_rect(self):
        """
        Returns the vehicle's bounding box as (x, y, width, height).
        Used by the Pygame renderer to draw the vehicle as a rectangle.

        Vehicles traveling North/South are taller than wide.
        Vehicles traveling East/West are wider than tall.
        This makes them look like they're actually moving in that direction.
        """
        if self.direction in ("N", "S"):
            w, h = self.width, self.length   # tall rectangle
        else:
            w, h = self.length, self.width   # wide rectangle

        # Pygame draws rects from top-left corner, but our x,y is the center
        # So we subtract half width and half height to get top-left
        return (self.x - w/2, self.y - h/2, w, h)