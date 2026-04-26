# simulation/intersection.py
#
# Models a complete 4-way intersection.

import random

from simulation.traffic_light import TrafficLight
from simulation.lane import Lane
from utils.config import SimConfig, VisualizationConfig


# TURN_MAP tells us which absolute direction corresponds to "left", "right",
# or "straight" for a driver currently heading in a given direction.
#
# The convention is driver's-perspective: if you're going NORTH, your left
# hand points WEST and your right hand points EAST. Same logic for the rest.
# This matches the right-side-driving lane layout baked into vehicle_spawner
# (NB lanes to the left of center, SB lanes to the right of center, etc.).
TURN_MAP = {
    "N": {"left": "W", "right": "E", "straight": "N"},
    "S": {"left": "E", "right": "W", "straight": "S"},
    "E": {"left": "N", "right": "S", "straight": "E"},
    "W": {"left": "S", "right": "N", "straight": "W"},
}


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
            # PHASE 2 — execute the pre-committed turn once the car is past
            # the box center. Order matters: we move FIRST, then check, so a
            # car that arrived at center this step still gets turned this
            # step (no one-step delay).
            self._maybe_execute_turn(vehicle)

        self._cleanup_inactive()

    def _process_lane_exits(self):
        """
        Promote cars from the lane queue into the crossing_vehicles list
        once they've passed the stop line on a green light.

        PHASE 2 ADDITION — turn assignment:
          The moment a car enters the box is also when it commits to a turn.
          We roll the three-way random here (straight/left/right) so the
          vehicle knows what it's doing before _maybe_execute_turn needs to
          act on it a few steps later. Rolling at entry (rather than at the
          stop line or at box-center) keeps _maybe_execute_turn pure — it
          just reads the flag, it doesn't decide anything.
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
                    # Commit this car to a turn at this intersection.
                    # If it was handed off from an upstream intersection, its
                    # old intended_turn was already consumed (set to None) when
                    # that turn executed — so we're always writing into a clean slate.
                    front.intended_turn = self._pick_turn()
                    self.crossing_vehicles.append(front)

    def _pick_turn(self) -> str:
        """
        Sample one of {"straight", "left", "right"} from the TURN_PROB_*
        probabilities in SimConfig.

        WHY INLINE RANDOM:
          Keeping this a two-line cumulative roll rather than reaching for
          random.choices lets us stay in lockstep with the unseeded Python
          random module the rest of the sim already uses. When we fix the
          spawner RNG (see PENDING_DISCUSSIONS.md #3) we'll route this
          through the same seeded Generator.
        """
        r = random.random()
        if r < SimConfig.TURN_PROB_STRAIGHT:
            return "straight"
        if r < SimConfig.TURN_PROB_STRAIGHT + SimConfig.TURN_PROB_LEFT:
            return "left"
        return "right"

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

    def _past_box_center(self, vehicle) -> bool:
        """
        Has the vehicle's center crossed the intersection's geometric center
        along its direction of travel?

        WHY THIS EXISTS:
          This is the trigger point for executing a queued turn. We wait
          until the car is deep enough into the box that a direction change
          looks roughly plausible — turning too early would skip the
          intersection entirely, turning too late would make the car jut
          out the exit edge before pivoting.
          The center is a simple, deterministic compromise.
        """
        if vehicle.direction == "N":
            return vehicle.y <= self.cy
        elif vehicle.direction == "S":
            return vehicle.y >= self.cy
        elif vehicle.direction == "E":
            return vehicle.x >= self.cx
        elif vehicle.direction == "W":
            return vehicle.x <= self.cx
        return False

    def _maybe_execute_turn(self, vehicle):
        """
        If this vehicle has a queued turn ("left" or "right") and is past the
        box center, flip its direction and re-position it into the outgoing
        lane. Clears intended_turn so the turn only fires once.

        PARAMETERS:
          vehicle — one of self.crossing_vehicles; must have intended_turn set
                    (possibly None or "straight", in which case we skip)

        POSITIONING — "snap + preserve overshoot":
          Two axes to think about after a turn:

          * Lane-axis (perpendicular to new direction):
              Hard-snap to the outgoing lane's exact coordinate. Lane-offset
              math mirrors vehicle_spawner._create_spawn_points —
                  N-bound lane i → x = cx - (i + 0.5) * lane_width
                  S-bound lane i → x = cx + (i + 0.5) * lane_width
                  E-bound lane i → y = cy + (i + 0.5) * lane_width
                  W-bound lane i → y = cy - (i + 0.5) * lane_width

          * Travel-axis (parallel to new direction):
              Preserve the OVERSHOOT — how far past the intersection center
              the vehicle went before the turn fired. Apply that overshoot
              as a head-start in the new direction.

          WHY OVERSHOOT MATTERS:
            Without preserving it, every turning car lands on the exact same
            pixel (cx, cy)-ish right after the turn. Under heavy spawning that
            pixel is occupied on almost every frame by a different car, which
            visually reads as "one car stuck forever at the intersection."
            Preserving overshoot means each car's post-turn position depends
            on how far past center it was when the turn fired — which varies
            step to step and car to car — so no two turning cars clump.

          DOWNSTREAM GUARANTEES (unchanged by this):
            - the rest of update()'s movement math treats the car as moving
              in the new direction, speed unchanged
            - has_vehicle_exited_box uses the NEW direction to check exit edge
            - handoff in Road uses .direction and .lane as-is

        WHY NOT "STRAIGHT":
          A straight "turn" means don't change anything. We explicitly skip
          those so we don't burn a branch or do needless work.
        """
        intent = vehicle.intended_turn
        if intent is None or intent == "straight":
            return

        if not self._past_box_center(vehicle):
            return

        # How far past center did the vehicle travel before the turn fired?
        # This is the "residual momentum" we want to carry into the new direction
        # so that cars don't all land on the exact same post-turn pixel.
        if vehicle.direction == "E":
            overshoot = vehicle.x - self.cx
        elif vehicle.direction == "W":
            overshoot = self.cx - vehicle.x
        elif vehicle.direction == "N":
            overshoot = self.cy - vehicle.y
        elif vehicle.direction == "S":
            overshoot = vehicle.y - self.cy
        else:
            overshoot = 0.0
        overshoot = max(0.0, overshoot)  # belt-and-braces — can't go backwards

        new_direction = TURN_MAP[vehicle.direction][intent]
        vehicle.direction = new_direction

        lw = SimConfig.LANE_WIDTH
        offset = (vehicle.lane + 0.5) * lw
        if new_direction == "N":
            vehicle.x = self.cx - offset
            vehicle.y = self.cy - overshoot   # already moving north
        elif new_direction == "S":
            vehicle.x = self.cx + offset
            vehicle.y = self.cy + overshoot   # already moving south
        elif new_direction == "E":
            vehicle.x = self.cx + overshoot   # already moving east
            vehicle.y = self.cy + offset
        elif new_direction == "W":
            vehicle.x = self.cx - overshoot   # already moving west
            vehicle.y = self.cy - offset

        # Turn consumed — prevents firing again if the vehicle lingers
        # past center for another step (e.g. while exiting the box).
        vehicle.intended_turn = None

    def has_vehicle_exited_box(self, vehicle) -> bool:
        """
        Has this vehicle cleared our intersection box on its OUTGOING side?

        WHY THIS EXISTS (PHASE 2):
          Used by Road._process_handoffs to know when a crossing vehicle
          should be handed off to the next intersection in its direction.
          Compare to _past_stop_line above, which asks the opposite question:
          "has the vehicle entered our box from its incoming side?"

        DIRECTION → EXIT-EDGE MAPPING:
          "N" (moving north, y decreasing)   → exits via north edge → y <= cy - half_box
          "S" (moving south, y increasing)   → exits via south edge → y >= cy + half_box
          "E" (moving east,  x increasing)   → exits via east edge  → x >= cx + half_box
          "W" (moving west,  x decreasing)   → exits via west edge  → x <= cx - half_box

        PARAMETERS:
          vehicle — any Vehicle with .x, .y, and .direction set

        RETURNS:
          True once the vehicle's center has moved past our box in the
          direction it's traveling — i.e. it's in the inter-intersection
          gap and ready to be handed off (or to drift off the grid edge
          if we're a boundary intersection).
        """
        half_box = self.box_size / 2
        if vehicle.direction == "N":
            return vehicle.y <= self.cy - half_box
        elif vehicle.direction == "S":
            return vehicle.y >= self.cy + half_box
        elif vehicle.direction == "E":
            return vehicle.x >= self.cx + half_box
        elif vehicle.direction == "W":
            return vehicle.x <= self.cx - half_box
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
