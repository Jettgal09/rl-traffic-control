# simulation/traffic_light.py
# Models a traffic light with duration-based phase control.

from enum import IntEnum
from utils.config import SimConfig

# Allowed green durations in steps. Agent can choose any of these at the start of a green phase.
DURATION_OPTIONS = [20, 40, 60, 80]
DEFAULT_GREEN_DURATION = 40
MAX_RED_DURATION = 100  # Forces a switch if one direction stays red too long


# each phase is mapped to an integer for easy comparison and observation encoding
class Phase(IntEnum):
    GREEN_NS = 0
    YELLOW_NS = 1
    ALL_RED_1 = 2
    GREEN_EW = 3
    YELLOW_EW = 4
    ALL_RED_2 = 5


# Maps each phase to the next one in the cycle
PHASE_CYCLE = {
    Phase.GREEN_NS: Phase.YELLOW_NS,
    Phase.YELLOW_NS: Phase.ALL_RED_1,
    Phase.ALL_RED_1: Phase.GREEN_EW,
    Phase.GREEN_EW: Phase.YELLOW_EW,
    Phase.YELLOW_EW: Phase.ALL_RED_2,
    Phase.ALL_RED_2: Phase.GREEN_NS,
}

# Fixed durations for yellow and all-red phases — green duration is set by the agent
PHASE_DURATION = {
    Phase.YELLOW_NS: SimConfig.YELLOW_DURATION,
    Phase.ALL_RED_1: SimConfig.ALL_RED_DURATION,
    Phase.YELLOW_EW: SimConfig.YELLOW_DURATION,
    Phase.ALL_RED_2: SimConfig.ALL_RED_DURATION,
}


class TrafficLight:
    """
    Represents a traffic light at the intersection.
    """

    def __init__(self):
        self.phase = Phase.GREEN_NS
        self.phase_timer = 0
        self.current_green_duration = DEFAULT_GREEN_DURATION

    def set_green_duration(self, duration: int):
        if self._is_green_phase():
            self.current_green_duration = max(SimConfig.MIN_GREEN_DURATION, duration)

    def update(self):
        """
        Advances the light by one step — handles green, yellow and all-red phases.
        """
        self.phase_timer += 1

        if self._is_green_phase():
            if self.phase_timer >= self.current_green_duration:
                self._advance_phase()
        else:
            required = PHASE_DURATION[self.phase]
            if self.phase_timer >= required:
                self._advance_phase()

    def _is_green_phase(self):
        return self.phase in (Phase.GREEN_NS, Phase.GREEN_EW)

    def _advance_phase(self):
        self.phase = PHASE_CYCLE[self.phase]
        self.phase_timer = 0
        if self._is_green_phase():
            self.current_green_duration = DEFAULT_GREEN_DURATION

    def is_green_for(self, direction: str) -> bool:
        if direction in ("N", "S"):
            return self.phase == Phase.GREEN_NS
        elif direction in ("E", "W"):
            return self.phase == Phase.GREEN_EW
        return False

    def is_start_of_green_phase(self) -> bool:
        return self._is_green_phase() and self.phase_timer == 1

    def reset(self):
        self.phase = Phase.GREEN_NS
        self.phase_timer = 0
        self.current_green_duration = DEFAULT_GREEN_DURATION
