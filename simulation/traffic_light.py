# simulation/traffic_light.py
#
# Models a single traffic light at one intersection.
# This is a STATE MACHINE — it is always in exactly one phase,
# and moves through phases in a fixed cycle.

from enum import IntEnum
from utils.config import SimConfig

# Phase duration options the RL agent can choose from (in steps)
DURATION_OPTIONS = [20, 40, 60, 80]

# Default green duration if agent hasn't set one yet
DEFAULT_GREEN_DURATION = 40

MAX_RED_DURATION = 100


class Phase(IntEnum):
    GREEN_NS = 0
    YELLOW_NS = 1
    ALL_RED_1 = 2
    GREEN_EW = 3
    YELLOW_EW = 4
    ALL_RED_2 = 5


PHASE_CYCLE = {
    Phase.GREEN_NS: Phase.YELLOW_NS,
    Phase.YELLOW_NS: Phase.ALL_RED_1,
    Phase.ALL_RED_1: Phase.GREEN_EW,
    Phase.GREEN_EW: Phase.YELLOW_EW,
    Phase.YELLOW_EW: Phase.ALL_RED_2,
    Phase.ALL_RED_2: Phase.GREEN_NS,
}

PHASE_DURATION = {
    Phase.YELLOW_NS: SimConfig.YELLOW_DURATION,
    Phase.ALL_RED_1: SimConfig.ALL_RED_DURATION,
    Phase.YELLOW_EW: SimConfig.YELLOW_DURATION,
    Phase.ALL_RED_2: SimConfig.ALL_RED_DURATION,
}


class TrafficLight:
    """
    Traffic light with duration-based phase control.

    Instead of the agent saying "switch now", it says
    "keep this phase green for X steps" at the START
    of each green phase.

    This eliminates the yellow/all-red penalty problem
    because the agent sets duration once and waits.
    """

    def __init__(self):
        self.phase = Phase.GREEN_NS
        self.phase_timer = 0
        self.current_green_duration = DEFAULT_GREEN_DURATION

    def set_green_duration(self, duration: int):
        """
        Agent calls this at the start of a green phase
        to set how long it should last.

        Only has effect during green phases.
        """
        if self._is_green_phase():
            self.current_green_duration = max(SimConfig.MIN_GREEN_DURATION, duration)

    def update(self):
        """
        Advance light by one step.
        Green phases last exactly current_green_duration steps.
        Yellow and all-red advance automatically.
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
        # Reset duration for next green phase
        if self._is_green_phase():
            self.current_green_duration = DEFAULT_GREEN_DURATION

    def is_green_for(self, direction: str) -> bool:
        if direction in ("N", "S"):
            return self.phase == Phase.GREEN_NS
        elif direction in ("E", "W"):
            return self.phase == Phase.GREEN_EW
        return False

    def is_start_of_green_phase(self) -> bool:
        """
        Returns True on the very first step of a new green phase.
        This is when the agent should set the duration.
        """
        return self._is_green_phase() and self.phase_timer == 1

    def reset(self):
        self.phase = Phase.GREEN_NS
        self.phase_timer = 0
        self.current_green_duration = DEFAULT_GREEN_DURATION
