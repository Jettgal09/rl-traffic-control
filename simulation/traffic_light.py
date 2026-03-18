# simulation/traffic_light.py
#
# Models a single traffic light at one intersection.
# This is a STATE MACHINE — it is always in exactly one phase,
# and moves through phases in a fixed cycle.

from enum import IntEnum
from utils.config import SimConfig

MAX_RED_DURATION = 100  # Maximum steps any direction can stay red to prevent starvation


class Phase(IntEnum):
    """
    All possible states a traffic light can be in.

    The cycle always goes in this exact order:
    GREEN_NS → YELLOW_NS → ALL_RED_1 → GREEN_EW → YELLOW_EW → ALL_RED_2 → (repeat)
    """

    GREEN_NS = 0  # North-South traffic can go. East-West is stopped.
    YELLOW_NS = 1  # North-South about to turn red. Warning phase.
    ALL_RED_1 = 2  # Everyone stopped. Safety gap before EW gets green.
    GREEN_EW = 3  # East-West traffic can go. North-South is stopped.
    YELLOW_EW = 4  # East-West about to turn red. Warning phase.
    ALL_RED_2 = 5  # Everyone stopped. Safety gap before NS gets green.


# When in phase X, the next phase is always Y.
# This replaces a long if/elif chain with a simple dictionary lookup.
PHASE_CYCLE = {
    Phase.GREEN_NS: Phase.YELLOW_NS,
    Phase.YELLOW_NS: Phase.ALL_RED_1,
    Phase.ALL_RED_1: Phase.GREEN_EW,
    Phase.GREEN_EW: Phase.YELLOW_EW,
    Phase.YELLOW_EW: Phase.ALL_RED_2,
    Phase.ALL_RED_2: Phase.GREEN_NS,
}

# How many steps each non-green phase lasts.
# Green phases are NOT here — their duration is controlled by the RL agent.
PHASE_DURATION = {
    Phase.YELLOW_NS: SimConfig.YELLOW_DURATION,
    Phase.ALL_RED_1: SimConfig.ALL_RED_DURATION,
    Phase.YELLOW_EW: SimConfig.YELLOW_DURATION,
    Phase.ALL_RED_2: SimConfig.ALL_RED_DURATION,
}


class TrafficLight:
    """
    One traffic light controller at one intersection.

    The RL agent interacts with this class in one way only:
    it calls request_switch() when it wants to change the phase.
    Everything else (yellow transitions, timing) is handled automatically.
    """

    def __init__(self):
        # What phase are we currently in?
        # We always start with North-South green.
        self.phase = Phase.GREEN_NS

        # How many simulation steps have we been in the current phase?
        # This resets to 0 every time the phase changes.
        self.phase_timer = 0

        # Has the RL agent asked us to switch phases?
        # This gets set to True by request_switch(), and back to False
        # once the switch actually happens.
        self.switch_requested = False

    def request_switch(self):
        """
        The RL agent calls this to request a phase change.
        The switch won't happen instantly — update() checks it each step.
        """
        self.switch_requested = True

    def update(self):
        """
        Called every simulation step. Advances the light through its cycle.

        Two behaviors:
        - Green phase: switch if agent requested AND min time passed
                       OR force switch if max red duration exceeded
        - Yellow/All-red: switch automatically when duration expires
        """
        self.phase_timer += 1

        if self._is_green_phase():
            agent_wants_switch = self.switch_requested
            been_green_long_enough = self.phase_timer >= SimConfig.MIN_GREEN_DURATION
            been_green_too_long = self.phase_timer >= MAX_RED_DURATION

            if been_green_too_long:
                # Force switch — opposing direction waited too long
                self._advance_phase()
                self.switch_requested = False
            elif agent_wants_switch and been_green_long_enough:
                # Agent requested switch and minimum time met
                self._advance_phase()
                self.switch_requested = False
        else:
            # Yellow and all-red switch automatically by time
            required_duration = PHASE_DURATION[self.phase]
            if self.phase_timer >= required_duration:
                self._advance_phase()

    def _is_green_phase(self):
        """Returns True if we are currently in a green phase."""
        return self.phase in (Phase.GREEN_NS, Phase.GREEN_EW)

    def _advance_phase(self):
        """Move to the next phase in the cycle and reset the timer."""
        self.phase = PHASE_CYCLE[self.phase]
        self.phase_timer = 0

    def is_green_for(self, direction: str) -> bool:
        """
        Vehicles call this to know if they are allowed to move.

        A vehicle traveling North or South checks if GREEN_NS is active.
        A vehicle traveling East or West checks if GREEN_EW is active.

        direction is "N", "S", "E", or "W"
        """
        if direction in ("N", "S"):
            return self.phase == Phase.GREEN_NS
        elif direction in ("E", "W"):
            return self.phase == Phase.GREEN_EW
        return False

    def reset(self):
        """
        Reset to initial state.
        Called at the start of each new RL episode so every episode
        starts from the same clean state.
        """
        self.phase = Phase.GREEN_NS
        self.phase_timer = 0
        self.switch_requested = False
