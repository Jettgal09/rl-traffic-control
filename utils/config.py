# utils/config.py
# Central config file — all simulation and training settings in one place
# One tweak here can change the whole behavior of the simulation or the learning process


class SimConfig:
    """
    Here all the settings for the traffic simulation itself go
    """

    # --- The City Grid ---
    GRID_SIZE = 1  # 1 means a 1x1 grid of 1 intersection, 3 means a 3x3 grid with 9 intersections
    CELL_SIZE = 250  # The difference in pixels between the center of one intersection and the next
    # NOTE: only affects grid_size > 1. For grid_size=1 the single intersection
    # always centers on the window, so Phase 1 results (phase1-paper-results tag)
    # are preserved regardless of this value.
    # Why 250? We want the outer APPROACH lanes (map edge → boundary box)
    # to be at least as long as the INTER-INTERSECTION lanes (box → box),
    # because spawning pressure lands on approach lanes while handoff only
    # trickles into inter-intersection lanes. For a 900x900 window with a
    # 3x3 grid and box_size=80, 250 gives ~160px approach vs ~170px between
    # neighbors — nicely balanced, both deep enough for ~4-5 queued cars.
    LANE_WIDTH = 20  # The width of each lane in pixels
    LANES_PER_DIRECTION = 2  # The number of lanes in each direction.

    # --- Vehicles ---
    VEHICLE_SPEED = 2.0  # How many pixels a car moves per simulation step
    VEHICLE_LENGTH = 15  # Length of the car rectangle
    VEHICLE_WIDTH = 10  # Width of the car rectangle
    MAX_VEHICLES = 100  # Maximum number of vehicles allowed in the simulation at once

    # --- Spawning ---
    # It controls how likely a new vehicle is to appear at each entry point on each step.
    SPAWN_RATE = 0.05

    # --- Turning Behavior (PHASE 2) ---
    # Each time a vehicle enters an intersection box, it rolls a random turn.
    # These probabilities must sum to 1.0.
    #
    # Why probabilistic per-intersection (not a pre-planned route)?
    #   Simpler sim logic, no pathfinding, and it still creates realistic
    #   inter-intersection coupling because downstream demand depends on
    #   what fraction of upstream cars turned toward each neighbor.
    #
    # Why this split (70/15/15)?
    #   Matches typical urban arterial distributions — most traffic goes
    #   straight, with left and right turns as minority behaviors. Easy to
    #   tweak later if we want to stress-test the agent under turn-heavy
    #   scenarios (e.g. 40/30/30 for a "grid with many destinations").
    TURN_PROB_STRAIGHT = 0.70
    TURN_PROB_LEFT = 0.15
    TURN_PROB_RIGHT = 0.15

    # --- Traffic Light Timing ---
    # Prevents the agent from learning to switch lights too rapidly.
    MIN_GREEN_DURATION = 15  # How many steps the green light must lasts before it can switch
    YELLOW_DURATION = 4  # How many steps the yellow light lasts before switching to red
    ALL_RED_DURATION = 2  # How many steps all directions are red between light changes

    # --- Episode Length ---
    # Episodes = one complete run of the simulation, from start to finish.
    MAX_STEPS_PER_EPISODE = 3000  # After this many steps the episode ends and resets

    # --- Reward V6: throughput bonus weight ---
    # Paid to the agent for each vehicle the spawner successfully fires per step.
    # Why this exists:
    #   V5 reward (queue + imbalance) was reward-hackable — Phase 2 iter2 A2C
    #   learned to BLOCK entry lanes, which prevents new spawns, which lowers
    #   observed queue, which raises reward. Operational metrics tanked while
    #   reward improved. Classic Goodhart. (See RESEARCH_NOTES iter2 post-mortem.)
    # Why throughput closes the loophole:
    #   The spawner only fires when the entry lane has room. If the agent
    #   throttles by congesting entry lanes, total_spawned stops growing and
    #   the agent stops collecting this bonus. So throttling stops paying.
    # Scale rationale:
    #   Δ(total_spawned) per step is typically 0–N_spawn_points * SPAWN_RATE
    #   (≈ 0.2 for 1×1, 0.4 for 2×2, 0.6 for 3×3). Aggregated V5 is in roughly
    #   [-1.5, -0.3]. Weight 1.0 makes the bonus a meaningful but not
    #   overwhelming counterweight. Tune up if exploit recurs, down if the
    #   agent ignores queue signal in favor of pure throughput chasing.
    # 2026-04-26: bumped from 1.0 to 2.0. RATIFIED — weight=2.0 result
    # nearly doubled the V6 win at 2×2 (waiting -7.0% at w=1.0 → -13.0% at
    # w=2.0; spawned +2.5% → +4.0%). Stronger throughput pull pays off
    # without re-introducing reward hacking. 2.0 is the keeper unless
    # exploring 3.0+ shows further gain.
    REWARD_THROUGHPUT_WEIGHT = 2.0


class RLConfig:
    """
    Settings for the RL system.
    All the hyperparameters for the learning algorithms go here.
    """

    # --- General RL settings ---
    TOTAL_TIMESTEPS = 500_000  # Total number of steps to train for
    ALGORITHM = "DQN"  # Which algorithm to use by default
    EVAL_FREQUENCY = 10_000  # How often to pause and test how well the agent is doing
    EVAL_EPISODES = 5  # When evaluating, how many full episodes to average across

    # --- Saving ---
    # Save a copy of the model every "xyz" steps
    SAVE_FREQUENCY = 50_000  # Number of steps between saves
    MODEL_SAVE_PATH = "experiments/"  # Path to save trained models and logs

    # --- DQN specific ---
    DQN_LEARNING_RATE = 1e-4  # Controls how drastically the model adjusts
    DQN_BUFFER_SIZE = 100_000  # How many past experiences to store for learning
    DQN_BATCH_SIZE = 64  # How many experiences to sample from the buffer for each step
    DQN_GAMMA = 0.99  # How much the agent values future rewards vs immediate ones
    DQN_EXPLORATION_FRACTION = 0.2  # How much time to spend on exploration (vs exploitation)
    DQN_TARGET_UPDATE_INTERVAL = 1000  # How many steps between updates to the target network

    # --- PPO specific ---
    PPO_LEARNING_RATE = 3e-4
    PPO_N_STEPS = 1024  # Steps to collect before each policy update
    PPO_BATCH_SIZE = 64
    PPO_N_EPOCHS = 10  # How many times to go through the collected data
    PPO_GAMMA = 0.99
    PPO_CLIP_RANGE = 0.2  # It limits how much the policy can change in a single update
    PPO_ENT_COEF = 0.05  # Higher entropy = more exploration

    # --- A2C specific ---
    A2C_LEARNING_RATE = 7e-4
    A2C_N_STEPS = 5  # A2C updates every 5 steps — much more frequent than PPO
    A2C_GAMMA = 0.99


class VisualizationConfig:
    """
    Settings for the Pygame window.
    It controls how the simulation will look when rendered.
    """

    # --- Window settings ---
    # The width and height of the Pygame window in pixels.
    WINDOW_WIDTH = 900
    WINDOW_HEIGHT = 900
    FPS = 30  # How many frames per second to render

    # --- Colors and styles ---
    # Defined as RGB tuples (Red, Green, Blue) with values from 0 to 255
    COLOR_BACKGROUND = (50, 50, 50)
    COLOR_ROAD = (80, 80, 80)
    COLOR_INTERSECTION = (100, 100, 100)
    COLOR_LANE_MARKING = (200, 200, 200)  # White dashed lines between lanes

    # --- Vehicle colors (PHASE 1 legacy, kept for reference) ---
    # Old scheme colored cars by current heading. Once turning landed in
    # Phase 2, a car's heading changes mid-trip, so these no longer uniquely
    # identify it. Superseded by COLOR_VEHICLE_FROM_* below; kept here so
    # older code that still references them (and the Phase 1 results commit)
    # does not break.
    COLOR_VEHICLE_NS = (100, 180, 255)  # Blue: cars going North or South
    COLOR_VEHICLE_EW = (255, 160, 60)   # Orange: cars going East or West

    # --- Vehicle colors by SPAWN ORIGIN (PHASE 2) ---
    # Keyed by the direction the car was first spawned heading — locked in
    # at spawn (vehicle.spawn_direction) and never flipped, even through
    # turns. So a car that entered at the west edge heading east stays
    # orange for its entire life, even after it turns north/south later.
    # Four visually distinct hues so you can eyeball flow across a 3x3 grid
    # and see "aha, most of the pink from the north edge made it through".
    COLOR_VEHICLE_FROM_S = (100, 200, 255)  # Cyan   — spawned heading N (entered from south edge)
    COLOR_VEHICLE_FROM_N = (255, 120, 200)  # Pink   — spawned heading S (entered from north edge)
    COLOR_VEHICLE_FROM_W = (255, 160, 60)   # Orange — spawned heading E (entered from west edge)
    COLOR_VEHICLE_FROM_E = (150, 230, 80)   # Lime   — spawned heading W (entered from east edge)

    COLOR_LIGHT_RED = (220, 50, 50)
    COLOR_LIGHT_YELLOW = (220, 200, 50)
    COLOR_LIGHT_GREEN = (50, 200, 80)
    COLOR_LIGHT_OFF = (60, 60, 60)  # Dark — light is inactive

    COLOR_TEXT = (240, 240, 240)
    FONT_SIZE = 16

    # --- Intersection-box alpha blend (PHASE 2 cosmetic fix, v2) ---
    # When two perpendicular cars both occupy the box — e.g. a northbound
    # car still flushing from GREEN_NS and a westbound car entering on the
    # new GREEN_EW — their rects sweep through the same pixels at some
    # point in the ~40-step crossing. In a 2D top-down render there's no
    # z-axis, so two bodies on crossing paths WILL overlap at some frame.
    #
    # We tried a 4-pixel positional nudge first — it separated the static
    # inner-lane crossing pixel but not the 39 other frames of sweep, so
    # the sim still looked like a pileup (v1 — superseded).
    #
    # This is the actual fix: cars inside the box get drawn at reduced
    # opacity. Two overlapping semi-transparent rectangles blend into a
    # brighter/mixed color rather than stacking as an opaque "crash" —
    # reads as motion-blur-y pass-through instead of accident. Lane cars
    # are untouched, so queues still look solid and legible.
    #
    # We don't fix this in the sim (that would require real gap-acceptance,
    # option C in PENDING_DISCUSSIONS.md #4 — locked in a cage for later).
    #
    # 150 / 255 ≈ 59% opacity. Picked by eye: low enough that overlaps
    # blend visibly, high enough that single cars still track cleanly
    # against the grey road background.
    BOX_RENDER_ALPHA = 150
