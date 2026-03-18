# utils/config.py
#
# This is the SINGLE SOURCE OF TRUTH for every number in this project.
# Instead of writing "speed = 2" in 10 different files, we define it
# once here. Every other file imports from here.
#
# If you want to experiment (bigger grid, faster cars, more traffic),
# you change ONE number here and the whole system adapts.


class SimConfig:
    """
    Settings for the physical simulation world.
    Cars, roads, intersections, traffic light timing — all defined here.
    """

    # --- The City Grid ---
    # GRID_SIZE = 1 means ONE intersection (Phase 1, what we build first)
    # GRID_SIZE = 3 means a 3x3 grid of 9 intersections (Phase 2 later)
    GRID_SIZE = 1

    # How many pixels apart each intersection is on screen
    CELL_SIZE = 200

    # How wide is one lane in pixels
    LANE_WIDTH = 20

    # How many lanes go in each direction at an intersection
    # 2 means: 2 lanes going north, 2 going south, 2 east, 2 west
    LANES_PER_DIRECTION = 2

    # --- Vehicles ---
    VEHICLE_SPEED = 2.0  # How many pixels a car moves per simulation step
    VEHICLE_LENGTH = 15  # Size of the car rectangle (pixels)
    VEHICLE_WIDTH = 10
    MAX_VEHICLES = 100  # Cap on total cars in the simulation at once

    # --- Spawning ---
    # Every simulation step, for each spawn point, we roll a random number.
    # If that number is below SPAWN_RATE, a new car appears.
    # 0.05 = 5% chance per step = moderate traffic
    SPAWN_RATE = 0.05

    # --- Traffic Light Timing ---
    # Once a light goes green, the RL agent cannot switch it for at least
    # this many steps. Prevents the agent from flickering lights every step.
    MIN_GREEN_DURATION = 15

    YELLOW_DURATION = 4  # How many steps the yellow light lasts
    ALL_RED_DURATION = 2  # Brief all-red pause between phases (safety)

    # --- Episode Length ---
    # One "episode" = one full run of the simulation from start to finish.
    # After this many steps the episode ends and resets.
    MAX_STEPS_PER_EPISODE = 3000


class RLConfig:
    """
    Settings for the Reinforcement Learning system.

    These are called HYPERPARAMETERS — settings YOU choose before training
    starts. The agent doesn't learn these, you set them manually.
    Think of them like the rules of how the agent is allowed to learn.
    """

    # Total number of steps to train for
    TOTAL_TIMESTEPS = 500_000

    # Which algorithm to use by default
    ALGORITHM = "DQN"

    # How often (in steps) to pause and test how well the agent is doing
    EVAL_FREQUENCY = 10_000

    # When evaluating, how many full episodes to average across
    EVAL_EPISODES = 5

    # Save a copy of the model every this many steps
    SAVE_FREQUENCY = 50_000

    MODEL_SAVE_PATH = "experiments/"

    # --- DQN specific settings ---
    # (we will explain every one of these when we get to the DQN lesson)
    DQN_LEARNING_RATE = 1e-4
    DQN_BUFFER_SIZE = 100_000
    DQN_BATCH_SIZE = 64
    DQN_GAMMA = 0.99
    DQN_EXPLORATION_FRACTION = 0.2
    DQN_TARGET_UPDATE_INTERVAL = 1000

    # --- PPO specific settings ---
    PPO_LEARNING_RATE = 3e-4
    PPO_N_STEPS = 1024        # was 2048, more frequent updates
    PPO_BATCH_SIZE = 64
    PPO_N_EPOCHS = 10
    PPO_GAMMA = 0.99
    PPO_CLIP_RANGE = 0.2
    PPO_ENT_COEF = 0.05       # ADD THIS — higher entropy = more exploration

    # --- A2C specific settings ---
    A2C_LEARNING_RATE = 7e-4
    A2C_N_STEPS = 5
    A2C_GAMMA = 0.99


class VisualizationConfig:
    """
    Settings for the Pygame window — colors, size, speed.
    The RL agent never sees any of this.
    This is purely for US to watch what is happening.
    """

    WINDOW_WIDTH = 900
    WINDOW_HEIGHT = 900
    FPS = 30

    # Colors are (Red, Green, Blue) tuples, values 0-255
    COLOR_BACKGROUND = (50, 50, 50)  # Dark gray — the ground
    COLOR_ROAD = (80, 80, 80)  # Slightly lighter — road surface
    COLOR_INTERSECTION = (100, 100, 100)  # The box where roads cross
    COLOR_LANE_MARKING = (200, 200, 200)  # White dashed lines between lanes

    COLOR_VEHICLE_NS = (100, 180, 255)  # Blue  — cars going North or South
    COLOR_VEHICLE_EW = (255, 160, 60)  # Orange — cars going East or West

    COLOR_LIGHT_RED = (220, 50, 50)
    COLOR_LIGHT_YELLOW = (220, 200, 50)
    COLOR_LIGHT_GREEN = (50, 200, 80)
    COLOR_LIGHT_OFF = (60, 60, 60)  # Dark — light is inactive

    COLOR_TEXT = (240, 240, 240)
    FONT_SIZE = 16
