# utils/config.py
# Central config file — all simulation and training settings in one place
# One tweak here can change the whole behavior of the simulation or the learning process


class SimConfig:
    """
    Here all the settings for the traffic simulation itself go
    """

    # --- The City Grid ---
    GRID_SIZE = 1  # 1 means a 1x1 grid of 1 intersection, 3 means a 3x3 grid with 9 intersections
    CELL_SIZE = 200  # The difference in pixels between the center of one intersection and the next
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

    # --- Traffic Light Timing ---
    # Prevents the agent from learning to switch lights too rapidly.
    MIN_GREEN_DURATION = 15  # How many steps the green light must lasts before it can switch
    YELLOW_DURATION = 4  # How many steps the yellow light lasts before switching to red
    ALL_RED_DURATION = 2  # How many steps all directions are red between light changes

    # --- Episode Length ---
    # Episodes = one complete run of the simulation, from start to finish.
    MAX_STEPS_PER_EPISODE = 3000  # After this many steps the episode ends and resets


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

    COLOR_VEHICLE_NS = (100, 180, 255)  # Blue: cars going North or South
    COLOR_VEHICLE_EW = (255, 160, 60)  # Orange: cars going East or West

    COLOR_LIGHT_RED = (220, 50, 50)
    COLOR_LIGHT_YELLOW = (220, 200, 50)
    COLOR_LIGHT_GREEN = (50, 200, 80)
    COLOR_LIGHT_OFF = (60, 60, 60)  # Dark — light is inactive

    COLOR_TEXT = (240, 240, 240)
    FONT_SIZE = 16
