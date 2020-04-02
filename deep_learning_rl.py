# This file will contain functions that will act as utility functions working with environments and simulations. 
from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget
from models.Agent import LearningAgent
import numpy as np
from models.logger import create_logger
module_logger = create_logger('RLBench_Simulation_Wrapper')
        
def get_demos(num_demos):
    # To use 'saved' demos, set the path below, and set live_demos=False
    DATASET = '' 
    obs_config = ObservationConfig()
    obs_config.set_all(False)
    obs_config.joint_positions = True
    obs_config.joint_velocities = True
    obs_config.gripper_open_amount = True
    action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
    env = Environment(
        action_mode, DATASET, obs_config, True)
    env.launch()

    task = env.get_task(ReachTarget)
    demos = task.get_demos(num_demos, live_demos=True)  # -> List[List[Observation]]
    demos = np.array(demos).flatten()
    env.shutdown()
    return demos


def simulate_trained_agent(agent:LearningAgent,training_steps = 120,episode_length = 40,headless=False):
    obs_config = ObservationConfig()
    obs_config.set_all(False)
    obs_config.joint_positions = True
    obs_config.joint_velocities = True
    obs_config.gripper_open_amount = True
    action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
    env = Environment(
        action_mode, '', obs_config, headless=headless)
    env.launch()
    task = env.get_task(ReachTarget)
    obs = None
    # $ action should contain 1 extra value for gripper open close state
    for i in range(training_steps):
        if i % episode_length == 0:
            module_logger.info('Reset Episode')
            descriptions, obs = task.reset()
            module_logger.info(descriptions)
        action = agent.predict_action([obs])
        selected_action = action[0]
        obs, reward, terminate = task.step(selected_action)
        if reward == 1:
            module_logger.info("Reward Of 1 Achieved. Task Completed By Agent :) ")
            return
        if terminate:
            module_logger.info("Recieved Terminate")