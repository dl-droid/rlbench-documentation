# This file will contain functions that will act as utility functions working with environments and simulations. 
from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget
from models.Agent import LearningAgent
import numpy as np
from model.logger import create_logger
module_logger = create_logger('RLBench_Simulation_Wrapper')
        
def get_demos(num_demos):
    # To use 'saved' demos, set the path below, and set live_demos=False
    live_demos = True
    DATASET = '' if live_demos else 'PATH/TO/YOUR/DATASET'
    obs_config = ObservationConfig()
    obs_config.set_all(False)
    obs_config.joint_positions = True
    obs_config.joint_velocities = True
    action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
    env = Environment(
        action_mode, DATASET, obs_config, True)
    env.launch()

    task = env.get_task(ReachTarget)
    demos = task.get_demos(num_demos, live_demos=True)  # -> List[List[Observation]]
    demos = np.array(demos).flatten()
    env.shutdown()
    return demos


def simulate_trained_agent(agent:LearningAgent,training_steps = 120,episode_length = 40):
    live_demos = True
    DATASET = '' if live_demos else 'PATH/TO/YOUR/DATASET'
    obs_config = ObservationConfig()
    obs_config.set_all(False)
    obs_config.joint_positions = True
    obs_config.joint_velocities = True
    action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
    env = Environment(
        action_mode, '', obs_config, True)
    env.launch()
    task = env.get_task(ReachTarget)
    obs = None
    for i in range(training_steps):
        if i % episode_length == 0:
            module_logger.info('Reset Episode')
            descriptions, obs = task.reset()
            module_logger.info(descriptions)
        action = agent.predict_action(obs)
        obs, reward, terminate = task.step(action)
        if reward == 1:
            module_logger.info("Reward Of 1 Achieved. Task Completed By Agent :) ")
            return

# print(demos)
# # An example of using the demos to 'train' using behaviour cloning loss.
# for i in range(100): # $EPOCHS
#     print("'training' iteration %d" % i)
#     # Get a batch/episode/demo from full dataset. 
#     batch = np.random.choice(demos, replace=False)
#     print(batch,len(batch))
#     joint_positions = [obs.joint_positions for obs in batch]
#     # print(joint_positions)
#     # Predict the action on basis of the input provided to the model.
#     # Here the prediction is join velocities
#     predicted_actions = imitation_learning_model.predict_action(joint_positions)
#     # Get Ground truth values of actions of the chosen batch.  
#     ground_truth_actions = [obs.joint_velocities for obs in batch]
#     print(ground_truth_actions[0])
#     # Apply the Gradients to model for next prediction in the training cycles. 
#     imitation_learning_model.apply_gradient(ground_truth_actions, predicted_actions)


# print('Done')