# Import Absolutes deps
from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.backend.observation import Observation
from rlbench.tasks import *
from typing import List
import numpy as np
from gym import spaces
# Import Relative deps
import sys
sys.path.append('..')
from models.Agent import LearningAgent
import logger
# list of state types
state_types = [ 'left_shoulder_rgb',
                'left_shoulder_depth',
                'left_shoulder_mask',
                'right_shoulder_rgb',
                'right_shoulder_depth',
                'right_shoulder_mask',
                'wrist_rgb',
                'wrist_depth',
                'wrist_mask',
                'joint_velocities',
                'joint_velocities_noise',
                'joint_positions',
                'joint_positions_noise',
                'joint_forces',
                'joint_forces_noise',
                'gripper_pose',
                'gripper_touch_forces',
                'task_low_dim_state']

# The CameraConfig controls if mask values will be RGB or 1 channel 
# https://github.com/stepjam/RLBench/blob/master/rlbench/observation_config.py#L5
# The depth Values are also single channel. Where depth will be of dims (width,height)
# https://github.com/stepjam/PyRep/blob/4a61f6756c3827db66423409632358de312b97e4/pyrep/objects/vision_sensor.py#L128
image_types=[ 
    'left_shoulder_rgb',
    'left_shoulder_depth', # Depth is in Black and White
    'left_shoulder_mask', # Mask can be single channel
    'right_shoulder_rgb',
    'right_shoulder_depth', # Depth is in Black and White
    'right_shoulder_mask', # Mask can be single channel
    'wrist_rgb',
    'wrist_depth', # Depth is in Black and White
    'wrist_mask' # Mask can be single channel
]



DEFAULT_ACTION_MODE = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
DEFAULT_TASK = ReachTarget
class SimulationEnvionment():
    """
    This can be a parent class from which we can have multiple child classes that 
    can diversify for different tasks and deeper functions within the tasks.
    """
    def __init__(self,\
                action_mode=DEFAULT_ACTION_MODE,\
                task=DEFAULT_TASK,\
                headless=True):
        obs_config = ObservationConfig()
        obs_config.set_all(True)
        action_mode = action_mode
        self.env = Environment(
            action_mode, obs_config=obs_config, headless=headless)
        # self.env.launch()
        self.env.launch()
        self.task = self.env.get_task(task)
        _, obs = self.task.reset()
        self.action_space =  spaces.Box(low=-1.0, high=1.0, shape=(action_mode.action_size,), dtype=np.float32)
        self.logger = logger.create_logger(__class__.__name__)
        self.logger.propagate = 0


    def _get_state(self, obs:Observation):
        # _get_state function is present so that some alterations can be made to observations so that
        # dimensionality management is handled from lower level. 
        for state_type in image_types:    # changing axis of images in `Observation`
            image = getattr(obs, state_type)
            if image is None:
                continue
            if len(image.shape) == 2:
                # Depth and Mask can be single channel.Hence we Reshape the image from (width,height) -> (width,height,1)
                image = image.reshape(*image.shape,1)
            # self.logger.info("Shape of : %s Before Move Axis %s" % (state_type,str(image.shape)))
            image=np.moveaxis(image, 2, 0)  # change (H, W, C) to (C, H, W) for torch
            # self.logger.info("After Moveaxis :: %s" % str(image.shape))
            setattr(obs,state_type,image)
        return obs

    
    def reset(self):
        descriptions, obs = self.task.reset()
        return self._get_state(obs)

    def step(self, action):
        obs_, reward, terminate = self.task.step(action)  # reward in original rlbench is binary for success or not
        return self._get_state(obs_), reward, terminate, None

    def shutdown(self):
        self.logger.info("Environment Shutdown! Create New Instance If u want to start again")
        self.logger.handlers.pop()
        self.env.shutdown()
    
    def get_demos(self,num_demos):
        demos = self.task.get_demos(num_demos, live_demos=True)  # -> List[List[Observation]]
        demos = np.array(demos).flatten()
        self.shutdown()
        new_demos = []
        for episode in demos:
            new_episode = []
            for i in range(len(episode)):
                new_episode.append(self._get_state(episode[i]))
            new_demos.append(new_episode)
        return new_demos


class ReachTargetSimulationEnv(SimulationEnvionment):
    """
    Inherits the `SimulationEnvironment` class. 
    This environment is specially ment for running traing agent for ReachTarget Task. 
    This can be inherited for different ways of doing learning. 
    """
    def __init__(self, action_mode=DEFAULT_ACTION_MODE, headless=True,training_steps = 120,episode_length = 40):
        super(ReachTargetSimulationEnv,self).__init__(action_mode=action_mode, task=ReachTarget, headless=headless)
        self.training_steps = training_steps
        self.episode_length = episode_length
        self.logger = logger.create_logger(__class__.__name__)
        self.logger.propagate = 0

    def get_goal_poistion(self):
        """
        This will return the postion of the target for the ReachTarget Task. 
        """
        return np.array(self.task.target.get_position())

    def run_trained_agent(self,agent:LearningAgent):
        for i in range(self.training_steps):
            if i % self.episode_length == 0:
                self.logger.info('Reset Episode %d'% i)
                descriptions, obs = self.task.reset()
                self.logger.info(descriptions)
            action = agent.predict_action([obs])
            selected_action = action[0]
            obs, reward, terminate = self.task.step(selected_action)
            if reward == 1:
                self.logger.info("Reward Of 1 Achieved. Task Completed By Agent In steps : %d"%i)
                return
            if terminate:
                self.logger.info("Recieved Terminate")

        self.shutdown()