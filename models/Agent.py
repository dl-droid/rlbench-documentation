import torch
from typing import List
from rlbench.backend.observation import Observation
class LearningAgent():
    """
    General Purpose class to abstract the functionality of the network from the agent.

    Use this as a base class to create differnt Learning Based agenets that can work and be trained on 
    
    different Deep Learning Algorithms. 
    """

    def __init__(self):
        self.learning_rate = None
        self.neural_network = None
        self.optimizer = None
        self.loss_function = None
        self.training_data = None
        self.logger = None
        self.input_state = None
        self.output_action = None

    def injest_demonstrations(self,demos:List[List[Observation]]):
        raise NotImplementedError()

    
    def train_agent(self,epochs:int):
        raise NotImplementedError()
    
    # Keeping it a list of Observation to keep flexibility for LSTM type networks.
    def predict_action(self, demonstration_episode:List[Observation]):
        """
        This should Use model.eval() in Pytoch to do prediction for an action
        This is ment for using saved model
        """
        raise NotImplementedError()

    def act(self,state:List[Observation]):
        """
        This will be used by the RL agents and Learn from feadback from the environment. 
        This will let pytorch hold gradients when running the network. 
        """
        raise NotImplementedError()

    def save_model(self,file_path):
        if not self.neural_network:
            return
        self.neural_network.to('cpu')
        torch.save(self.neural_network.state_dict(), file_path)

    def load_model(self,file_path):
        if not self.neural_network:
            return
        # $ this will load a model from file path.
        self.neural_network.load_state_dict(torch.load(file_path))
    
