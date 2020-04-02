import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from rlbench.backend.observation import Observation
from typing import List
import numpy as np
from . import logger
 


class FullyConnectedPolicyEstimator(nn.Module):
    
    def __init__(self,num_actions,num_states):
        super(FullyConnectedPolicyEstimator, self).__init__()
        self.fc1 = nn.Linear(num_states, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, num_actions)

    # x is input to the network.
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



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
        raise NotImplementedError()

    def apply_gradient(self, ground_truth_actions, predicted_actions):
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
    

class ImmitationLearningAgent(LearningAgent):
    """
    ImmitationLearningAgent
    -----------------------

    Dumb Learning Agent that will try to eastimate an action given a state. 
    This will Not be considering Past actions/states while making these predictions. 
    
    todo : Make LSTM Based Networks that can remember over a batch of given observations. 
    https://stackoverflow.com/a/27516930 : For LSTM Array Stacking
    """
    def __init__(self,learning_rate = 0.01):
        super(LearningAgent,self).__init__()
        self.learning_rate = learning_rate
        self.neural_network = FullyConnectedPolicyEstimator(7,7)
        self.optimizer = optim.SGD(self.neural_network.parameters(), lr=learning_rate, momentum=0.9)
        self.loss_function = nn.SmoothL1Loss()
        self.training_data = None
        self.logger = logger.create_logger(__name__)
        self.input_state = 'joint_positions'
        self.output_action = 'joint_velocities'
        self.data_loader = None
        self.dataset = None

    def injest_demonstrations(self,demos:List[List[Observation]]):
        self.training_data = demos # These hold Individual Demonstrations As each Batch.
        # todo : Figure Batch Size Problem 
        # https://stats.stackexchange.com/questions/187591/when-the-data-set-size-is-not-a-multiple-of-the-mini-batch-size-should-the-last
        
        # $ CREATE Matrix of shape (total_step_from_all_demos,shape_of_observation)
        # $ This is done because we are training a dumb agent to estimate a policy based on just dumb current state
        # $ So for training we will use a 2D Matrix. If we were doing LSTM based training then the data modeling will change. 
        train_vectors = torch.from_numpy(np.array([getattr(observation,self.input_state) for episode in demos for observation in episode]))
        ground_truth = torch.from_numpy(np.array([getattr(observation,self.output_action) for episode in demos for observation in episode]))
        
        self.dataset = torch.utils.data.TensorDataset(train_vectors, ground_truth)
        self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=64, shuffle=True)


    def train_agent(self,epochs:int):
        if not self.dataset:
            raise Exception("No Training Data Set to Train Agent. Please set Training Data using ImmitationLearningAgent.injest_demonstrations")
        
        self.logger.info("Starting Training of Agent ")
        self.neural_network.train()
        for epoch in range(epochs):
            running_loss = 0.0
            steps = 0
            for batch_idx, (data, target) in enumerate(self.data_loader):
                data, target = Variable(data), Variable(target)
                self.optimizer.zero_grad()
                network_pred = self.neural_network(data.float()) 
                loss = self.loss_function(network_pred,target.float())
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                steps+=1

            # self.logger.info('[%d] loss: %.6f' % (epoch + 1, running_loss / (steps+1)))

    def predict_action(self, demonstration_episode:List[Observation]):
        self.neural_network.eval()
        train_vectors = torch.from_numpy(np.array([getattr(observation,self.input_state) for observation in demonstration_episode]))
        input_val = Variable(train_vectors)
        output = self.neural_network(input_val.float())
        return output.data.cpu().numpy()
        # return np.random.uniform(size=(len(batch), 7))
    
# class RoboticsObservationDataset(data.Dataset):

#     def __init__(self, robot_states, labels):
#         'Initialization'
#         self.labels = labels
#         self.list_IDs = list_IDs

# from deep_learning_rl import get_demos 
# demos = get_demos(2)
# import models.ImmitationLearning as IL  
# agent = IL.ImmitationLearningAgent()
# agent.injest_demonstrations(demos)   