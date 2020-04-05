# import deep_learning_rl as Simulator  
from SimulationEnvironment.Environment import ReachTargetSimulationEnv
import models.ImmitationMutant as IL   
# Set dataset_root to load from a folder and datasst will load from there. 
curr_env = ReachTargetSimulationEnv(dataset_root='/tmp/rlbench_data')
# Set image_paths=False when loading dataset from file. 
demos = curr_env.get_demos(100,live_demos=False,image_paths=False) 
agent = IL.ImmitationLearningMutantAgent() 
agent.injest_demonstrations(demos)   
agent.train_agent(100)
# del curr_env
curr_env = ReachTargetSimulationEnv(headless=False)
curr_env.run_trained_agent(agent)
