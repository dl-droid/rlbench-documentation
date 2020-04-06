# import deep_learning_rl as Simulator  
from SimulationEnvironment.Environment import ReachTargetSimulationEnv
import models.SmartImmitationAgent as IL   
import time
# Set dataset_root to load from a folder and datasst will load from there. 
curr_env = ReachTargetSimulationEnv(dataset_root='/tmp/rlbench_data',headless=True)
# Set image_paths_output=False when loading dataset from file. 
demos = curr_env.get_demos(200,live_demos=False,image_paths_output=True) 
agent = IL.SimpleImmitationLearningAgent() 
# agent.load_model('SavedModels/ImmitationLearningConvolvingMutantAgent-2020-04-05-04-18.pt')
agent.injest_demonstrations(demos)   
agent.train_agent(100)
# del curr_env
curr_env = ReachTargetSimulationEnv(headless=False,episode_length=50,num_episodes=500)
simulation_stats = curr_env.run_trained_agent(agent)

current_time = time.localtime()
model_name = agent.__class__.__name__+'-'+time.strftime('%Y-%m-%d-%H-%M', current_time)
# agent.save_model('SavedModels/'+model_name+'.pt')