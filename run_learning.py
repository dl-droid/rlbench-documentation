# import deep_learning_rl as Simulator  
from SimulationEnvironment.Environment import ReachTargetSimulationEnv
import models.ImmitationLearning as IL   

curr_env = ReachTargetSimulationEnv()
demos = curr_env.get_demos(2) 
agent = IL.ImmitationLearningAgent() 
agent.injest_demonstrations(demos)   
agent.train_agent(100)
# del curr_env
curr_env = ReachTargetSimulationEnv(headless=False)
curr_env.run_trained_agent(agent)
