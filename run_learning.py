import deep_learning_rl as Simulator  
import models.ImmitationLearning as IL   


demos = Simulator.get_demos(20) 
agent = IL.ImmitationLearningAgent() 
agent.injest_demonstrations(demos)   
agent.train_agent(100)

Simulator.simulate_trained_agent(agent)
