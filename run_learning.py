from deep_learning_rl import get_demos  
demos = get_demos(2) 
import models.ImmitationLearning as IL   
agent = IL.ImmitationLearningAgent() 
agent.injest_demonstrations(demos)   
agent.train_agent(100)