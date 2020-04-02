from deep_learning_rl import get_demos  
import models.ImmitationLearning as IL   


demos = get_demos(2) 
agent = IL.ImmitationLearningAgent() 
agent.injest_demonstrations(demos)   
agent.train_agent(100)