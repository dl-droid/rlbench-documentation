from metaflow import FlowSpec, step


class TrainingSimulatorFlow(FlowSpec):

    @step
    def start(self):
        print("Importing data in this step")
        self.num_demos = 2
        self.num_epochs = 10
        self.next(self.train)

    @step
    def train(self):
        # todo : Create an image for training 
        from SimulationEnvironment.Environment import ReachTargetSimulationEnv
        import models.ImmitationMutationConv as IL   

        curr_env = ReachTargetSimulationEnv(dataset_root='/tmp/rlbench_data')
        # Set image_paths=False when loading dataset from file. 
        demos = curr_env.get_demos(self.num_demos,live_demos=False,image_paths=False) 
        agent = IL.ImmitationLearningConvolvingMutantAgent() 
        # agent.load_model('SavedModels/ImmitationLearningConvolvingMutantAgent-2020-04-05-04-18.pt')
        agent.injest_demonstrations(demos)   
        loss = agent.train_agent(self.num_epochs)
        self.loss = loss
        self.model = agent.neural_network.state_dict()
        self.optimizer = agent.optimizer.state_dict()
        self.next(self.end)

    @step
    def end(self):
        print("Done Computation")

if __name__ == '__main__':
    TrainingSimulatorFlow()