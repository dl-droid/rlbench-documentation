# RLBench Documentation
To Use RLBench for simulated Training in Robotics there are some key components to Take care of. 

## Environment
- The `Environment` can be imported. Every environment takes an `ActionMode` and `ObservationConfig` which will help determine the inputs actions and the observations the environment will make. 
- Every `Observation` consists of the below data points which are captured in the demos. 
    ```python
    class Observation(object):
        """Storage for both visual and low-dimensional observations."""

        def __init__(self,
                    left_shoulder_rgb: np.ndarray,
                    left_shoulder_depth: np.ndarray,
                    left_shoulder_mask: np.ndarray,
                    right_shoulder_rgb: np.ndarray,
                    right_shoulder_depth: np.ndarray,
                    right_shoulder_mask: np.ndarray,
                    wrist_rgb: np.ndarray,
                    wrist_depth: np.ndarray,
                    wrist_mask: np.ndarray,
                    joint_velocities: np.ndarray,
                    joint_positions: np.ndarray,
                    joint_forces: np.ndarray,
                    gripper_open_amount: float,
                    gripper_pose: np.ndarray,
                    gripper_joint_positions: np.ndarray,
                    gripper_touch_forces: np.ndarray,
                    task_low_dim_state: np.ndarray):
    ```

- Observations for an experiment can be configured using `ObservationConfig`.
    ```python 
    obs_config = ObservationConfig()
    obs_config.set_all(False)
    obs_config.left_shoulder_camera.rgb = True
    obs_config.right_shoulder_camera.rgb = True
    ```
- `ArmActionMode.ABS_JOINT_VELOCITY` helps specify the what the action will mean. More action Modes available [here](https://github.com/stepjam/RLBench/blob/9f3bf886ce5d59d2eff8d9ec93ac49cb2b816b2f/rlbench/action_modes.py#L7). This basically means that when U provide an action it will expect it to mean what is selected in ArmActionMode.
    ```python
    from rlbench.environment import Environment
    from rlbench.action_modes import ArmActionMode, ActionMode
    action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
    env = Environment(
        action_mode, DATASET, obs_config, False)
    # for obs_config refer to Tasks 
    env.launch()
    ```

## Task
- Every `Environment` consists of tasks which can use Different Robots to do different things. 
    ```python
    from rlbench.tasks import ReachTarget
    task = env.get_task(ReachTarget)
    ```
- Every `Task` consists of a method called `get_demos`. The `get_demos` function will perform the `Task` and get the obs
ervations which achieve the task.

    ```python
    demos = task.get_demos(2, live_demos=True)  # -> List[List[Observation]] -> List[Observation] represents a individual Demonstration with every item in that List represnts a step in that Demonstration
    ```

## Immitation Learning Setup 
- To do immitation learning with RLBench, create an object that will expose methods for the model to `predict_actions` and `apply_gradient`. The `apply_gradient` method will be used to evaluate the loss on basis of the 
    ```python
    import pytorch
    class ImitationLearningModel(object):

        def __init__(self):
            # Get a crafted model from somewhere. 
            self.model = get_pytorch_model()

        def predict_action(self, batch):
            """
            Predict an action on basis of the Model.
            """
            return self.model.evaluate(batch) # Dimensions of each action should be according to ActionMode

        def apply_gradient(self, ground_truth_actions, predicted_actions):
            """
            Propagate Loss inside the Model
            """

            return self.model.propage_loss(ground_truth_actions,predicted_actions)
    ```
- Using an object like this and the getting the data gathered using `demos` from a `Task`. Run training for the Model predicting actions of the robot. 
    ```python
    imitation_learning_model = ImitationLearningModel()
    NUMBER_OF_DEMONSTRATIONS = 1000
    demos = task.get_demos(NUMBER_OF_DEMONSTRATIONS, live_demos=True)  # -> List[List[Observation]]
    demos = np.array(demos).flatten()
    # An example of using the demos to 'train' using behaviour cloning loss.
    for i in range(100):
        print("'training' iteration %d" % i)
        # Get a batch/episode/demo from full dataset. 
        batch = np.random.choice(demos, replace=False)
        # Get the values of the endeffector positions which can be used as training data
        # For Other options for observations/training datapoint to take check : Observation class
        effector_positions = [obs.gripper_joint_positions for obs in batch]
        # Predict the action on basis of the input provided to the model.
        predicted_actions = imitation_learning_model.predict_action(effector_positions)
        # Get Ground truth values of actions of the chosen batch.  
        ground_truth_actions = [obs.joint_velocities for obs in batch]
        # Apply the Gradients to model for next prediction in the training cycles. 
        imitation_learning_model.apply_gradient(ground_truth_actions, predicted_actions)
    ```

## Observations / Available Data

`Observation.joint_positions`: The intrinsic position of a joint. This is a one-dimensional value: if the joint is revolute, the rotation angle is returned, if the joint is prismatic, the translation amount is returned, etc. 

# Running the Code 

1. Prerequisites PyRep and RLBench
2. Running Dumb Immitation Learning Agent Training / Simulation:
    ```sh
    python run_learning.py
    ```

# TODO 
- [ ] Document Agents
- [ ] Document Observations/Available data
- [ ] Add Complete Package Install Scripts
- [ ] Add More deep learning approaches for task based training for agents. 