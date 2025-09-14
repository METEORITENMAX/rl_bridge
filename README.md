# rl_bridge

Interface for executing models with moveit on Kinova Gen3

# Requirements
Install requirements and kinova gen3 in webots from here:
[webots_ros2_gen3](https://github.com/skpawar1305/webots_ros2_gen3)

#

Launch `robot_launch.py` and `moveit_launch.py` for starting the gen3.

Run `test_jointPos` for testing joint trajectory controller.

Run `gen3_rl_jointPos` to execute your model with joint trajectory controller.

Therefore change:
```sys.path.append('/home/user/robosuite_ws/robosuite_training_examples')```
to the folder of your `network.py`
```
    actor_model_path = "/home/max/robosuite_ws/robosuite_training_examples/tmp/kinova_td3/best_actor.pth"  # Adjust this path as needed
```
to the path where your model is.


