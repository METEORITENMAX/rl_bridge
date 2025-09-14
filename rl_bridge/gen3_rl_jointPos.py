#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import torch
import numpy as np
import os
from sensor_msgs.msg import JointState
from control_msgs.action import GripperCommand
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformListener, Buffer
import tf2_geometry_msgs
import sys
sys.path.append('/home/max/robosuite_ws/robosuite_training_examples')  # Directory containing networks.py
from networks import TD3, Actor, Critic  # Import your network classes

class JointPublisher(Node):
    def __init__(self, joint_names, actor_model_path="tmp/kinova_td3/best_actor.pth"):
        super().__init__('joint_publisher_rl')
        self.joint_names = joint_names

        # Initialize RL agent using the full TD3 class
        # These dimensions should match your training setup
        state_dim = 57  # Match the trained model's state dimension
        action_dim = 7   # Match the trained model's action dimension (7, not 6)
        max_action = 1.0

        self.agent = TD3(state_dim, action_dim, max_action)

        # Load the trained actor model
        if os.path.exists(actor_model_path):
            self.agent.actor.load_state_dict(torch.load(actor_model_path, map_location='cpu', weights_only=True))
            self.agent.actor.eval()
            self.get_logger().info(f"Successfully loaded actor model from: {actor_model_path}")
        else:
            self.get_logger().error(f"Failed to load actor model from: {actor_model_path}")
            self.get_logger().info("Using random actions instead")
            self.agent = None

        # Initialize state (you may need to adjust this based on your actual state representation)
        self.current_state = np.zeros(state_dim)
        self.current_joint_velocities = [0.0] * 7  # Kinova3 has 7 joints

        # TF2 for object detection and transforms
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Object state (will be updated from TF or sensors)
        self.object_position = np.array([0.0, 0.0, 0.0])  # x, y, z
        self.object_orientation = np.array([0.0, 0.0, 0.0, 1.0])  # quaternion x, y, z, w
        self.gripper_position = np.array([0.0, 0.0, 0.0])  # end-effector position
        self.gripper_orientation = np.array([0.0, 0.0, 0.0, 1.0])  # end-effector orientation

        # ROS2 setup
        self.publisher = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10
        )

        # Subscribe to joint states to get current robot state
        self.joint_state_subscriber = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # Initialize gripper action client
        self.gripper_client = ActionClient(self, GripperCommand, '/robotiq_gripper_controller/gripper_cmd')

        self.current_joint_positions = [0.0] * len(joint_names)

        # Wait for controller to subscribe
        self.get_logger().info("Waiting for joint_trajectory_controller...")
        self.timer = self.create_timer(0.1, self.check_and_send)

        # Timer for updating transforms and object detection
        self.tf_timer = self.create_timer(0.05, self.update_transforms)  # 20Hz for TF updates

    def joint_state_callback(self, msg):
        """Update current joint positions from joint state feedback"""
        try:
            # Map joint names to positions and velocities
            for i, joint_name in enumerate(self.joint_names):
                if joint_name in msg.name:
                    joint_idx = msg.name.index(joint_name)
                    self.current_joint_positions[i] = msg.position[joint_idx]
                    if len(msg.velocity) > joint_idx:
                        self.current_joint_velocities[i] = msg.velocity[joint_idx]

            # Update full state representation to match training (61 dimensions)
            self.update_full_state()

        except Exception as e:
            self.get_logger().warn(f"Error updating joint state: {e}")

    def update_transforms(self):
        """Update object and end-effector transforms"""
        try:
            # Get end-effector position (adjust frame names as needed for your setup)
            try:
                ee_transform = self.tf_buffer.lookup_transform('base_link', 'end_effector_link', rclpy.time.Time())
                self.gripper_position = np.array([
                    ee_transform.transform.translation.x,
                    ee_transform.transform.translation.y,
                    ee_transform.transform.translation.z
                ])
                self.gripper_orientation = np.array([
                    ee_transform.transform.rotation.x,
                    ee_transform.transform.rotation.y,
                    ee_transform.transform.rotation.z,
                    ee_transform.transform.rotation.w
                ])
            except Exception:
                pass  # TF not available yet

            # Get object position (you'll need to publish this from Webots or vision system)
            try:
                obj_transform = self.tf_buffer.lookup_transform('base_link', 'target_object', rclpy.time.Time())
                self.object_position = np.array([
                    obj_transform.transform.translation.x,
                    obj_transform.transform.translation.y,
                    obj_transform.transform.translation.z
                ])
                self.object_orientation = np.array([
                    obj_transform.transform.rotation.x,
                    obj_transform.transform.rotation.y,
                    obj_transform.transform.rotation.z,
                    obj_transform.transform.rotation.w
                ])
            except Exception:
                # If no object TF available, use a default position or last known position
                pass

        except Exception as e:
            self.get_logger().debug(f"TF lookup failed: {e}")

    def update_full_state(self):
        """Update the full 61-dimensional state to match training setup EXACTLY"""
        try:
            # Compose the 57-dim state vector in the correct order
            joint_pos = np.array(self.current_joint_positions[:6])  # (6,)
            joint_pos_cos = np.cos(joint_pos)  # (6,)
            joint_pos_sin = np.sin(joint_pos)  # (6,)
            joint_vel = np.array(self.current_joint_velocities[:6])  # (6,)
            ee_pos = self.gripper_position[:3]  # (3,)
            ee_quat = self.gripper_orientation[:4]  # (4,)
            ee_quat_site = np.zeros(4)  # (4,) placeholder, update if available
            gripper_qpos = np.zeros(6)  # (6,) placeholder, update if available
            gripper_qvel = np.zeros(6)  # (6,) placeholder, update if available
            cube_pos = self.object_position[:3]  # (3,)
            cube_quat = self.object_orientation[:4]  # (4,)
            gripper_to_cube_pos = cube_pos - ee_pos  # (3,)

            # ToDo: concatenate state
            state = []
            self.current_state = state
            if hasattr(self, '_state_debug_count'):
                self._state_debug_count += 1
            else:
                self._state_debug_count = 1
            if self._state_debug_count <= 3:
                self.get_logger().info(f"State reconstruction (call {self._state_debug_count}):")
                self.get_logger().info(f"  First 10 state values: {self.current_state[:10]}")
                self.get_logger().info(f"  Total state dim: {len(self.current_state)}")
        except Exception as e:
            self.get_logger().warn(f"Error updating full state: {e}")
            self.current_state = np.zeros(57)

    def set_target_object_position(self, x, y, z, qx=0.0, qy=0.0, qz=0.0, qw=1.0):
        """Manually set target object position if TF is not available"""
        self.object_position = np.array([x, y, z])
        self.object_orientation = np.array([qx, qy, qz, qw])
        self.get_logger().info(f"Target object position set to: [{x:.3f}, {y:.3f}, {z:.3f}]")

    def check_and_send(self):
        if self.publisher.get_subscription_count() > 0:
            self.get_logger().info("Controller detected! Starting RL-based control...")

            # Cancel the initial connection check timer
            self.timer.cancel()

            # Set up timer for RL control loop (e.g., every 100ms)
            self.control_timer = self.create_timer(0.1, self.rl_control_step)

    def rl_control_step(self):
        """Get action from RL model and send to robot (joint deltas only, no IK/pose control)"""
        try:
            if self.agent is not None:
                # Get action from trained model (should be joint deltas)
                action = self.agent.select_action(self.current_state)

                # ToDo: Apply action as delta to current joint positions
                # target_positions = self.current_joint_positions + delta action
                # Pass target_positions to send_joint_positions(positions)

                # Optionally, send gripper command if action includes gripper
                if len(action) > 6:
                    gripper_action = action[6]
                    self.send_gripper_command_from_rl(gripper_action)
            else:
                # Fallback: random small movements if model failed to load
                random_deltas = np.random.uniform(-0.05, 0.05, len(self.joint_names))
                target_positions = [pos + delta for pos, delta in zip(self.current_joint_positions, random_deltas)]
                self.send_joint_positions(target_positions)
        except Exception as e:
            self.get_logger().error(f"Error in RL control step: {e}")

    def send_gripper_command_from_rl(self, gripper_action):
        """Send gripper command based on RL action (-1=open, +1=close)"""
        try:
            # ToDo: Map RL action to gripper position
            # Convert RL action (-1 to +1) to gripper position (0.0 to 0.7)
            # -1 (RL) -> 0.0 (fully open), +1 (RL) -> 0.7 (fully closed, Webots limit)
            gripper_position = gripper_action  # Maps [-1,1] to [0.0, 0.7]

            goal_msg = GripperCommand.Goal()
            goal_msg.command.position = gripper_position
            goal_msg.command.max_effort = 0.0  # Let controller decide effort

            if self.gripper_client.server_is_ready():
                self.gripper_client.send_goal_async(goal_msg)
                self.get_logger().info(f"RL Gripper command: action={gripper_action:.3f} -> position={gripper_position:.3f}")
            else:
                self.get_logger().warn("Gripper action server not ready")

        except Exception as e:
            self.get_logger().error(f"Error sending gripper command: {e}")

    def send_gripper_command(self, gripper_position):
        """Send gripper command based on position (0.0 = open, 0.7 = closed)"""
        try:
            goal_msg = GripperCommand.Goal()
            goal_msg.command.position = gripper_position
            goal_msg.command.max_effort = 0.0  # Let controller decide effort

            if self.gripper_client.server_is_ready():
                self.gripper_client.send_goal_async(goal_msg)
                self.get_logger().info(f"Gripper command sent: position={gripper_position:.3f}")
            else:
                self.get_logger().warn("Gripper action server not ready")

        except Exception as e:
            self.get_logger().error(f"Error sending gripper command: {e}")

    def send_joint_positions(self, positions, duration_sec=0.1):
        """Send joint trajectory to the controller"""
        traj_msg = JointTrajectory()
        traj_msg.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = int(duration_sec)
        point.time_from_start.nanosec = int((duration_sec % 1) * 1e9)
        traj_msg.points.append(point)

        # Publish the trajectory
        self.publisher.publish(traj_msg)

    def quaternion_to_euler(self, quat):
        """Convert quaternion to euler angles (simplified)"""
        # This is a simplified version - for production use scipy.spatial.transform
        x, y, z, w = quat
        roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        pitch = np.arcsin(2*(w*y - z*x))
        yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        return np.array([roll, pitch, yaw])

    def euler_to_quaternion(self, euler):
        """Convert euler angles to quaternion (simplified)"""
        # This is a simplified version - for production use scipy.spatial.transform
        roll, pitch, yaw = euler
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return np.array([x, y, z, w])



def main():
    rclpy.init()
    # Kinova3 has 7 joints (not 6)
    joint_names = ['joint_1','joint_2','joint_3','joint_4','joint_5','joint_6']

    # Path to your trained actor model
    actor_model_path = "/home/max/robosuite_ws/robosuite_training_examples/tmp/kinova_td3/best_actor.pth"  # Adjust this path as needed

    node = JointPublisher(joint_names, actor_model_path)

    # Example: Set a target object position manually (adjust coordinates for your setup)
    # You can call this from Webots or update it based on vision/sensors
    node.set_target_object_position(x=0.3, y=0.1, z=0.1)  # 30cm forward, 10cm right, 10cm up

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
