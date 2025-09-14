#!/usr/bin/env python3
# This script is a copy of gen3_rl_moveit.py but uses MoveIt for IK.
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import torch
import numpy as np
import os
from sensor_msgs.msg import JointState
from control_msgs.action import GripperCommand
from geometry_msgs.msg import TransformStamped, PoseStamped
from tf2_ros import TransformListener, Buffer
import tf2_geometry_msgs
import sys
sys.path.append('/home/max/robosuite_ws/robosuite_training_examples')  # Directory containing networks.py
from networks import TD3, Actor, Critic  # Import your network classes

class JointPublisher(Node):
    def send_trajectory_to_moveit(self, joint_positions, duration_sec=1.0):
        # Send a trajectory to the /execute_trajectory action server (MoveIt)
        from moveit_msgs.action import ExecuteTrajectory
        from moveit_msgs.msg import RobotTrajectory
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        import rclpy
        import time
        # Build JointTrajectory
        jt = JointTrajectory()
        jt.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = joint_positions
        point.time_from_start.sec = int(duration_sec)
        point.time_from_start.nanosec = int((duration_sec % 1) * 1e9)
        jt.points.append(point)
        # Build RobotTrajectory
        robot_traj = RobotTrajectory()
        robot_traj.joint_trajectory = jt
        # Create action client
        self.moveit_action_client = ActionClient(self, ExecuteTrajectory, '/execute_trajectory')
        if not self.moveit_action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('MoveIt /execute_trajectory action server not available!')
            return
        goal_msg = ExecuteTrajectory.Goal()
        goal_msg.trajectory = robot_traj
        send_goal_future = self.moveit_action_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().error('MoveIt did not accept trajectory goal!')
            return
        self.get_logger().info('MoveIt accepted trajectory goal, waiting for result...')
        get_result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, get_result_future)
        result = get_result_future.result().result
        self.get_logger().info(f'MoveIt trajectory execution result: {result}')
    def __init__(self, joint_names, actor_model_path="/home/max/robosuite_ws/robosuite_training_examples/tmp/kinova_td3/best_actor.pth"):
        super().__init__('joint_publisher_rl')
        self.joint_names = joint_names
        state_dim = 57
        action_dim = 7
        max_action = 1.0
        self.agent = TD3(state_dim, action_dim, max_action)
        if os.path.exists(actor_model_path):
            self.agent.actor.load_state_dict(torch.load(actor_model_path, map_location='c' \
            'pu', weights_only=True))
            self.agent.actor.eval()
            self.get_logger().info(f"Successfully loaded actor model from: {actor_model_path}")
        else:
            self.get_logger().error(f"Failed to load actor model from: {actor_model_path}")
            self.get_logger().info("Using random actions instead")
            self.agent = None
        self.current_state = np.zeros(state_dim)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.object_position = np.array([0.0, 0.0, 0.0])
        self.object_orientation = np.array([0.0, 0.0, 0.0, 1.0])
        self.gripper_position = np.array([0.0, 0.0, 0.0])
        self.gripper_orientation = np.array([0.0, 0.0, 0.0, 1.0])
        self.current_joint_positions = [0.0] * len(self.joint_names)
        self.current_joint_velocities = [0.0] * len(self.joint_names)
        self.joint_state_received = False
        self.publisher = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10
        )
        self.joint_state_subscriber = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        self.gripper_client = ActionClient(self, GripperCommand, '/robotiq_gripper_controller/gripper_cmd')
        self.get_logger().info("Waiting for joint_trajectory_controller...")
        self.timer = self.create_timer(0.1, self.check_and_send)
        self.tf_timer = self.create_timer(0.05, self.update_transforms)

    def joint_state_callback(self, msg):
        try:
            # Only update for joints in self.joint_names (ignore gripper/fingers)
            for i, joint_name in enumerate(self.joint_names):
                if joint_name in msg.name:
                    joint_idx = msg.name.index(joint_name)
                    self.current_joint_positions[i] = msg.position[joint_idx]
                    if len(msg.velocity) > joint_idx:
                        self.current_joint_velocities[i] = msg.velocity[joint_idx]
            self.joint_state_received = True
            self.update_full_state()
        except Exception as e:
            self.get_logger().warn(f"Error updating joint state: {e}")

    def update_transforms(self):
        try:
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
                pass
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
                pass
        except Exception as e:
            self.get_logger().debug(f"TF lookup failed: {e}")

    def update_full_state(self):
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

            state = np.concatenate([
                joint_pos,            # [0:6]
                joint_pos_cos,        # [6:12]
                joint_pos_sin,        # [12:18]
                joint_vel,            # [18:24]
                ee_pos,               # [24:27]
                ee_quat,              # [27:31]
                ee_quat_site,         # [31:35] (placeholder)
                gripper_qpos,         # [35:41] (placeholder)
                gripper_qvel,         # [41:47] (placeholder)
                cube_pos,             # [47:50]
                cube_quat,            # [50:54]
                gripper_to_cube_pos   # [54:57]
            ])
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
        self.object_position = np.array([x, y, z])
        self.object_orientation = np.array([qx, qy, qz, qw])
        self.get_logger().info(f"Target object position set to: [{x:.3f}, {y:.3f}, {z:.3f}]")

    def check_and_send(self):
        if self.publisher.get_subscription_count() > 0:
            self.get_logger().info("Controller detected! Starting RL-based control...")
            self.timer.cancel()
            self.control_timer = self.create_timer(0.1, self.rl_control_step)

    def rl_control_step(self):
        try:
            if not self.joint_state_received:
                self.get_logger().warn("No joint state received yet, skipping control step.")
                return
            if self.agent is not None:
                full_action = self.agent.select_action(self.current_state)
                pose_deltas = full_action[:6]
                gripper_action = full_action[6]
                max_translation = 0.05
                max_rotation = 0.5
                pos_deltas = pose_deltas[:3] * max_translation
                rot_deltas = pose_deltas[3:] * max_rotation
                target_pos = self.gripper_position + pos_deltas
                current_euler = self.quaternion_to_euler(self.gripper_orientation)
                target_euler = current_euler + rot_deltas
                target_quat = self.euler_to_quaternion(target_euler)
                self.get_logger().info(f"RL Action - Pose Δ: {pos_deltas} | Rot Δ: {rot_deltas} | Gripper: {gripper_action:.3f}")
                # Use MoveIt for proper inverse kinematics
                self.send_pose_with_moveit(target_pos, target_quat)
                self.send_gripper_command_from_rl(gripper_action)
        except Exception as e:
            self.get_logger().error(f"Error in RL control step: {e}")

    def send_gripper_command_from_rl(self, gripper_action):
        try:
            gripper_position = 0.35 * (gripper_action + 1)
            goal_msg = GripperCommand.Goal()
            goal_msg.command.position = gripper_position
            goal_msg.command.max_effort = 0.0
            if self.gripper_client.server_is_ready():
                self.gripper_client.send_goal_async(goal_msg)
                self.get_logger().info(f"RL Gripper command: action={gripper_action:.3f} -> position={gripper_position:.3f}")
            else:
                self.get_logger().warn("Gripper action server not ready")
        except Exception as e:
            self.get_logger().error(f"Error sending gripper command: {e}")

    def send_joint_positions(self, positions, duration_sec=0.1):
        traj_msg = JointTrajectory()
        traj_msg.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = int(duration_sec)
        point.time_from_start.nanosec = int((duration_sec % 1) * 1e9)
        traj_msg.points.append(point)
        self.publisher.publish(traj_msg)

    def quaternion_to_euler(self, quat):
        x, y, z, w = quat
        roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        pitch = np.arcsin(2*(w*y - z*x))
        yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        return np.array([roll, pitch, yaw])

    def euler_to_quaternion(self, euler):
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

    def send_pose_with_moveit(self, target_pos, target_quat):
        """
        Proper pose control using MoveIt inverse kinematics via /compute_ik service, then execute via MoveIt action server.
        """
        from moveit_msgs.srv import GetPositionIK
        from moveit_msgs.msg import PositionIKRequest
        from geometry_msgs.msg import PoseStamped
        import builtin_interfaces.msg
        import rclpy
        try:
            # Prepare IK request
            ik_req = PositionIKRequest()
            ik_req.group_name = 'manipulator'  # Change if your group is different
            ik_req.ik_link_name = 'end_effector_link'  # Change if your EE link is different
            ik_req.pose_stamped = PoseStamped()
            ik_req.pose_stamped.header.frame_id = 'base_link'
            ik_req.pose_stamped.header.stamp = self.get_clock().now().to_msg()
            ik_req.pose_stamped.pose.position.x = float(target_pos[0])
            ik_req.pose_stamped.pose.position.y = float(target_pos[1])
            ik_req.pose_stamped.pose.position.z = float(target_pos[2])
            ik_req.pose_stamped.pose.orientation.x = float(target_quat[0])
            ik_req.pose_stamped.pose.orientation.y = float(target_quat[1])
            ik_req.pose_stamped.pose.orientation.z = float(target_quat[2])
            ik_req.pose_stamped.pose.orientation.w = float(target_quat[3])

            # Create IK service client
            cli = self.create_client(GetPositionIK, '/compute_ik')
            if not cli.wait_for_service(timeout_sec=2.0):
                self.get_logger().error('MoveIt IK service /compute_ik not available!')
                return

            req = GetPositionIK.Request()
            req.ik_request = ik_req

            future = cli.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
            if not future.done():
                self.get_logger().error('IK service call timed out!')
                return
            resp = future.result()
            if resp.error_code.val != 1:
                self.get_logger().error(f'MoveIt IK failed with code {resp.error_code.val}')
                return
            joint_state = resp.solution.joint_state
            # Map returned joint positions to your joint order
            joint_positions = [0.0] * len(self.joint_names)
            for i, name in enumerate(self.joint_names):
                if name in joint_state.name:
                    idx = joint_state.name.index(name)
                    joint_positions[i] = joint_state.position[idx]
            self.get_logger().info(f"MoveIt IK solution: {joint_positions}")
            # Send trajectory to MoveIt for execution
            self.send_trajectory_to_moveit(joint_positions)
        except Exception as e:
            self.get_logger().error(f"Error in send_pose_with_moveit: {e}")


def main():
    rclpy.init()
    joint_names = ['joint_1','joint_2','joint_3','joint_4','joint_5','joint_6']
    actor_model_path = "/home/max/robosuite_ws/robosuite_training_examples/tmp/kinova_td3/best_actor.pth"
    node = JointPublisher(joint_names, actor_model_path)
    node.set_target_object_position(x=0.3, y=0.1, z=0.1)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
