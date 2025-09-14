#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import PositionIKRequest, RobotTrajectory
from moveit_msgs.msg import RobotState
from sensor_msgs.msg import JointState
from moveit_msgs.action import ExecuteTrajectory
from control_msgs.action import GripperCommand
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

class IKTestNode(Node):
    def __init__(self):
        super().__init__('ik_test_node')
        # --- ADJUST THESE TO YOUR ROBOT ---
        self.joint_names = ['joint_1','joint_2','joint_3','joint_4','joint_5','joint_6']
        self.group_name = 'manipulator'
        self.ik_link_name = 'end_effector_link'
        self.planning_frame = 'base_link'  # change if your MoveIt uses a different frame
        # ----------------------------------
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        self.exec_client = ActionClient(self, ExecuteTrajectory, '/execute_trajectory')
        self.gripper_client = ActionClient(self, GripperCommand, '/robotiq_gripper_controller/gripper_cmd')
        self.joint_state_msg = None
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        # single-shot timer to run once after we have joint states
        self.timer = self.create_timer(1.0, self.run_test)
        self.test_done = False

    def joint_state_callback(self, msg: JointState):
        if msg.name and msg.position:
            self.joint_state_msg = msg

    def run_test(self):
        if self.test_done:
            return
        if self.joint_state_msg is None:
            self.get_logger().info('Waiting for /joint_states...')
            return

        # --- target pose: actual pose of end_effector_link in base_link frame ---
        target_pose = PoseStamped()
        target_pose.header.frame_id = self.planning_frame
        target_pose.header.stamp = self.get_clock().now().to_msg()
        target_pose.pose.position.x = 0.080
        target_pose.pose.position.y = 0.001
        target_pose.pose.position.z = 1.113
        target_pose.pose.orientation.x = -0.242
        target_pose.pose.orientation.y = -0.242
        target_pose.pose.orientation.z = 0.664
        target_pose.pose.orientation.w = 0.665

        # --- Build IK request ---
        ik_req = PositionIKRequest()
        ik_req.group_name = self.group_name
        ik_req.ik_link_name = self.ik_link_name
        ik_req.pose_stamped = target_pose

        # Use current joint state from /joint_states as the IK seed, filtered and ordered
        seed_js = JointState()
        seed_js.header.frame_id = self.planning_frame
        seed_js.name = []
        seed_js.position = []
        seed_js.velocity = []
        seed_js.effort = []
        for n in self.joint_names:
            if n in self.joint_state_msg.name:
                idx = self.joint_state_msg.name.index(n)
                seed_js.name.append(n)
                seed_js.position.append(self.joint_state_msg.position[idx])
                if len(self.joint_state_msg.velocity) > idx:
                    seed_js.velocity.append(self.joint_state_msg.velocity[idx])
                else:
                    seed_js.velocity.append(0.0)
                if len(self.joint_state_msg.effort) > idx:
                    seed_js.effort.append(self.joint_state_msg.effort[idx])
                else:
                    seed_js.effort.append(0.0)
            else:
                self.get_logger().error(f"Joint {n} not found in joint_states! Aborting IK request.")
                self.test_done = True
                return
        ik_req.robot_state = RobotState()
        ik_req.robot_state.joint_state = seed_js
        ik_req.robot_state.is_diff = False

        # Set additional IK request fields
        from builtin_interfaces.msg import Duration
        ik_req.timeout = Duration(sec=3, nanosec=0)  # 3 second timeout
        ik_req.avoid_collisions = False

        req = GetPositionIK.Request()
        req.ik_request = ik_req

        self.get_logger().info('Checking /compute_ik service availability (waiting up to 10s)...')
        if not self.ik_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error('IK service not available (service name: /compute_ik). Use `ros2 service list` to verify.')
            self.test_done = True
            return

        self.get_logger().info('Calling compute_ik...')
        future = self.ik_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)
        if not future.done():
            self.get_logger().error('compute_ik call timed out.')
            self.test_done = True
            return
        resp = future.result()
        if resp is None:
            self.get_logger().error('compute_ik returned None.')
            self.test_done = True
            return

        # Print full error code for debugging
        err = getattr(resp, 'error_code', None)
        self.get_logger().info(f'IK response error_code: {err}')
        # MoveIt uses 1 == SUCCESS; print the numeric val
        if not (hasattr(err, 'val') and err.val == 1):
            self.get_logger().error(f'IK failed (code={getattr(err,"val",None)}). Pose may be unreachable or joint names wrong.')
            self.test_done = True
            return

        joint_state = resp.solution.joint_state
        # construct ordered joint positions matching self.joint_names
        joint_positions = []
        for name in self.joint_names:
            if name in joint_state.name:
                idx = joint_state.name.index(name)
                joint_positions.append(joint_state.position[idx])
            else:
                self.get_logger().error(f'IK result missing joint {name}; aborting.')
                self.test_done = True
                return

        self.get_logger().info(f'IK solution: {joint_positions}')

        # --- Build a simple trajectory with one waypoint ---
        jt = JointTrajectory()
        jt.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = joint_positions
        # safe time_from_start — use Duration message
        point.time_from_start = Duration(sec=3, nanosec=0)
        jt.points = [point]
        robot_traj = RobotTrajectory()
        robot_traj.joint_trajectory = jt

        # --- Execute trajectory via action ---
        self.get_logger().info('Waiting for execute_trajectory action server (10s)...')
        if not self.exec_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('ExecuteTrajectory action server not available (server: /execute_trajectory). Use `ros2 action list` to verify.')
            self.test_done = True
            return

        goal_msg = ExecuteTrajectory.Goal()
        goal_msg.trajectory = robot_traj
        self.get_logger().info('Sending ExecuteTrajectory goal...')
        send_goal_future = self.exec_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future, timeout_sec=10.0)
        if not send_goal_future.done():
            self.get_logger().error('Send goal timed out.')
            self.test_done = True
            return
        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().error('ExecuteTrajectory server rejected goal.')
            self.test_done = True
            return

        self.get_logger().info('Goal accepted; waiting for result (20s)...')
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=20.0)
        if not result_future.done():
            self.get_logger().error('Getting result timed out.')
            self.test_done = True
            return
        result = result_future.result().result
        self.get_logger().info(f'ExecuteTrajectory result: {result}')

        # --- Optionally close gripper ---
        self.get_logger().info('Attempting to close gripper if server present...')
        if self.gripper_client.wait_for_server(timeout_sec=5.0):
            gripper_goal = GripperCommand.Goal()
            gripper_goal.command.position = 0.7
            gripper_goal.command.max_effort = 0.0
            gf = self.gripper_client.send_goal_async(gripper_goal)
            rclpy.spin_until_future_complete(self, gf, timeout_sec=5.0)
            gh = gf.result()
            if gh and gh.accepted:
                grf = gh.get_result_async()
                rclpy.spin_until_future_complete(self, grf, timeout_sec=5.0)
                self.get_logger().info('Gripper command completed.')
        else:
            self.get_logger().info('No gripper action server found; skipping.')

        self.get_logger().info('Test done.')
        self.test_done = True


def main():
    rclpy.init()
    node = IKTestNode()
    try:
        # spin the node until it completes (the node handles waiting for joint_states)
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
