#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

class JointPublisher(Node):
    def __init__(self, joint_names):
        super().__init__('joint_publisher')
        self.joint_names = joint_names
        self.current_state = 0  # 0 for all zeros, 1 for all 2.0s
        self.positions = [0.0] * len(joint_names)  # Start with all joints at 0
        self.publisher = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10
        )

        # Wait for controller to subscribe
        self.get_logger().info("Waiting for joint_trajectory_controller...")
        self.timer = self.create_timer(0.1, self.check_and_send)

    def check_and_send(self):
        if self.publisher.get_subscription_count() > 0:
            self.get_logger().info("Controller detected! Sending trajectory...")
            self.send_joint_positions(self.positions)

            # Toggle state and update positions for next time
            self.toggle_positions()

            # Set up timer for next movement (e.g., every 3 seconds)
            self.timer.cancel()  # Cancel the initial connection check timer
            self.movement_timer = self.create_timer(3.0, self.send_next_trajectory)

    def toggle_positions(self):
        """Toggle between all joints at 0.0 and all joints at 2.0"""
        if self.current_state == 0:
            # Switch to all 2.0s
            self.positions = [2.0] * len(self.joint_names)
            self.current_state = 1
            self.get_logger().info("Next position: all joints to 2.0")
        else:
            # Switch to all 0.0s
            self.positions = [0.0] * len(self.joint_names)
            self.current_state = 0
            self.get_logger().info("Next position: all joints to 0.0")

    def send_next_trajectory(self):
        """Send the next trajectory and toggle for the following one"""
        self.get_logger().info(f"Sending trajectory: {self.positions}")
        self.send_joint_positions(self.positions)
        self.toggle_positions()

    def send_joint_positions(self, positions, duration_sec=1.0):
        traj_msg = JointTrajectory()
        traj_msg.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = int(duration_sec)
        point.time_from_start.nanosec = int((duration_sec % 1) * 1e9)
        traj_msg.points.append(point)

        # Publish multiple times to ensure controller receives it
        for _ in range(5):
            self.publisher.publish(traj_msg)

def main():
    rclpy.init()
    joint_names = ['joint_1','joint_2','joint_3','joint_4','joint_5','joint_6']

    node = JointPublisher(joint_names)

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
