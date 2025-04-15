# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to load and simulate a biped robot in IsaacLab.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/demos/kuavo_robot_mpc/kuavo_locamotion.py
"""

"""Launch Isaac Sim Simulator first."""

import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Demo for loading biped robot.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ImuCfg # IMU设置

# 导入你的机器人配置
from kuavo_cfg import KINOVA_ROBOTIQ, KINOVA_ROBOTIQ_HPD

""" ROS ROBOT CONTROL """
import rospy
from kuavo_msgs.msg import jointCmd    # /joint_cmd
from kuavo_msgs.msg import sensorsData # /sensor_data_raw
DEBUG_FLAG = True

rospy.init_node('isaac_lab_kuavo_robot_mpc', anonymous=True) # 随机后缀

"""
    ImuData(
    pos_w=tensor([[-0.0819, -0.0140,  0.4382]], device='cuda:0'), 
    quat_w=tensor([[ 0.9277, -0.2014, -0.3001, -0.0936]], device='cuda:0'), 
    lin_vel_b=tensor([[-0.5847, -0.0380, -0.1088]], device='cuda:0'), 
    ang_vel_b=tensor([[-1.0506, -2.7600, -0.7210]], device='cuda:0'), 
    lin_acc_b=tensor([[-5.9137,  1.5053, -3.6032]], device='cuda:0'), 
    ang_acc_b=tensor([[ 1.4921, -2.1454,  6.5448]], device='cuda:0'))

    Received linear velocity:  tensor([[-0.1597, -0.1628, -0.0402]], device='cuda:0')
    Received angular velocity:  tensor([[-0.3645,  0.3846,  0.2246]], device='cuda:0')
    Received linear acceleration:  tensor([[-4.7626,  1.2978, -4.7949]], device='cuda:0')
    Received angular acceleration:  tensor([[-11.0339,  20.4661,  27.7198]], device='cuda:0')
    Received linear velocity:  [[-0.15968865156173706, -0.16275319457054138, -0.04015269875526428]]
    Received angular velocity:  [[-0.3644615709781647, 0.38461756706237793, 0.22463053464889526]]
    Received linear acceleration:  [[-4.76259183883667, 1.2977550029754639, -4.7949113845825195]]
    Received angular acceleration:  [[-11.033914566040039, 20.466079711914062, 27.719755172729492]]
"""
import json
import os

class KuavoRobotController():
    """
    # 45 - 机器人
    [2, 3, 7, 8, 12, 13, 16, 17, 20, 21, 24, 25, 26, 27] # 手
    [0, 1, 5, 6, 10, 11, 14, 15, 18, 19, 22, 23] # 脚
    [4, 9] # 头
    """
    def __init__(self):
        self.robot_sensor_data_pub = rospy.Publisher('/sensors_data_raw', sensorsData, queue_size=10)
        self.robot_joint_cmd_sub = rospy.Subscriber('/joint_cmd', jointCmd, self.joint_cmd_callback)

        # Load joint configurations from JSON file
        config_path = os.path.join(os.path.dirname(__file__), "config", "joint_name.json")
        try:
            with open(config_path, 'r') as f:
                joint_config = json.load(f)
                self.arm_joints = joint_config["arm_joints"]
                self.leg_joints = joint_config["leg_joints"]
                self.head_joints = joint_config["head_joints"]
        except Exception as e:
            print(f"Error loading joint configuration: {e}")
            # Fallback to default values if JSON loading fails
            self.arm_joints = []
            self.leg_joints = []
            self.head_joints = []

        # Initialize empty indices lists
        self._arm_idx = []
        self._leg_idx = []
        self._head_idx = []

    def setup_joint_indices(self, joint_names):
        """Setup joint indices based on joint names"""
        # Find indices for arm joints
        self._arm_idx = [i for i, name in enumerate(joint_names) if name in self.arm_joints]
        
        # Find indices for leg joints
        self._leg_idx = [i for i, name in enumerate(joint_names) if name in self.leg_joints]
        
        # Find indices for head joints
        self._head_idx = [i for i, name in enumerate(joint_names) if name in self.head_joints]

        if DEBUG_FLAG:
            print("Arm joint indices:", self._arm_idx)
            print("Leg joint indices:", self._leg_idx)
            print("Head joint indices:", self._head_idx)

    def joint_cmd_callback(self, joint_cmd):
        pass

    def update_sensor_data(self, lin_vel_b, ang_vel_b, lin_acc_b, ang_acc_b, quat_w, joint_pos, joint_vel):
        """
        lin_vel_b = scene["imu_base"].data.lin_vel_b.tolist()  # 线速度
        ang_vel_b = scene["imu_base"].data.ang_vel_b.tolist()  # 角速度
        lin_acc_b = scene["imu_base"].data.lin_acc_b.tolist()  # 线加速度
        ang_acc_b = scene["imu_base"].data.ang_acc_b.tolist()  # 角加速度
        """
        # IMU数据组合
        sensor_data = sensorsData()

        # 状态时间更新
        current_time = rospy.Time.now()
        sensor_data.header.stamp = current_time
        sensor_data.header.frame_id = "world"  # 设置适当的frame_id
        sensor_data.sensor_time = current_time

        # IMU数据
        sensor_data.imu_data.gyro.x = ang_vel_b[0][0]   # ang_vel
        sensor_data.imu_data.gyro.y = ang_vel_b[0][1]  # ang_vel
        sensor_data.imu_data.gyro.z = ang_vel_b[0][2]  # ang_vel
        sensor_data.imu_data.acc.x = lin_acc_b[0][0]  # lin_acc
        sensor_data.imu_data.acc.y = lin_acc_b[0][1]  # lin_acc
        sensor_data.imu_data.acc.z = lin_acc_b[0][2]  # lin_acc
        sensor_data.imu_data.quat.w = quat_w[0][0]  # 旋转矩阵
        sensor_data.imu_data.quat.x = quat_w[0][1]  # 旋转矩阵
        sensor_data.imu_data.quat.y = quat_w[0][2]  # 旋转矩阵
        sensor_data.imu_data.quat.z = quat_w[0][3]  # 旋转矩阵

        # 发布数据
        self.robot_sensor_data_pub.publish(sensor_data)

@configclass
class BipedSceneCfg(InteractiveSceneCfg):
    """Scene configuration with biped robot."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0
            )
        )
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # 使用你的机器人配置
    robot = KINOVA_ROBOTIQ  # 或者使用 KINOVA_ROBOTIQ_HPD
    robot.prim_path = "{ENV_REGEX_NS}/Robot"

    # 添加IMU传感器
    """
        固定的偏置为 +9.81 m/s^2
        imu_RF = ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/LF_FOOT", debug_vis=True) # 默认消除了重力
        imu_LF = ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/RF_FOOT", gravity_bias=(0, 0, 0), debug_vis=True) # 没有消除重力
    """
    imu_base = ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/base_link", gravity_bias=(0, 0, 0), debug_vis=True) # 没有消除重力

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, robot: KuavoRobotController):
    """Run the simulator."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    print("sim_dt: ", sim_dt)
    sim_time = 0.0
    count = 0

    body_names = scene["robot"].data.body_names   # 包含所有fix固定的joint
    joint_names = scene["robot"].data.joint_names # 只包含可活动的joint
    robot.setup_joint_indices(joint_names)
    
    # Simulate physics
    while simulation_app.is_running():
        # Reset every 500 steps
        if count % 500 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            root_state = scene["robot"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            scene["robot"].write_root_pose_to_sim(root_state[:, :7])
            scene["robot"].write_root_velocity_to_sim(root_state[:, 7:])
            # set joint positions
            joint_pos = scene["robot"].data.default_joint_pos.clone()
            joint_vel = scene["robot"].data.default_joint_vel.clone()
            scene["robot"].write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting robot state...")
        # write data to sim
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        scene.update(sim_dt)

        # 获取IMU数据
        lin_vel_b = scene["imu_base"].data.lin_vel_b.tolist()  # 线速度
        ang_vel_b = scene["imu_base"].data.ang_vel_b.tolist()  # 角速度
        lin_acc_b = scene["imu_base"].data.lin_acc_b.tolist()  # 线加速度
        ang_acc_b = scene["imu_base"].data.ang_acc_b.tolist()  # 角加速度
        quat_w = scene["imu_base"].data.quat_w.tolist()  # 旋转矩阵

        # 获取机器人的数据
        joint_pos = scene["robot"].data.joint_pos.tolist()
        joint_vel = scene["robot"].data.joint_vel.tolist()

        if DEBUG_FLAG:
            # print("-------------------------------")
            # print(scene["imu_base"].data)
            # print("Received linear velocity: ", scene["imu_base"].data.lin_vel_b)
            # print("Received angular velocity: ", scene["imu_base"].data.ang_vel_b)
            # print("Received linear acceleration: ", scene["imu_base"].data.lin_acc_b)
            # print("Received angular acceleration: ", scene["imu_base"].data.ang_acc_b)

            # print("Received linear velocity: ", lin_vel_b)
            # print("Received angular velocity: ", ang_vel_b)
            # print("Received linear acceleration: ", lin_acc_b)
            # print("Received angular acceleration: ", ang_acc_b)

            # print("joint_pos: ", joint_pos)
            # print("joint_vel: ", joint_vel)
            # print("body_names: ", body_names)
            # print("joint_names: ", joint_names)
            # print("-------------------------------")
            pass

        # 更新传感器数据
        robot.update_sensor_data(lin_vel_b, ang_vel_b, lin_acc_b, ang_acc_b, quat_w, joint_pos, joint_vel)

        # TODO: 更新MPC控制器

def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    
    # Set main camera
    sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])
    
    # Create scene
    scene_cfg = BipedSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    # 创建机器人控制实例
    robot = KuavoRobotController()

    # Reset simulation
    sim.reset()
    
    # Now we are ready!
    print("[INFO]: Setup complete...")
    
    # Run the simulator
    run_simulator(sim, scene, robot)

if __name__ == "__main__":
    main()
    simulation_app.close()