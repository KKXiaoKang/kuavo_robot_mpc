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
from kuavo_cfg import KINOVA_ROBOTIQ
# from kuavo_cfg import KINOVA_ROBOTIQ_HPD

""" ROS ROBOT CONTROL """
import rospy
from kuavo_msgs.msg import jointCmd    # /joint_cmd
from kuavo_msgs.msg import sensorsData # /sensor_data_raw
from std_srvs.srv import SetBool, SetBoolResponse  

DEBUG_FLAG = True

rospy.init_node('isaac_lab_kuavo_robot_mpc', anonymous=True) # 随机后缀

FIRST_TIME_FLAG = True
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
"""
            --- ocs2_idx:  0  ---- i : 0
            --- ocs2_idx:  1  ---- i : 1
            --- ocs2_idx:  5  ---- i : 2
            --- ocs2_idx:  6  ---- i : 3
            --- ocs2_idx:  10  ---- i : 4
            --- ocs2_idx:  11  ---- i : 5
            --- ocs2_idx:  14  ---- i : 6
            --- ocs2_idx:  15  ---- i : 7
            --- ocs2_idx:  18  ---- i : 8
            --- ocs2_idx:  19  ---- i : 9
            --- ocs2_idx:  22  ---- i : 10
            --- ocs2_idx:  23  ---- i : 11
            -- ocs2_idx:  2  ---- i : 12
            --- ocs2_idx:  3  ---- i : 13
            --- ocs2_idx:  7  ---- i : 14
            --- ocs2_idx:  8  ---- i : 15
            --- ocs2_idx:  12  ---- i : 16
            --- ocs2_idx:  13  ---- i : 17
            --- ocs2_idx:  16  ---- i : 18
            --- ocs2_idx:  17  ---- i : 19
            --- ocs2_idx:  20  ---- i : 20
            --- ocs2_idx:  21  ---- i : 21
            --- ocs2_idx:  24  ---- i : 22
            --- ocs2_idx:  25  ---- i : 23
            --- ocs2_idx:  26  ---- i : 24
            --- ocs2_idx:  27  ---- i : 25
            --- ocs2_idx:  4  ---- i : 26
            --- ocs2_idx:  9  ---- i : 27
"""

import json
import os

class KuavoRobotController():
    """
    # 45 - 机器人
    [2, 3, 7, 8, 12, 13, 
    16, 17, 20, 21, 24, 25, 
    26, 27] # 手
    zarm_l1_joint / zarm_r1_joint / zarm_l2_joint / zarm_r2_joint / zarm_l3_joint / zarm_r3_joint 
    zarm_l4_joint / zarm_r4_joint / zarm_l5_joint / zarm_r5_joint / zarm_l6_joint / zarm_r6_joint
    zarm_l7_joint / zarm_r7_joint
    
    [0,  1,  5,  6, 10, 11, 
    14, 15, 18, 19, 22, 23] # 脚 
    leg_l1_joint / leg_r1_joint / leg_l2_joint / leg_r2_joint / leg_l3_joint / leg_r3_joint 
    leg_l4_joint / leg_r4_joint / leg_l5_joint / leg_r5_joint / leg_l6_joint / leg_r6_joint

    [4, 9] # 头
    """
    def __init__(self):
        # 添加对scene的引用
        self.scene = None
        self.joint_cmd = None
        # state/cmd
        self.robot_sensor_data_pub = rospy.Publisher('/sensors_data_raw', sensorsData, queue_size=10)
        self.robot_joint_cmd_sub = rospy.Subscriber('/joint_cmd', jointCmd, self.joint_cmd_callback)

        #  仿真开始标志
        self.sim_running = True
        self.sim_start_srv = rospy.Service('sim_start', SetBool, self.sim_start_callback)

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
        self._ocs2_idx = []

    def sim_start_callback(self, req):
        """
        仿真启动服务的回调函数
        Args:
            req: SetBool请求，data字段为True表示启动仿真，False表示停止仿真
        Returns:
            SetBoolResponse: 服务响应
        """
        response = SetBoolResponse()
        
        self.sim_running = req.data

        if req.data:
            rospy.loginfo("Simulation started")
        else:
            rospy.loginfo("Simulation stopped")
        
        response.success = True
        response.message = "Simulation control successful"
        
        return response

    def setup_joint_indices(self, joint_names):
        """Setup joint indices based on joint names"""
        # Find indices for arm joints
        self._arm_idx = [i for i, name in enumerate(joint_names) if name in self.arm_joints]
        
        # Find indices for leg joints
        self._leg_idx = [i for i, name in enumerate(joint_names) if name in self.leg_joints]
        
        # Find indices for head joints
        self._head_idx = [i for i, name in enumerate(joint_names) if name in self.head_joints]

        # ocs2 idx 
        self._ocs2_idx = self._leg_idx + self._arm_idx + self._head_idx # 脚/手/头

        if DEBUG_FLAG:
            print("Arm joint indices:", self._arm_idx)
            print("Leg joint indices:", self._leg_idx)
            print("Head joint indices:", self._head_idx)    
            print("OCS2 joint indices:", self._ocs2_idx)

    def joint_cmd_callback(self, joint_cmd):
        """处理接收到的关节力矩命令
        
        Args:
            joint_cmd (jointCmd): 包含关节力矩命令的ROS消息
        """
        self.joint_cmd = joint_cmd
        
    def update_sensor_data(self, ang_vel_b, lin_acc_b, quat_w, joint_pos, joint_vel, applied_torque, joint_acc, sim_time):
        """
        lin_vel_b = scene["imu_base"].data.lin_vel_b.tolist()  # 线速度
        ang_vel_b = scene["imu_base"].data.ang_vel_b.tolist()  # 角速度
        lin_acc_b = scene["imu_base"].data.lin_acc_b.tolist()  # 线加速度
        ang_acc_b = scene["imu_base"].data.ang_acc_b.tolist()  # 角加速度
        """
        # IMU数据组合
        sensor_data = sensorsData()

        # 使用仿真时间dt更新状态时间
        current_time = rospy.Time.from_sec(float(sim_time))
        sensor_data.header.stamp = current_time
        sensor_data.header.frame_id = "world"  # 设置适当的frame_id
        sensor_data.sensor_time = current_time
        print("current_time: ", current_time)

        # IMU数据
        sensor_data.imu_data.gyro.x = ang_vel_b[0]   # ang_vel
        sensor_data.imu_data.gyro.y = ang_vel_b[1]  # ang_vel
        sensor_data.imu_data.gyro.z = ang_vel_b[2]  # ang_vel
        sensor_data.imu_data.acc.x = lin_acc_b[0]  # lin_acc
        sensor_data.imu_data.acc.y = lin_acc_b[1]  # lin_acc
        # sensor_data.imu_data.acc.z = lin_acc_b[2]  # lin_acc
        sensor_data.imu_data.acc.z = lin_acc_b[2] + 9.81  # lin_acc

        # sensor_data.imu_data.free_acc.x = lin_acc_b[0]  # lin_acc
        # sensor_data.imu_data.free_acc.y = lin_acc_b[1]  # lin_acc
        # sensor_data.imu_data.free_acc.z = lin_acc_b[2]  # lin_acc

        sensor_data.imu_data.quat.w = quat_w[0]  # 旋转矩阵
        sensor_data.imu_data.quat.x = quat_w[1]  # 旋转矩阵
        sensor_data.imu_data.quat.y = quat_w[2]  # 旋转矩阵
        sensor_data.imu_data.quat.z = quat_w[3]  # 旋转矩阵

        # 关节数据赋值优化
        # 初始化数组
        sensor_data.joint_data.joint_q = [0.0] * 28
        sensor_data.joint_data.joint_v = [0.0] * 28
        sensor_data.joint_data.joint_vd = [0.0] * 28
        sensor_data.joint_data.joint_current = [0.0] * 28

        # 腿部
        for i in range(len(self._leg_idx)//2):
            sensor_data.joint_data.joint_q[i] = joint_pos[self._leg_idx[2*i]]       
            sensor_data.joint_data.joint_q[i+6] = joint_pos[self._leg_idx[2*i+1]]   

            sensor_data.joint_data.joint_v[i] = joint_vel[self._leg_idx[2*i]]
            sensor_data.joint_data.joint_v[i+6] = joint_vel[self._leg_idx[2*i+1]]

            sensor_data.joint_data.joint_current[i] = applied_torque[self._leg_idx[2*i]]
            sensor_data.joint_data.joint_current[i+6] = applied_torque[self._leg_idx[2*i+1]]

            sensor_data.joint_data.joint_vd[i] = joint_acc[self._leg_idx[2*i]]
            sensor_data.joint_data.joint_vd[i+6] = joint_acc[self._leg_idx[2*i+1]]

        # 手部
        for i in range(len(self._arm_idx)//2):
            sensor_data.joint_data.joint_q[12+i] = joint_pos[self._arm_idx[2*i]]
            sensor_data.joint_data.joint_q[19+i] = joint_pos[self._arm_idx[2*i+1]]

            sensor_data.joint_data.joint_v[12+i] = joint_vel[self._arm_idx[2*i]]
            sensor_data.joint_data.joint_v[19+i] = joint_vel[self._arm_idx[2*i+1]]
            
            sensor_data.joint_data.joint_current[12+i] = applied_torque[self._arm_idx[2*i]]
            sensor_data.joint_data.joint_current[19+i] = applied_torque[self._arm_idx[2*i+1]]

            sensor_data.joint_data.joint_vd[12+i] = joint_acc[self._arm_idx[2*i]]
            sensor_data.joint_data.joint_vd[19+i] = joint_acc[self._arm_idx[2*i+1]]

        # 头部
        sensor_data.joint_data.joint_q[26] = joint_pos[self._head_idx[0]]
        sensor_data.joint_data.joint_q[27] = joint_pos[self._head_idx[1]]

        sensor_data.joint_data.joint_v[26] = joint_vel[self._head_idx[0]]
        sensor_data.joint_data.joint_v[27] = joint_vel[self._head_idx[1]]

        sensor_data.joint_data.joint_vd[26] = joint_acc[self._head_idx[0]]
        sensor_data.joint_data.joint_vd[27] = joint_acc[self._head_idx[1]]

        sensor_data.joint_data.joint_current[26] = applied_torque[self._head_idx[0]]
        sensor_data.joint_data.joint_current[27] = applied_torque[self._head_idx[1]]
        
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
                static_friction=0.5,
                dynamic_friction=0.5,
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

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, kuavo_robot: KuavoRobotController):
    """Run the simulator."""
    global FIRST_TIME_FLAG
    # 设置scene引用
    kuavo_robot.scene = scene
    
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    print("sim_dt: ", sim_dt)
    sim_time = 0.0
    count = 0

    body_names = scene["robot"].data.body_names   # 包含所有fix固定的joint
    joint_names = scene["robot"].data.joint_names # 只包含可活动的joint
    default_mass = scene["robot"].data.default_mass.tolist()[0] # 检查mass质量
    total_mass = 0.0
    for i in range(len(scene["robot"].data.body_names)):
        total_mass += default_mass[i]

    kuavo_robot.setup_joint_indices(joint_names)
    print("joint_names: ", joint_names) 
    print("body_names: ", body_names)
    print("total_mass: ", total_mass)

    # 设置机器人初始状态
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
    print("[INFO]: Setting initial robot state...")
        
    # Simulate physics
    while simulation_app.is_running():
        if not kuavo_robot.sim_running:
            rospy.sleep(1)
            rospy.loginfo("Waiting for simulation start signal...")
            continue

        # # Reset -- 重置
        # if count % 500 == 0:
        #     # reset counter
        #     count = 0
        #     # reset the scene entities
        #     # root state
        #     # we offset the root state by the origin since the states are written in simulation world frame
        #     # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
        #     root_state = scene["robot"].data.default_root_state.clone()
        #     root_state[:, :3] += scene.env_origins
        #     scene["robot"].write_root_pose_to_sim(root_state[:, :7])
        #     scene["robot"].write_root_velocity_to_sim(root_state[:, 7:])
        #     # set joint positions with some noise
        #     joint_pos, joint_vel = (
        #         scene["robot"].data.default_joint_pos.clone(),
        #         scene["robot"].data.default_joint_vel.clone(),
        #     )
        #     joint_pos += torch.rand_like(joint_pos) * 0.1
        #     scene["robot"].write_joint_state_to_sim(joint_pos, joint_vel)
        #     # clear internal buffers
        #     scene.reset()
        #     print("[INFO]: Resetting robot state...")

        # 更新传感器数据
        if not FIRST_TIME_FLAG:
            # 获取IMU数据
            lin_vel_b = scene["imu_base"].data.lin_vel_b.tolist()[0]  # 线速度
            ang_vel_b = scene["imu_base"].data.ang_vel_b.tolist()[0]  # 角速度
            lin_acc_b = scene["imu_base"].data.lin_acc_b.tolist()[0]  # 线加速度
            ang_acc_b = scene["imu_base"].data.ang_acc_b.tolist()[0]  # 角加速度
            quat_w = scene["imu_base"].data.quat_w.tolist()[0]  # 旋转矩阵

            # 获取机器人的数据/关节层
            joint_pos = scene["robot"].data.joint_pos.tolist()[0]
            joint_vel = scene["robot"].data.joint_vel.tolist()[0]
            joint_acc = scene["robot"].data.joint_acc.tolist()[0]
            applied_torque = scene["robot"].data.applied_torque.tolist()[0]
            
            # 获取机器人的数据/质心层
            body_state_w = scene["robot"].data.body_state_w.tolist()[0]
            body_acc_w = scene["robot"].data.body_acc_w.tolist()[0]

            # 提取位置
            base_link_state_w = body_state_w[1][0:3] # [x, y, z]
            # 提取姿态
            base_link_quat_w = body_state_w[1][3:7]  # [w, x, y, z]
            # 提取线速度
            base_link_lin_vel = body_state_w[1][7:10] # [vx, vy, vz]
            # 提取角速度
            base_link_ang_vel = body_state_w[1][10:13] # [wx, wy, wz]
            
            # 提取加速度
            base_link_lin_acc = body_acc_w[1][0:3] # [vax, vay, vaz]
            base_link_ang_acc = body_acc_w[1][3:6] # [wax, way, waz]

            # target 
            # target_ang_vel = ang_vel_b
            # target_lin_acc = lin_acc_b
            # target_quat_w = quat_w
            target_ang_vel = base_link_ang_vel
            target_lin_acc = base_link_lin_acc
            target_quat_w = base_link_quat_w

            kuavo_robot.update_sensor_data(target_ang_vel, target_lin_acc, target_quat_w, 
                                         joint_pos, joint_vel, applied_torque, joint_acc, sim_time)

            joint_cmd = kuavo_robot.joint_cmd
            if joint_cmd is None:
                continue
            else:   
                # 创建一个与机器人总关节数相同的零力矩数组
                full_torque_cmd = [0.0] * len(scene["robot"].data.joint_names)
                # # 将收到的力矩命令映射到对应的关节索引上
                # for i in range((len(kuavo_robot._leg_idx)//2)): # type: ignore
                #     full_torque_cmd[kuavo_robot._leg_idx[2*i]] = joint_cmd.tau[i]  # type: ignore # 左腿
                #     full_torque_cmd[kuavo_robot._leg_idx[2*i+1]] = joint_cmd.tau[i+6]  # type: ignore # 右腿

                # # 手部                    
                # for i in range((len(kuavo_robot._arm_idx)//2)): # type: ignore
                #     full_torque_cmd[kuavo_robot._arm_idx[2*i]] = joint_cmd.tau[12+i]  # type: ignore # 左臂
                #     full_torque_cmd[kuavo_robot._arm_idx[2*i+1]] = joint_cmd.tau[19+i]  # type: ignore # 右臂

                # 左腿
                full_torque_cmd[kuavo_robot._leg_idx[0]] = joint_cmd.tau[0]  # type: ignore # 左腿
                full_torque_cmd[kuavo_robot._leg_idx[2]] = joint_cmd.tau[1]  # type: ignore # 左腿
                full_torque_cmd[kuavo_robot._leg_idx[4]] = joint_cmd.tau[2]  # type: ignore # 左腿
                full_torque_cmd[kuavo_robot._leg_idx[6]] = joint_cmd.tau[3]  # type: ignore # 左腿
                full_torque_cmd[kuavo_robot._leg_idx[8]] = joint_cmd.tau[4]  # type: ignore # 左腿
                full_torque_cmd[kuavo_robot._leg_idx[10]] = joint_cmd.tau[5]  # type: ignore # 左腿

                # 右腿
                full_torque_cmd[kuavo_robot._leg_idx[1]] = joint_cmd.tau[6]  # type: ignore # 右腿
                full_torque_cmd[kuavo_robot._leg_idx[3]] = joint_cmd.tau[7]  # type: ignore # 右腿
                full_torque_cmd[kuavo_robot._leg_idx[5]] = joint_cmd.tau[8]  # type: ignore # 右腿
                full_torque_cmd[kuavo_robot._leg_idx[7]] = joint_cmd.tau[9]  # type: ignore # 右腿
                full_torque_cmd[kuavo_robot._leg_idx[9]] = joint_cmd.tau[10]  # type: ignore # 右腿
                full_torque_cmd[kuavo_robot._leg_idx[11]] = joint_cmd.tau[11]  # type: ignore # 右腿

                # 左手     
                full_torque_cmd[kuavo_robot._arm_idx[0]] = joint_cmd.tau[12]  # type: ignore # 左臂
                full_torque_cmd[kuavo_robot._arm_idx[2]] = joint_cmd.tau[13]  # type: ignore # 左臂
                full_torque_cmd[kuavo_robot._arm_idx[4]] = joint_cmd.tau[14]  # type: ignore # 左臂
                full_torque_cmd[kuavo_robot._arm_idx[6]] = joint_cmd.tau[15]  # type: ignore # 左臂
                full_torque_cmd[kuavo_robot._arm_idx[8]] = joint_cmd.tau[16]  # type: ignore # 左臂
                full_torque_cmd[kuavo_robot._arm_idx[10]] = joint_cmd.tau[17]  # type: ignore # 左臂    
                full_torque_cmd[kuavo_robot._arm_idx[12]] = joint_cmd.tau[18]  # type: ignore # 左臂
                
                # 右手
                full_torque_cmd[kuavo_robot._arm_idx[1]] = joint_cmd.tau[19]  # type: ignore # 右臂
                full_torque_cmd[kuavo_robot._arm_idx[3]] = joint_cmd.tau[20]  # type: ignore # 右臂
                full_torque_cmd[kuavo_robot._arm_idx[5]] = joint_cmd.tau[21]  # type: ignore # 右臂
                full_torque_cmd[kuavo_robot._arm_idx[7]] = joint_cmd.tau[22]  # type: ignore # 右臂
                full_torque_cmd[kuavo_robot._arm_idx[9]] = joint_cmd.tau[23]  # type: ignore # 右臂
                full_torque_cmd[kuavo_robot._arm_idx[11]] = joint_cmd.tau[24]  # type: ignore # 右臂
                full_torque_cmd[kuavo_robot._arm_idx[13]] = joint_cmd.tau[25]  # type: ignore # 右臂
                
                # 头部
                full_torque_cmd[kuavo_robot._head_idx[0]] = joint_cmd.tau[26]  # type: ignore # head_l1_joint
                full_torque_cmd[kuavo_robot._head_idx[1]] = joint_cmd.tau[27]  # type: ignore # head_r1_joint

                # 将力矩命令转换为tensor并发送给机器人
                torque_tensor = torch.tensor([full_torque_cmd], device=scene["robot"].device)
                scene["robot"].set_joint_effort_target(torque_tensor)

        # write data to sim
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        scene.update(sim_dt)
        # 第一帧结束
        FIRST_TIME_FLAG = False
        # 打印时间
        # print("sim_time: ", sim_time)


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.002, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    
    # Set main camera
    sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])
    
    # Create scene
    scene_cfg = BipedSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    # 创建机器人控制实例
    kuavo_robot = KuavoRobotController()

    # Reset simulation
    sim.reset()
    
    # Now we are ready!
    print("[INFO]: Setup complete...")
    
    # Run the simulator
    run_simulator(sim, scene, kuavo_robot)

if __name__ == "__main__":
    main()
    simulation_app.close()