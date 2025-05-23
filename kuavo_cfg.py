# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
"""
    Configuration for the Kuavo Robotics legs and arms.
"""
"""
    # isaac lab 1.0 API接口
    import omni.isaac.lab.sim as sim_utils
    from omni.isaac.lab.actuators import ImplicitActuatorCfg
    from omni.isaac.lab.assets.articulation import ArticulationCfg
    import os
"""
"""
    effort_limit - 组中关节的力/扭矩限制。
    velocity_limit - 组中关节的速度限制。
    effort_limit_sim - 组中关节应用于模拟物理解算器的努力极限。
    velocity_limit_sim - 应用于模拟物理解算器的组中关节的速度限制。
"""
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg # 隐式执行器配置
from isaaclab.assets import ArticulationCfg # 多连体配置
import os
import rospkg
import rospy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
usd_dir_path = os.path.join(BASE_DIR, "usd/")

# 添加rospack获取USD文件路径
def get_robot_usd_path(robot_version):
    rospack = rospkg.RosPack()
    try:
        kuavo_assets_path = rospack.get_path('kuavo_assets')
        usd_path = os.path.join(kuavo_assets_path, 
                               f'models/biped_s{robot_version}/urdf/biped_s{robot_version}.usd')
        if not os.path.exists(usd_path):
            rospy.logwarn(f"USD file not found at {usd_path}, falling back to default path")
            usd_path = os.path.join(usd_dir_path, f"biped_s{robot_version}.usd")
    except rospkg.ResourceNotFound:
        rospy.logwarn("kuavo_assets package not found, falling back to default path")
        usd_path = os.path.join(usd_dir_path, f"biped_s{robot_version}.usd")
    
    return usd_path

# 从ROS参数服务器获取机器人版本号
try:    
    # 获取机器人版本参数，如果未设置则默认为45
    robot_version = rospy.get_param('isaac_robot_version', 45)
    rospy.loginfo(f"Using robot version: {robot_version}")
except Exception as e:
    rospy.logwarn(f"Failed to get robot version from ROS param server: {e}")
    robot_version = 45  # 默认版本
    rospy.logwarn(f"Using default robot version: {robot_version}")

# 使用函数获取USD文件路径
robot_usd = get_robot_usd_path(robot_version)
rospy.loginfo(f"Loading USD file from: {robot_usd}")

##
# Configuration
##

# Global parameters for PD control
USE_TORQUE_CONTROL = True  # 设置为True时使用全力矩模式，False时使用PD控制

# Arm parameters
ARM_STIFFNESS = 0.0 if USE_TORQUE_CONTROL else 15.0
ARM_DAMPING = 0.0 if USE_TORQUE_CONTROL else 3.0

# Leg parameters
LEG_STIFFNESS_1_4 = 0.0 if USE_TORQUE_CONTROL else 60.0
LEG_STIFFNESS_5 = 0.0 if USE_TORQUE_CONTROL else 30.0
LEG_STIFFNESS_6 = 0.0 if USE_TORQUE_CONTROL else 15.0

LEG_DAMPING_1 = 0.0 if USE_TORQUE_CONTROL else 10.0
LEG_DAMPING_2 = 0.0 if USE_TORQUE_CONTROL else 6.0
LEG_DAMPING_3_4 = 0.0 if USE_TORQUE_CONTROL else 12.0
LEG_DAMPING_5_6 = 0.0 if USE_TORQUE_CONTROL else 22.0

# Head parameters
HEAD_STIFFNESS = 60.0 if USE_TORQUE_CONTROL else 300.0
HEAD_DAMPING = 3.0 if USE_TORQUE_CONTROL else 30.0

KINOVA_ROBOTIQ = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=robot_usd,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=0,
            fix_root_link = False
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8464),
        rot=(1.0, 0.0, 0.0, 0.0),
        lin_vel=(0.0, 0.0, 0.0),
        ang_vel=(0.0, 0.0, 0.0),
        joint_pos={
            # 腿部关节
            "leg_l1_joint": 0.01867, "leg_r1_joint": -0.01867,
            "leg_l2_joint": -0.00196, "leg_r2_joint": 0.00196,
            "leg_l3_joint": -0.43815, "leg_r3_joint": -0.43815,
            "leg_l4_joint": 0.80691, "leg_r4_joint": 0.80691,
            "leg_l5_joint": -0.31346, "leg_r5_joint": -0.31346,
            "leg_l6_joint": 0.01878, "leg_r6_joint": -0.01878,
            # 手臂关节
            "zarm_l1_joint": 0.0, "zarm_r1_joint": 0.0,
            "zarm_l2_joint": 0.0, "zarm_r2_joint": 0.0,
            "zarm_l3_joint": 0.0, "zarm_r3_joint": 0.0,
            "zarm_l4_joint": 0.0, "zarm_r4_joint": 0.0,
            "zarm_l5_joint": 0.0, "zarm_r5_joint": 0.0,
            "zarm_l6_joint": 0.0, "zarm_r6_joint": 0.0,
            "zarm_l7_joint": 0.0, "zarm_r7_joint": 0.0,
            # 头部关节
            "zhead_1_joint": 0.0, "zhead_2_joint": 0.0,
        },
    ),
    actuators={
        "arms": ImplicitActuatorCfg(
            joint_names_expr=["zarm_[lr][1-7]_joint"],
            velocity_limit=1e30,
            effort_limit={
                "zarm_[lr][1-4]_joint": 1e30,
                "zarm_[lr][5-7]_joint": 1e30,
            },
            stiffness={
                "zarm_[lr][1]_joint": ARM_STIFFNESS,
                "zarm_[lr][2]_joint": ARM_STIFFNESS,
                "zarm_[lr][3]_joint": ARM_STIFFNESS,
                "zarm_[lr][4]_joint": ARM_STIFFNESS,
                "zarm_[lr][5]_joint": ARM_STIFFNESS,
                "zarm_[lr][6]_joint": ARM_STIFFNESS,
                "zarm_[lr][7]_joint": ARM_STIFFNESS,
            },
            damping={
                "zarm_[lr][1]_joint": ARM_DAMPING,
                "zarm_[lr][2]_joint": ARM_DAMPING,
                "zarm_[lr][3]_joint": ARM_DAMPING,
                "zarm_[lr][4]_joint": ARM_DAMPING,
                "zarm_[lr][5]_joint": ARM_DAMPING,
                "zarm_[lr][6]_joint": ARM_DAMPING,
                "zarm_[lr][7]_joint": ARM_DAMPING,
            },
        ),
        "legs": ImplicitActuatorCfg(
            joint_names_expr=["leg_[lr][1-6]_joint"],
            effort_limit=1e30,
            velocity_limit=1e30,
            stiffness={
                "leg_[lr][1]_joint": LEG_STIFFNESS_1_4,
                "leg_[lr][2]_joint": LEG_STIFFNESS_1_4,
                "leg_[lr][3]_joint": LEG_STIFFNESS_1_4,
                "leg_[lr][4]_joint": LEG_STIFFNESS_1_4,
                "leg_[lr][5]_joint": LEG_STIFFNESS_5,
                "leg_[lr][6]_joint": LEG_STIFFNESS_6,
            },
            damping={
                "leg_[lr][1]_joint": LEG_DAMPING_1,
                "leg_[lr][2]_joint": LEG_DAMPING_2,
                "leg_[lr][3]_joint": LEG_DAMPING_3_4,
                "leg_[lr][4]_joint": LEG_DAMPING_3_4,
                "leg_[lr][5]_joint": LEG_DAMPING_5_6,
                "leg_[lr][6]_joint": LEG_DAMPING_5_6,
            },
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["zhead_[1-2]_joint"],
            effort_limit=1e30,
            velocity_limit=1e30,
            stiffness=HEAD_STIFFNESS,
            damping=HEAD_DAMPING,
        ),
    },
)

# """Configuration of robot with stiffer PD control."""
# KINOVA_ROBOTIQ_HPD = ArticulationCfg(
#     spawn=sim_utils.UsdFileCfg(
#         usd_path=usd_dir_path + robot_usd,
#         rigid_props=sim_utils.RigidBodyPropertiesCfg(
#             disable_gravity=False,
#             max_depenetration_velocity=5.0,
#         ),
#         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
#             enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=0,
#             fix_root_link = False
#         ),
#         activate_contact_sensors=False,
#     ),
#     init_state=ArticulationCfg.InitialStateCfg(
#         pos=(0.0, 0.0, 0.8464),
#         rot=(1.0, 0.0, 0.0, 0.0),
#         lin_vel=(0.0, 0.0, 0.0),
#         ang_vel=(0.0, 0.0, 0.0),
#         joint_pos={
#             # 腿部关节
#             "leg_l1_joint": 0.01867, "leg_r1_joint": -0.01867,
#             "leg_l2_joint": -0.00196, "leg_r2_joint": 0.00196,
#             "leg_l3_joint": -0.43815, "leg_r3_joint": -0.43815,
#             "leg_l4_joint": 0.80691, "leg_r4_joint": 0.80691,
#             "leg_l5_joint": -0.31346, "leg_r5_joint": -0.31346,
#             "leg_l6_joint": 0.01878, "leg_r6_joint": -0.01878,
#             # 手臂关节
#             "zarm_l1_joint": 0.0, "zarm_r1_joint": 0.0,
#             "zarm_l2_joint": 0.0, "zarm_r2_joint": 0.0,
#             "zarm_l3_joint": 0.0, "zarm_r3_joint": 0.0,
#             "zarm_l4_joint": 0.0, "zarm_r4_joint": 0.0,
#             "zarm_l5_joint": 0.0, "zarm_r5_joint": 0.0,
#             "zarm_l6_joint": 0.0, "zarm_r6_joint": 0.0,
#             "zarm_l7_joint": 0.0, "zarm_r7_joint": 0.0,
#             # 头部关节
#             "zhead_1_joint": 0.0, "zhead_2_joint": 0.0,
#         },
#     ),
#     actuators={
#         "arms": ImplicitActuatorCfg(
#             joint_names_expr=["zarm_[lr][1-7]_joint"],
#             velocity_limit=10.0,
#             effort_limit={
#                 "zarm_[lr][1-4]_joint": 0.0,
#                 "zarm_[lr][5-7]_joint": 0.0,
#             },
#             stiffness={
#                 "zarm_[lr][1-4]_joint": 0.0,
#                 "zarm_[lr][5-7]_joint": 0.0,
#             },
#             damping={
#                 "zarm_[lr][1-4]_joint": 0.0,
#                 "zarm_[lr][5-7]_joint": 0.0,
#             },
#         ),
#         "legs": ImplicitActuatorCfg(
#             joint_names_expr=["leg_[lr][1-6]_joint"],
#             effort_limit=200.0,
#             velocity_limit=10,
#             stiffness=0.0,  # 增加腿部刚度以提供更好的支撑
#             damping=0.0,     # 增加阻尼以提供更好的稳定性
#         ),
#         "head": ImplicitActuatorCfg(
#             joint_names_expr=["zhead_[1-2]_joint"],
#             effort_limit=200.0,
#             velocity_limit=10,
#             stiffness=0.0,
#             damping=0.0,
#         ),
#     },
# )