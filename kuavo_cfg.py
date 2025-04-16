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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
usd_dir_path = os.path.join(BASE_DIR, "usd/")
robot_usd = "biped_s45.usd"
print(" usd_path : ", usd_dir_path + robot_usd)

##
# Configuration
##

KINOVA_ROBOTIQ = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=usd_dir_path + robot_usd,
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
        pos=(0.0, 0.0, 0.8664),
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
            velocity_limit=1.2,
            effort_limit={
                "zarm_[lr][1-4]_joint": 200.0,
                "zarm_[lr][5-7]_joint": 30.0,
            },
            stiffness={
                "zarm_[lr][1]_joint": 15.0,
                "zarm_[lr][2]_joint": 15.0,
                "zarm_[lr][3]_joint": 15.0,
                "zarm_[lr][4]_joint": 15.0,
                "zarm_[lr][5]_joint": 15.0,
                "zarm_[lr][6]_joint": 15.0,
                "zarm_[lr][7]_joint": 15.0,
            },
            damping={
                "zarm_[lr][1]_joint": 3.0,
                "zarm_[lr][2]_joint": 3.0,
                "zarm_[lr][3]_joint": 3.0,
                "zarm_[lr][4]_joint": 3.0,
                "zarm_[lr][5]_joint": 3.0,
                "zarm_[lr][6]_joint": 3.0,
                "zarm_[lr][7]_joint": 3.0,
            },
        ),
        "legs": ImplicitActuatorCfg(
            joint_names_expr=["leg_[lr][1-6]_joint"],
            effort_limit=200.0,
            velocity_limit=10,
            stiffness={
                "leg_[lr][1]_joint": 60.0,
                "leg_[lr][2]_joint": 60.0,
                "leg_[lr][3]_joint": 60.0,
                "leg_[lr][4]_joint": 60.0,
                "leg_[lr][5]_joint": 30.0,
                "leg_[lr][6]_joint": 15.0,
            },
            damping={
                "leg_[lr][1]_joint": 10.0,
                "leg_[lr][2]_joint": 6.0,
                "leg_[lr][3]_joint": 12.0,
                "leg_[lr][4]_joint": 12.0,
                "leg_[lr][5]_joint": 22.0,
                "leg_[lr][6]_joint": 22.0,
            },
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["zhead_[1-2]_joint"],
            effort_limit=200.0,
            velocity_limit=10,
            stiffness=300.0,
            damping=30.0,
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