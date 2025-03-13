# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Diablo Biped-Wheeled robot."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

#from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Configuration
##

HUNTER_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/lee/Hybrid_DRL_Deployments/source/isaaclab_assets/isaaclab_assets/robots/USD_Hunter/hunter_aim4.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=0.1,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.2)
    ),

    actuators={
        "wheels_actuator": ImplicitActuatorCfg(
            joint_names_expr=["re_.*"],
            stiffness=0.0,
            damping=1000.0,
        ),
        "steer_actuator": ImplicitActuatorCfg(
            joint_names_expr=["fr_.*"],
            stiffness={
                "fr_steer_left_joint": 20.0,
                "fr_steer_right_joint": 20.0,
                
            },
            damping={
                "fr_steer_left_joint": 0.5,
                "fr_steer_right_joint": 0.5,
            },
        ),
    }
)

## For velocity control - Set stiffness=0 and for Torque control - Set both stiffness and damping = 0