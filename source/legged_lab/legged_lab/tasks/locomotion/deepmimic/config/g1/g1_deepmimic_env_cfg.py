import os
import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import legged_lab.tasks.locomotion.deepmimic.mdp as mdp
from legged_lab.tasks.locomotion.deepmimic.deepmimic_env_cfg import DeepMimicEnvCfg, DeepMimicSceneCfg
from legged_lab import LEGGED_LAB_ROOT_DIR

##
# Pre-defined configs
##
from legged_lab.assets.unitree import UNITREE_G1_29DOF_CFG

# The order must align with the retarget config file scripts/tools/retarget/config/g1_29dof.yaml
KEY_BODY_NAMES = [
    "left_ankle_roll_link", 
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
]
ANIMATION_TERM_NAME = "animation"

@configclass
class G1DeepMimicEnvCfg(DeepMimicEnvCfg):
    
    def __post_init__(self):
        super().__post_init__()
        
        self.episode_length_s = 10.0

        self.scene.robot = UNITREE_G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.motion_data.motion_dataset.motion_data_dir = os.path.join(
            LEGGED_LAB_ROOT_DIR, "data", "MotionData", "g1_29dof", "deepmimic"
        )
        self.motion_data.motion_dataset.motion_data_weights = {
            "C4_-_run_to_walk_a_stageii": 1.0,
        }

        # -----------------------------------------------------
        # Observations
        # -----------------------------------------------------
        self.observations.policy.key_body_pos_b.params = {
            "asset_cfg": SceneEntityCfg(
                name="robot", 
                body_names=KEY_BODY_NAMES, 
                preserve_order=True
            )
        }
        self.observations.policy.ref_root_pos_error.params = {
            "animation": ANIMATION_TERM_NAME
        }
        self.observations.policy.ref_root_rot_tan_norm.params = {
            "animation": ANIMATION_TERM_NAME
        }
        self.observations.policy.ref_joint_pos.params = {
            "animation": ANIMATION_TERM_NAME
        }
        self.observations.policy.ref_key_body_pos_b.params = {
            "animation": ANIMATION_TERM_NAME
        }
        
        # -----------------------------------------------------
        # Events
        # -----------------------------------------------------
        self.events.add_base_mass.params["asset_cfg"].body_names = "torso_link"
        self.events.base_com.params["asset_cfg"].body_names = "torso_link"
        self.events.reset_from_ref.params = {
            "animation": ANIMATION_TERM_NAME,
            "height_offset": 0.1
        }
        # self.events.reset_from_ref = None
        
        # -----------------------------------------------------
        # Rewards
        # -----------------------------------------------------
        self.rewards.ref_track_root_pos_w_error_exp.weight = 0.15
        self.rewards.ref_track_root_pos_w_error_exp.params = {
            "std": 0.5,
            "animation": ANIMATION_TERM_NAME,
        }
        self.rewards.ref_track_quat_error_exp.weight = 0.08
        self.rewards.ref_track_quat_error_exp.params = {
            "std": 0.5,
            "animation": ANIMATION_TERM_NAME,
        }
        self.rewards.ref_track_root_vel_w_error_exp.weight = 0.1
        self.rewards.ref_track_root_vel_w_error_exp.params = {
            "std": 1.0,
            "animation": ANIMATION_TERM_NAME,
        }
        self.rewards.ref_track_root_ang_vel_w_error_exp.weight = 0.05
        self.rewards.ref_track_root_ang_vel_w_error_exp.params = {
            "std": 1.0,
            "animation": ANIMATION_TERM_NAME,
        }
        self.rewards.ref_track_key_body_pos_b_error_exp.weight = 0.15
        self.rewards.ref_track_key_body_pos_b_error_exp.params = {
            "std": 0.3,
            "animation": ANIMATION_TERM_NAME,
            "asset_cfg": SceneEntityCfg(
                name="robot", 
                body_names=KEY_BODY_NAMES,
                preserve_order=True
            )
        }
        self.rewards.ref_track_dof_pos_error_exp.weight = 0.5
        self.rewards.ref_track_dof_pos_error_exp.params = {
            "std": 2.0,
            "animation": ANIMATION_TERM_NAME,
        }
        self.rewards.ref_track_dof_vel_error_exp.weight = 0.1
        self.rewards.ref_track_dof_vel_error_exp.params = {
            "std": 10.0,
            "animation": ANIMATION_TERM_NAME,
        }
        
        # -----------------------------------------------------
        # Terminations
        # -----------------------------------------------------
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "waist_yaw_link", "pelvis", ".*_shoulder_.*_link", ".*_elbow_link",
        ]
        self.terminations.deviation_root_pos_w.params = {
            "threshold": 1.0,
            "animation": ANIMATION_TERM_NAME,
            "asset_cfg": SceneEntityCfg("robot"),
        }
        self.terminations.deviation_key_body_pos_w.params = {
            "threshold": 1.0,
            "animation": ANIMATION_TERM_NAME,
            "asset_cfg": SceneEntityCfg(
                name="robot", 
                body_names=KEY_BODY_NAMES, 
                preserve_order=True
            ),
        }
        
        self.terminations.bad_orientation = None
    
# For debug only
@configclass
class G1DeepMimicEnvCfg_DEBUG(G1DeepMimicEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.num_envs = 8
        self.scene.env_spacing = 3.0
        
        self.scene.robot_anim = UNITREE_G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_anim")
        self.scene.robot_anim.spawn.rigid_props.disable_gravity = True # type: ignore
        self.scene.robot_anim.spawn.articulation_props.enabled_self_collisions = False # type: ignore
        self.scene.robot_anim.spawn.activate_contact_sensors = False # type: ignore
        self.scene.robot_anim.spawn.collision_props = sim_utils.CollisionPropertiesCfg( # type: ignore
            collision_enabled=False
        ) 
        
        self.animation.animation.enable_visualization = True
        self.animation.animation.vis_root_offset = [2.0, 0.0, 0.0]
        self.animation.animation.random_initialize = False
        
        # self.terminations.bad_orientation = None
        # self.terminations.base_height = None
        # self.terminations.base_contact = None
    
    
@configclass
class G1DeepMimicEnvCfg_PLAY(G1DeepMimicEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        
        self.animation.animation.random_initialize = False


@configclass
class G1DeepMimicRobustEnvCfg(G1DeepMimicEnvCfg):
    """G1 robot DeepMimic configuration for robust locomotion with external force resistance."""

    def __post_init__(self):
        super().__post_init__()

        self.events.base_external_force_torque.mode = "interval"
        self.events.base_external_force_torque.interval_range_s = (5.0, 10.0)
        self.events.base_external_force_torque.params["force_range"] = (0.0, 30.0)
        self.events.base_external_force_torque.params["torque_range"] = (0.0, 10.0)

        self.events.push_robot.mode = "interval"
        self.events.push_robot.interval_range_s = (5.0, 10.0)
        self.events.push_robot.params["velocity_range"] = {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}


@configclass
class G1DeepMimicRobustEnvCfg_PLAY(G1DeepMimicRobustEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.animation.animation.random_initialize = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class G1DeepMimicVelocityEnvCfg(G1DeepMimicEnvCfg):
    """G1 robot DeepMimic configuration with velocity command tracking.

    This configuration combines:
    - Human-like walking style from DeepMimic motion imitation
    - Velocity command tracking for remote control
    - External force disturbance resistance
    """

    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)

        self.events.base_external_force_torque.mode = "interval"
        self.events.base_external_force_torque.interval_range_s = (5.0, 10.0)
        self.events.base_external_force_torque.params["force_range"] = (0.0, 30.0)
        self.events.base_external_force_torque.params["torque_range"] = (0.0, 10.0)

        self.events.push_robot.mode = "interval"
        self.events.push_robot.interval_range_s = (5.0, 10.0)
        self.events.push_robot.params["velocity_range"] = {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}


@configclass
class G1DeepMimicVelocityEnvCfg_PLAY(G1DeepMimicVelocityEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.animation.animation.random_initialize = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.2, 0.2)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.events.base_external_force_torque = None
        self.events.push_robot = None
