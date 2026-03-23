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

from legged_lab.tasks.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg
from legged_lab.assets.unitree import UNITREE_G1_29DOF_CFG
import legged_lab.tasks.locomotion.velocity.mdp as mdp


@configclass
class G1RobustEnvCfg(G1FlatEnvCfg):
    """G1 robot configuration for robust locomotion with remote control and external force resistance."""

    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        self.scene.num_envs = 2048
        self.episode_length_s = 20

        self.terminations.root_height_below_minimum.params["minimum_height"] = 0.35

        self.events.base_external_force_torque.mode = "interval"
        self.events.base_external_force_torque.interval_range_s = (5.0, 10.0)
        self.events.base_external_force_torque.params["force_range"] = (0.0, 50.0)
        self.events.base_external_force_torque.params["torque_range"] = (0.0, 20.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "torso_link"

        self.events.push_robot.mode = "interval"
        self.events.push_robot.interval_range_s = (5.0, 10.0)
        self.events.push_robot.params["velocity_range"] = {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}


@configclass
class G1RobustEnvCfg_PLAY(G1RobustEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 10
        self.mode = "play"
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class G1RobustEnvCfg_DEBUG(G1RobustEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.scene.env_spacing = 3.0
