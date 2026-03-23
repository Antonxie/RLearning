#!/usr/bin/env python
"""临时脚本：打印 G1 29DOF 机器人的实际 link 名称"""

import argparse
import sys

parser = argparse.ArgumentParser(description="Print G1 body names")
parser.add_argument("--headless", type=str, default="True")
args_cli = parser.parse_args(args=[])

from isaaclab.app import AppLauncher
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from legged_lab.assets.unitree import UNITREE_G1_29DOF_CFG
import isaaclab.sim as sim_utils
import isaacsim.core.utils.prims as prim_utils
from isaaclab.assets import Articulation

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01))
prim_utils.create_prim("/World/Origin", "Xform", translation=[0.0, 0.0, 0.0])
robot = Articulation(UNITREE_G1_29DOF_CFG.replace(prim_path="/World/Origin/Robot"))
sim.reset()

print("=" * 60)
print("G1 29DOF Joint Names (DOFs):")
print("=" * 60)
for i, name in enumerate(robot.joint_names):
    print(f"  [{i:2d}] {name}")

print()
print("=" * 60)
print("G1 29DOF Body Names (Links):")
print("=" * 60)
for i, name in enumerate(robot.body_names):
    print(f"  [{i:2d}] {name}")

simulation_app.close()
