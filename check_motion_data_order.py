#!/usr/bin/env python3
"""检查 motion data 的实际关节顺序"""

import joblib
import numpy as np

motion_file = "source/legged_lab/data/MotionData/g1_29dof/amp/walk_and_run/B10_-__Walk_turn_left_45_stageii.pkl"

print("Loading motion data...")
data = joblib.load(motion_file)

print("\n=== Motion Data Keys ===")
print(f"Keys: {data.keys()}")

print("\n=== Data Shapes ===")
for key, value in data.items():
    if isinstance(value, np.ndarray):
        print(f"  {key}: shape = {value.shape}")
    else:
        print(f"  {key}: {type(value)}")

print("\n=== dof_pos shape ===")
dof_pos = data['dof_pos']
print(f"dof_pos shape: {dof_pos.shape}")
print(f"Number of DOFs: {dof_pos.shape[1] if len(dof_pos.shape) > 1 else 'N/A'}")

# 加载 retarget 配置
import yaml
with open('scripts/tools/retarget/config/g1_29dof.yaml', 'r') as f:
    config = yaml.safe_load(f)

gmr_dof_names = config['gmr_dof_names']
lab_dof_names = config['lab_dof_names']

print("\n=== GMR DOF Names (29) ===")
for i, name in enumerate(gmr_dof_names):
    print(f"  [{i:2d}] {name}")

print("\n=== lab_dof_names in config (29) ===")
for i, name in enumerate(lab_dof_names):
    print(f"  [{i:2d}] {name}")

print("\n=== Analysis ===")
print(f"retarget 后的 dof_pos 形状: {dof_pos.shape}")
print(f"dof_pos[i] 对应 lab_dof_names[i]")
print(f"例如: dof_pos[:, 0] 是 {lab_dof_names[0]} 的数据")

print("\n=== 检查 motion data 中各关节的数据范围 ===")
for i in range(min(6, dof_pos.shape[1])):
    col_data = dof_pos[:, i]
    print(f"  dof_pos[:, {i}] ({lab_dof_names[i]:25s}): min={col_data.min():.4f}, max={col_data.max():.4f}, mean={col_data.mean():.4f}")
