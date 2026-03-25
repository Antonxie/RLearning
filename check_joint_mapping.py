#!/usr/bin/env python3
"""
检查 GMR、USD、AMP 三个系统的关节配对关系
"""

import yaml

# 加载 retarget 配置
with open('scripts/tools/retarget/config/g1_29dof.yaml', 'r') as f:
    config = yaml.safe_load(f)

gmr_dof_names = config['gmr_dof_names']
lab_dof_names = config['lab_dof_names']
lab_key_body_names = config['lab_key_body_names']

# USD joint_sdk_names (从 unitree.py 获取)
usd_joint_names = [
    "left_hip_pitch_joint",     # 0
    "right_hip_pitch_joint",    # 1
    "waist_yaw_joint",          # 2
    "left_hip_roll_joint",      # 3
    "right_hip_roll_joint",     # 4
    "waist_roll_joint",         # 5
    "left_hip_yaw_joint",       # 6
    "right_hip_yaw_joint",      # 7
    "waist_pitch_joint",        # 8
    "left_knee_joint",          # 9
    "right_knee_joint",         # 10
    "left_shoulder_pitch_joint",# 11
    "right_shoulder_pitch_joint",# 12
    "left_ankle_pitch_joint",   # 13
    "right_ankle_pitch_joint",  # 14
    "left_shoulder_roll_joint", # 15
    "right_shoulder_roll_joint",# 16
    "left_ankle_roll_joint",    # 17
    "right_ankle_roll_joint",   # 18
    "left_shoulder_yaw_joint",  # 19
    "right_shoulder_yaw_joint", # 20
    "left_elbow_joint",         # 21
    "right_elbow_joint",        # 22
    "left_wrist_roll_joint",    # 23
    "right_wrist_roll_joint",   # 24
    "left_wrist_pitch_joint",   # 25
    "right_wrist_pitch_joint",  # 26
    "left_wrist_yaw_joint",     # 27
    "right_wrist_yaw_joint",    # 28
]

print("=" * 100)
print("GMR、USD、AMP 关节配对关系检查报告")
print("=" * 100)

print("\n" + "=" * 50)
print("1. GMR 数据顺序 (人类运动数据 - 29个关节)")
print("=" * 50)
for i, name in enumerate(gmr_dof_names):
    print(f"  [{i:2d}] {name}")

print("\n" + "=" * 50)
print("2. lab_dof_names 顺序 (retarget配置中)")
print("=" * 50)
for i, name in enumerate(lab_dof_names):
    print(f"  [{i:2d}] {name}")

print("\n" + "=" * 50)
print("3. USD joint_sdk_names 顺序 (unitree.py)")
print("=" * 50)
for i, name in enumerate(usd_joint_names):
    print(f"  [{i:2d}] {name}")

print("\n" + "=" * 50)
print("4. 配对分析：GMR → lab_dof_names")
print("=" * 50)
print("\n  索引 | GMR关节名称                | lab_dof_names关节名称       | 匹配?")
print("  " + "-" * 85)
for i in range(len(gmr_dof_names)):
    gmr = gmr_dof_names[i]
    lab = lab_dof_names[i] if i < len(lab_dof_names) else "N/A"
    match = "✓" if gmr == lab else "✗"
    if gmr != lab:
        print(f"  [{i:2d}] | {gmr:28s} | {lab:28s} | {match} <-- 错误!")
    else:
        print(f"  [{i:2d}] | {gmr:28s} | {lab:28s} | {match}")

print("\n" + "=" * 50)
print("5. 配对分析：lab_dof_names → USD joint_sdk_names")
print("=" * 50)
print("\n  索引 | lab_dof_names关节名称        | USD关节名称                  | 匹配?")
print("  " + "-" * 85)
mismatches = []
for i in range(len(lab_dof_names)):
    lab = lab_dof_names[i]
    usd = usd_joint_names[i] if i < len(usd_joint_names) else "N/A"
    match = "✓" if lab == usd else "✗"
    if lab != usd:
        print(f"  [{i:2d}] | {lab:28s} | {usd:28s} | {match} <-- 错误!")
        mismatches.append((i, lab, usd))
    else:
        print(f"  [{i:2d}] | {lab:28s} | {usd:28s} | {match}")

print("\n" + "=" * 50)
print("6. 总结")
print("=" * 50)
if mismatches:
    print(f"\n  ❌ 发现 {len(mismatches)} 个配对错误!")
    print("\n  错误的配对:")
    for i, lab, usd in mismatches:
        print(f"    索引 {i}: lab_dof_names={lab}, USD={usd}")
    print("\n  建议修正 lab_dof_names 使其与 USD joint_sdk_names 完全一致!")
else:
    print("\n  ✓ lab_dof_names 与 USD joint_sdk_names 完全匹配!")

print("\n" + "=" * 50)
print("7. 按身体部位分组分析")
print("=" * 50)

body_parts = {
    "左腿": ["hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"],
    "右腿": ["hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"],
    "腰部": ["waist_yaw", "waist_roll", "waist_pitch"],
    "左臂": ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"],
    "右臂": ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"],
}

for part, keywords in body_parts.items():
    print(f"\n  {part}:")
    for i, name in enumerate(gmr_dof_names):
        if any(kw in name for kw in keywords):
            print(f"    GMR[{i}]: {name}")
    print()
    for i, name in enumerate(usd_joint_names):
        if any(kw in name for kw in keywords):
            print(f"    USD[{i}]: {name}")
