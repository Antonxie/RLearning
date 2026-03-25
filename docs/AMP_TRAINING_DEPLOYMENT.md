# G1 AMP+PPO 训练部署指南

## 目录
1. [环境架构概述](#1-环境架构概述)
2. [正确的启动方式](#2-正确的启动方式)
3. [AMP+GMR+URDF 关节命名映射规则](#3-ampgmrurdf-关节命名映射规则)
4. [关键配置文件清单](#4-关键配置文件清单)
5. [常见问题排查](#5-常见问题排查)

---

## 1. 环境架构概述

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Isaac Lab + Isaac Sim 架构                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  IsaacLab (学习框架)                                                         │
│       │                                                                    │
│       ├── source/isaaclab/          # IsaacLab 核心模块                      │
│       ├── source/isaaclab_tasks/    # 任务定义                              │
│       ├── source/legged_lab/        # 你的机器人任务 (AMP训练)               │
│       │                                                                          │
│       └── _isaac_sim/               # Isaac Sim 仿真器                        │
│                ├── kit/python/bin/python3  # Python 3.11 解释器              │
│                └── kit/python/lib/python3.11/site-packages/  # isaacsim    │
│                                                                             │
│  关键：必须使用 _isaac_sim/python.sh 来启动训练，而不是 conda 的 python!        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 正确的启动方式

### 2.1 启动命令（已验证可用）

```bash
# 方式：使用 _isaac_sim/python.sh 作为解释器
cd /home/robot/StarBot/isaac-projects/legged_lab-main

# 如果使用物理显示器1
export DISPLAY=:1

/home/robot/StarBot/isaac-projects/IsaacLab/IsaacLab-main/_isaac_sim/python.sh \
    scripts/rsl_rl/train.py \
    --task LeggedLab-Isaac-AMP-G1-v0 \
    --num_envs 2000 \
    --device cuda:0
```

### 2.2 后台运行方式

```bash
cd /home/robot/StarBot/isaac-projects/legged_lab-main
export DISPLAY=:1

nohup /home/robot/StarBot/isaac-projects/IsaacLab/IsaacLab-main/_isaac_sim/python.sh \
    scripts/rsl_rl/train.py \
    --task LeggedLab-Isaac-AMP-G1-v0 \
    --num_envs 2000 \
    > /tmp/amp_train.log 2>&1 &

# 查看日志
tail -f /tmp/amp_train.log
```

### 2.3 错误的启动方式（避免！）

```bash
# ❌ 错误1: 使用 conda 的 python
conda activate isaaclab
python scripts/rsl_rl/train.py  # 会缺少 isaacsim 模块

# ❌ 错误2: 使用 isaaclab.sh -p
source isaaclab.sh -p
python scripts/rsl_rl/train.py  # 会使用 conda_isaac 的 Python 3.8

# ❌ 错误3: 直接使用 pip 安装的 python
/usr/bin/python3 scripts/rsl_rl/train.py  # 缺少所有 Isaac 相关模块

# ❌ 错误4: --headless=False 参数格式错误
python.sh train.py --headless=False  # 会报错 "ignored explicit argument"
# 正确做法是不加 --headless 参数，或使用 --headless
```

---

## 3. AMP+GMR+URDF 关节命名映射规则

### 3.1 三个系统概述

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          三个系统的关节命名                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. GMR (Gaussian Mixture Regression) - 人类运动数据                         │
│     来源: scripts/tools/retarget/config/g1_29dof.yaml 中的 gmr_dof_names     │
│     用途: 记录人类行走时的关节角度数据（29个DOF，顺序与机器人无关）            │
│     ★ 这是基准！所有映射都以此为基准                                         │
│                                                                             │
│  2. URDF/USD (Robot Model) - 机器人模型实际关节                             │
│     来源: legged_lab/assets/unitree.py 中的 joint_sdk_names                  │
│     用途: 定义 G1 机器人的实际关节顺序（29个DOF）                            │
│                                                                             │
│  3. AMP Discriminator - 对抗模仿学习                                         │
│     来源: g1_amp_env_cfg.py 中的 observation groups                          │
│     用途: 让机器人学习模仿人类运动风格                                        │
│                                                                             │
│  ★ 核心映射规则:                                                            │
│     gmr_dof_names[i] → lab_dof_names[i]                                    │
│     即：gmr_dof_names 和 lab_dof_names 必须有完全相同的顺序！               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 ★★★ 关键发现：以 GMR 为基准

**retarget 配置的映射逻辑是基于索引位置的：**
```python
# motion_data[joint_index] 对应 gmr_dof_names[joint_index]
# 然后映射到 lab_dof_names[joint_index]
# 最后应用到 USD robot 的 lab_dof_names[joint_index]
```

因此：
- **`gmr_dof_names` 的顺序** = 人类运动数据的关节顺序（基准）
- **`lab_dof_names` 的顺序** = 必须与 `gmr_dof_names` **完全一致**
- **`unitree.py joint_sdk_names`** = USD 模型的真实关节顺序（可能与上述不同）

### 3.3 gmr_dof_names / lab_dof_names 正确顺序（2026-03-24 修正）

| 索引 | 关节名称 | 说明 | 左/右 | 部位 |
|------|----------|------|--------|------|
| 0 | left_hip_pitch_joint | 左髋屈伸 | 左 | 腿 |
| 1 | left_hip_roll_joint | 左髋外展/内收 | 左 | 腿 |
| 2 | left_hip_yaw_joint | 左髋旋转 | 左 | 腿 |
| 3 | left_knee_joint | 左膝 | 左 | 腿 |
| 4 | left_ankle_pitch_joint | 左踝屈伸 | 左 | 腿 |
| 5 | left_ankle_roll_joint | 左踝侧翻 | 左 | 腿 |
| 6 | right_hip_pitch_joint | 右髋屈伸 | 右 | 腿 |
| 7 | right_hip_roll_joint | 右髋外展/内收 | 右 | 腿 |
| 8 | right_hip_yaw_joint | 右髋旋转 | 右 | 腿 |
| 9 | right_knee_joint | 右膝 | 右 | 腿 |
| 10 | right_ankle_pitch_joint | 右踝屈伸 | 右 | 腿 |
| 11 | right_ankle_roll_joint | 右踝侧翻 | 右 | 腿 |
| 12 | waist_yaw_joint | 腰部偏航 | 中 | 腰 |
| 13 | waist_roll_joint | 腰部侧翻 | 中 | 腰 |
| 14 | waist_pitch_joint | 腰部前屈 | 中 | 腰 |
| 15 | left_shoulder_pitch_joint | 左肩屈伸 | 左 | 臂 |
| 16 | left_shoulder_roll_joint | 左肩外展/内收 | 左 | 臂 |
| 17 | left_shoulder_yaw_joint | 左肩旋转 | 左 | 臂 |
| 18 | left_elbow_joint | 左肘 | 左 | 臂 |
| 19 | left_wrist_roll_joint | 左腕滚动 | 左 | 臂 |
| 20 | left_wrist_pitch_joint | 左腕屈伸 | 左 | 臂 |
| 21 | left_wrist_yaw_joint | 左腕偏航 | 左 | 臂 |
| 22 | right_shoulder_pitch_joint | 右肩屈伸 | 右 | 臂 |
| 23 | right_shoulder_roll_joint | 右肩外展/内收 | 右 | 臂 |
| 24 | right_shoulder_yaw_joint | 右肩旋转 | 右 | 臂 |
| 25 | right_elbow_joint | 右肘 | 右 | 臂 |
| 26 | right_wrist_roll_joint | 右腕滚动 | 右 | 臂 |
| 27 | right_wrist_pitch_joint | 右腕屈伸 | 右 | 臂 |
| 28 | right_wrist_yaw_joint | 右腕偏航 | 右 | 臂 |

**顺序规律**: 先左腿(6) → 右腿(6) → 腰部(3) → 左臂(7) → 右臂(7)

### 3.4 关键肢体名称（Key Body Names）

用于 AMP discriminator 观察的 6 个关键肢体位置（已验证存在于 USD 模型）：

| 索引 | Key Body Name (Link) | 说明 |
|------|---------------------|------|
| 0 | left_ankle_roll_link | 左脚踝 |
| 1 | right_ankle_roll_link | 右脚踝 |
| 2 | left_wrist_yaw_link | 左手腕（手臂摆动） |
| 3 | right_wrist_yaw_link | 右手腕（手臂摆动） |
| 4 | left_shoulder_roll_link | 左肩（手臂摆动） |
| 5 | right_shoulder_roll_link | 右肩（手臂摆动） |

**注意**: 这些是 **link** 名称，不是 joint 名称，名称必须与 USD/URDF 模型中的实际 link 名称匹配！

### 3.5 Symmetry（左右镜像）索引

用于数据增强的左右镜像索引（基于修正后的 gmr_dof_names/lab_dof_names 顺序）：

```python
# Lab joint indices (matching gmr_dof_names / lab_dof_names order)
# 索引: 先左腿(0-5) → 右腿(6-11) → 腰部(12-14) → 左臂(15-21) → 右臂(22-28)

left_indices = [0, 1, 2, 3, 4, 5, 15, 16, 17, 18, 19, 20, 21]
# 对应: left_hip_pitch, left_hip_roll, left_hip_yaw, left_knee,
#       left_ankle_pitch, left_ankle_roll,
#       left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw,
#       left_elbow, left_wrist_roll, left_wrist_pitch, left_wrist_yaw

right_indices = [6, 7, 8, 9, 10, 11, 22, 23, 24, 25, 26, 27, 28]
# 对应: right_hip_pitch, right_hip_roll, right_hip_yaw, right_knee,
#       right_ankle_pitch, right_ankle_roll,
#       right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw,
#       right_elbow, right_wrist_roll, right_wrist_pitch, right_wrist_yaw

# Roll joints (需要符号翻转)
roll_indices = [1, 4, 5, 7, 10, 11, 16, 19, 23, 26]
# 对应: left_hip_roll, left_ankle_pitch, left_ankle_roll,
#       right_hip_roll, right_ankle_pitch, right_ankle_roll,
#       left_shoulder_roll, left_wrist_roll,
#       right_shoulder_roll, right_wrist_roll

# Yaw joints (需要符号翻转)
yaw_indices = [2, 8, 17, 21, 24, 28]
# 对应: left_hip_yaw, right_hip_yaw,
#       left_shoulder_yaw, left_wrist_yaw,
#       right_shoulder_yaw, right_wrist_yaw

# Waist joints (不参与镜像，保持不变)
waist_indices = [12, 13, 14]
# 对应: waist_yaw, waist_roll, waist_pitch
```

### 3.6 常见错误及后果

| 错误类型 | 后果 | 示例 |
|---------|------|------|
| lab_dof_names 与 USD joint 顺序不一致 | disc_loss = 0，机器人无法学习人类运动风格 | 把 right_hip_pitch 放在了 index 1 |
| GMR 与 lab_dof_names 映射错误 | 模仿动作异常，如手臂乱动 | 把 shoulder joint 映射到 ankle joint |
| Key Body 名称错误 | AMP 奖励无法计算 | 使用不存在的 link 名称 |
| Symmetry 索引错误 | 数据增强效果相反，机器人运动不自然 | 左右镜像时膝盖对应错误 |
| Roll/Yaw 索引遗漏 | 左右镜像时方向错误，机器人"顺拐" | 行走时左右手同边甩 |

---

## 4. 关键配置文件清单

### 4.1 必须保持一致的文件

| 文件路径 | 用途 | 关键内容 |
|---------|------|---------|
| `scripts/tools/retarget/config/g1_29dof.yaml` | GMR→URDF 映射 | gmr_dof_names (人类数据), lab_dof_names (机器人, 必须与USD一致) |
| `source/legged_lab/tasks/locomotion/amp/config/g1/g1_amp_env_cfg.py` | AMP 环境配置 | KEY_BODY_NAMES, observation groups |
| `source/legged_lab/tasks/locomotion/amp/mdp/symmetry/g1.py` | 左右镜像配置 | left_indices, right_indices, roll_indices, yaw_indices |
| `source/legged_lab/assets/unitree.py` | 机器人模型 | joint_sdk_names (29 DOF, 必须与USD一致) |

### 4.2 配置一致性检查表

在修改任何关节相关配置前，请逐项确认：

```
□ lab_dof_names[0-28] 顺序是否与 unitree.py joint_sdk_names[0-28] 完全一致？
□ symmetry/g1.py 中的索引是否与上述 USD joint 顺序一致？
□ KEY_BODY_NAMES 中的 link 名称是否在 USD 模型中存在？
□ gmr_dof_names → lab_dof_names 的映射是否正确（retargeting）？
□ 任何新增加的关节是否同时更新了所有相关位置？
```

### 4.3 验证脚本

```python
# 文件位置: /home/robot/StarBot/isaac-projects/legged_lab-main/verify_joint_order.py

# USD 模型的关节顺序（从 print_g1_links.py 获得）
USD_JOINT_ORDER = [
    "left_hip_pitch_joint",      # 0
    "right_hip_pitch_joint",     # 1
    "waist_yaw_joint",           # 2
    "left_hip_roll_joint",       # 3
    "right_hip_roll_joint",      # 4
    "waist_roll_joint",          # 5
    "left_hip_yaw_joint",        # 6
    "right_hip_yaw_joint",       # 7
    "waist_pitch_joint",         # 8
    "left_knee_joint",           # 9
    "right_knee_joint",          # 10
    "left_shoulder_pitch_joint", # 11
    "right_shoulder_pitch_joint",# 12
    "left_ankle_pitch_joint",    # 13
    "right_ankle_pitch_joint",   # 14
    "left_shoulder_roll_joint",  # 15
    "right_shoulder_roll_joint", # 16
    "left_ankle_roll_joint",     # 17
    "right_ankle_roll_joint",    # 18
    "left_shoulder_yaw_joint",  # 19
    "right_shoulder_yaw_joint",  # 20
    "left_elbow_joint",          # 21
    "right_elbow_joint",         # 22
    "left_wrist_roll_joint",     # 23
    "right_wrist_roll_joint",    # 24
    "left_wrist_pitch_joint",    # 25
    "right_wrist_pitch_joint",   # 26
    "left_wrist_yaw_joint",      # 27
    "right_wrist_yaw_joint",     # 28
]

# 运行验证
# /home/robot/conda_isaac/bin/python verify_joint_order.py
```

---

## 5. 常见问题排查

### 5.1 ModuleNotFoundError: No module named 'isaacsim'

**原因**: 使用了错误的 Python 解释器

**解决**:
```bash
# ❌ 错误
python scripts/rsl_rl/train.py
conda activate isaaclab && python scripts/rsl_rl/train.py

# ✅ 正确
/home/robot/StarBot/isaac-projects/IsaacLab/IsaacLab-main/_isaac_sim/python.sh scripts/rsl_rl/train.py
```

### 5.2 disc_loss = 0.0

**可能原因**:
1. expert 数据未正确加载（检查 MotionData 链接）
2. disc_demo observation group 配置错误
3. **lab_dof_names 与 USD joint 顺序不一致**（最常见！）

**排查步骤**:
```bash
# 1. 检查 MotionData 是否存在
ls -la /home/robot/StarBot/isaac-projects/legged_lab-main/source/legged_lab/data/MotionData/

# 2. 运行验证脚本检查关节顺序
/home/robot/conda_isaac/bin/python verify_joint_order.py

# 3. 检查 unitree.py 的 joint_sdk_names 是否与 USD 一致
grep -A 30 "joint_sdk_names" source/legged_lab/assets/unitree.py
```

### 5.3 机器人手臂乱动

**原因**: GMR 数据被错误映射到不对应的机器人关节

**解决**: 按照本文档第3节的表格修正 `g1_29dof.yaml` 中的 `lab_dof_names`

### 5.4 训练启动后立即崩溃 (segmentation fault)

**原因**: DISPLAY 环境变量设置问题

**解决**:
```bash
# 使用显示器1时
export DISPLAY=:1

# 或不使用 DISPLAY（headless 模式）
unset DISPLAY
```

### 5.5 "ignored explicit argument 'False'" 错误

**原因**: --headless 参数格式错误

**解决**:
```bash
# ❌ 错误
python.sh train.py --headless=False  # 会报错

# ✅ 正确（不加 --headless 参数，或只写 --headless）
python.sh train.py  # 默认非 headless
python.sh train.py --headless  # headless 模式
```

---

## 6. 快速启动命令汇总

```bash
# 完整的一键启动命令（带物理显示器）
cd /home/robot/StarBot/isaac-projects/legged_lab-main && \
export DISPLAY=:1 && \
nohup /home/robot/StarBot/isaac-projects/IsaacLab/IsaacLab-main/_isaac_sim/python.sh \
    scripts/rsl_rl/train.py \
    --task LeggedLab-Isaac-AMP-G1-v0 \
    --num_envs 2000 \
    > /tmp/amp_train.log 2>&1 &

# 查看训练进度
tail -f /tmp/amp_train.log
```

---

## 7. 调试工具

### 7.1 打印 G1 实际关节和 Link 名称

```bash
# 位置: /home/robot/StarBot/isaac-projects/legged_lab-main/print_g1_links.py
# 用途: 打印实际 USD 模型中的 joint_names 和 body_names

cd /home/robot/StarBot/isaac-projects/legged_lab-main && \
export DISPLAY=:1 && \
/home/robot/StarBot/isaac-projects/IsaacLab/IsaacLab-main/_isaac_sim/python.sh print_g1_links.py
```

### 7.2 验证关节顺序一致性

```bash
# 位置: /home/robot/StarBot/isaac-projects/legged_lab-main/verify_joint_order.py
/home/robot/conda_isaac/bin/python verify_joint_order.py
```

---

## 8. 文档维护记录

| 日期 | 修改内容 | 修改人 |
|------|---------|--------|
| 2026-03-23 | 初始创建，修复 g1_29dof.yaml 和 symmetry/g1.py 映射错误 | AI |
| 2026-03-23 | **重大修正**: 发现 USD 模型关节顺序是左右交替而非左腿集中，更新所有相关配置和文档 | AI |
| 2026-03-24 | **再次修正**: 确认 lab_dof_names 必须与 gmr_dof_names 完全一致（以 GMR 为基准），更新对称索引 | AI |

---

## 9. 当前训练状态 (2026-03-24)

### 最新训练指标（Iteration 1391/50000）

| 指标 | 当前值 | 状态 |
|------|--------|------|
| disc_loss | 0.0154 | ✅ 正常（非零） |
| expert_accuracy | 100% | ✅ 正常 |
| policy_accuracy | 49.7% | 📈 学习中 |
| Mean reward | 7.24 | 📈 持续上升 |
| action std | 0.34 | 📉 收敛中 |
| bad_orientation | 78.1% | ⚠️ 仍需改善 |

**观察**:
- disc_loss 已正常（不再是0）
- 机器人的 bad_orientation 终止率从 99.7% 下降到 78.1%，说明平衡能力在改善
- 训练预计还需约 2 小时完成 5 万次迭代

---

## 附录：核心问题发现记录

### 问题描述
disc_loss = 0，机器人手臂乱动，无法学习人类行走风格

### 根本原因
通过多次调试发现 retarget 映射的核心规则：

1. **gmr_dof_names 与 lab_dof_names 必须有完全相同的顺序**
2. 这是因为 retarget 是基于索引位置映射的：motion_data[i] → lab_dof_names[i]
3. **unitree.py joint_sdk_names** 是 USD 模型的真实关节顺序，但 retarget 过程不需要与其一致

### 修复措施
1. 修正 `g1_29dof.yaml` 的 `gmr_dof_names` 和 `lab_dof_names` 为同一顺序（以 GMR 为基准）
2. 修正 `symmetry/g1.py` 的 `left_indices`, `right_indices`, `roll_indices`, `yaw_indices`
3. 验证 `KEY_BODY_NAMES` 与实际 USD link 名称匹配

### 验证结果
- disc_loss: 0.7430 → 6.9023 → 0.0154（正常，趋于收敛）
- expert_accuracy: 99.5%（正常）
- bad_orientation: 99.7% → 78.1%（显著改善）
- 训练正常进行中
