# G1 AMP+PPO 机器人训练项目

基于 Isaac Lab + Isaac Sim 的 G1 人形机器人 AMP（对抗运动先验）+ PPO 强化学习训练框架。

## 目录

- [项目概述](#项目概述)
- [核心功能](#核心功能)
- [环境要求](#环境要求)
- [安装步骤](#安装步骤)
- [快速启动](#快速启动)
- [代码结构](#代码结构)
- [关键配置文件](#关键配置文件)
- [训练架构](#训练架构)
- [调试工具](#调试工具)
- [常见问题](#常见问题)
- [技术细节](#技术细节)
- [维护记录](#维护记录)

---

## 项目概述

本项目实现了一个人形机器人（Unitree G1）的强化学习训练系统，结合了：

- **AMP (Adversarial Motion Priors)**: 对抗运动先验，让机器人学习人类运动风格
- **PPO (Proximal Policy Optimization)**: 近端策略优化算法
- **速度跟踪控制**: 让机器人跟踪用户设定的速度命令
- **外部力干扰**: 增强机器人在外力干扰下的稳定性
- **对称性数据增强**: 左右镜像增强，提高泛化能力

### 机器人模型

- **机器人**: Unitree G1 (29 DOF)
- **仿真器**: Isaac Sim + Isaac Lab
- **运动数据**: AMCP (Adversarial Motion Control Priors) 数据集

---

## 核心功能

### 1. AMP 对抗模仿学习

```
GMR 人类运动数据 → AMP Discriminator → 风格奖励
                                         ↓
              PPO Policy ←←←←←←←←←←←←←←←←←←
```

AMP Discriminator 学习区分专家（人类）运动和机器人运动，产生风格奖励促使机器人模仿人类运动方式。

### 2. 速度跟踪控制

机器人跟踪用户设定的线性速度和角速度命令，用于遥控操作。

### 3. 外部力干扰

在训练过程中随机施加外部力，提高机器人的抗干扰能力。

### 4. 左右对称性数据增强

利用人形机器人的左右对称性，通过镜像变换扩充训练数据，提高泛化能力。

---

## 环境要求

### 硬件要求

- **GPU**: NVIDIA GPU with CUDA support (推荐 RTX 4070 或更高)
- **内存**: 32GB+
- **CPU**: 多核处理器

### 软件依赖

- **Isaac Sim**: 5.1+
- **Isaac Lab**: 最新版
- **Python**: 3.11
- **PyTorch**: 与 Isaac Lab 兼容版本
- **CUDA**: 11.8 或 12.1

---

## 安装步骤

### 1. 克隆仓库

```bash
git clone git@github.com:Antonxie/RLearning.git
cd RLearning
```

### 2. 配置 Isaac Lab 环境

参考 [Isaac Lab 官方文档](https://isaac-lab.github.io/IsaacLab/main/setup/installation.html) 进行环境配置。

### 3. 安装依赖

```bash
# 使用 Isaac Sim 的 Python 环境
/home/robot/StarBot/isaac-projects/IsaacLab/IsaacLab-main/_isaac_sim/python.sh -m pip install -r requirements.txt
```

---

## 快速启动

### 训练命令

```bash
cd /home/robot/StarBot/isaac-projects/legged_lab-main

# 在物理显示器上运行
export DISPLAY=:1

# 启动训练
/home/robot/StarBot/isaac-projects/IsaacLab/IsaacLab-main/_isaac_sim/python.sh \
    scripts/rsl_rl/train.py \
    --task LeggedLab-Isaac-AMP-G1-v0 \
    --num_envs 2000 \
    --device cuda:0
```

### 后台训练

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

### 查看训练进度关键指标

```bash
# disc_loss: 应该 > 0，表示 discriminator 正常工作
# expert_accuracy: 应该接近 1.0，表示专家数据正确输入
# Mean reward: 逐渐增加
```

---

## 代码结构

```
legged_lab-main/
├── scripts/
│   ├── rsl_rl/
│   │   └── train.py                 # 训练入口脚本
│   └── tools/
│       └── retarget/
│           └── config/
│               └── g1_29dof.yaml   # GMR→URDF 关节映射配置
├── source/
│   └── legged_lab/
│       └── legged_lab/
│           ├── assets/
│           │   └── unitree.py       # 机器人模型定义 (joint_sdk_names)
│           ├── rsl_rl/
│           │   ├── amp_runner.py    # AMP 训练 runner
│           │   ├── amp_algorithm.py  # PPO+AMP 算法
│           │   └── amp_networks.py   # Discriminator 网络
│           └── tasks/
│               └── locomotion/
│                   └── amp/
│                       ├── config/
│                       │   └── g1/
│                       │       └── g1_amp_env_cfg.py  # AMP 环境配置
│                       └── mdp/
│                           └── symmetry/
│                               └── g1.py  # 左右镜像对称配置
└── docs/
    └── AMP_TRAINING_DEPLOYMENT.md  # 详细部署文档
```

---

## 关键配置文件

### 1. g1_29dof.yaml - 关节映射配置

位置: `scripts/tools/retarget/config/g1_29dof.yaml`

定义 GMR 人类运动数据到机器人关节的映射。

**关键字段**:
- `gmr_dof_names`: 人类运动数据的关节顺序
- `lab_dof_names`: 机器人关节的实际顺序（必须与 USD 模型一致！）

### 2. unitree.py - 机器人模型

位置: `source/legged_lab/legged_lab/assets/unitree.py`

定义 G1 机器人的物理模型和关节配置。

**关键字段**:
- `joint_sdk_names`: 机器人关节的实际顺序

### 3. g1.py - 对称性配置

位置: `source/legged_lab/legged_lab/tasks/locomotion/amp/mdp/symmetry/g1.py`

定义左右镜像变换的索引映射。

### 4. g1_amp_env_cfg.py - AMP 环境配置

位置: `source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1/g1_amp_env_cfg.py`

定义 AMP 环境的观察空间、奖励函数等。

**关键字段**:
- `KEY_BODY_NAMES`: 6 个关键肢体位置

---

## 训练架构

### AMP+PPO 训练流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                         训练循环 (50000 iterations)                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐        │
│  │   Env Reset  │ ──▶ │  Collect Exp │ ──▶ │  Update Disc  │        │
│  └──────────────┘     └──────────────┘     └──────────────┘        │
│                              │                     │                 │
│                              ▼                     ▼                 │
│                       ┌──────────────┐     ┌──────────────┐         │
│                       │ Expert Demo  │     │ Update Policy│         │
│                       └──────────────┘     └──────────────┘         │
│                              │                     │                │
│                              ▼                     ▼                │
│                       ┌──────────────┐     ┌──────────────┐        │
│                       │ Disc Loss    │     │ Policy Loss  │        │
│                       │ (should > 0) │     │ (surrogate)  │        │
│                       └──────────────┘     └──────────────┘         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 奖励函数组成

| 奖励项 | 描述 | 典型值 |
|--------|------|--------|
| track_lin_vel_xy | 线性速度跟踪 | ~0.004 |
| track_ang_vel_z | 角速度跟踪 | ~0.005 |
| flat_orientation_l2 | 姿态稳定 | ~-0.007 |
| joint_deviation_hip | 髋关节偏移惩罚 | ~-0.002 |
| joint_deviation_arms | 手臂关节偏移惩罚 | ~-0.009 |
| termination_penalty | 终止惩罚 | -0.2 |
| **discriminator** | AMP 风格奖励 | 来自 Discriminator |

### 关键超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| num_envs | 2000 | 并行环境数 |
| max_iterations | 50000 | 最大训练迭代数 |
| learning_rate | 0.0001 | 学习率 |
| num_steps_per_env | 24 | 每回合步数 |
| disc_update_steps | 5 | Discriminator 更新步数 |

---

## 调试工具

### 1. 打印 G1 实际关节和 Link 名称

```bash
cd /home/robot/StarBot/isaac-projects/legged_lab-main
export DISPLAY=:1

/home/robot/StarBot/isaac-projects/IsaacLab/IsaacLab-main/_isaac_sim/python.sh \
    print_g1_links.py
```

输出示例：
```
============================================================
G1 29DOF Joint Names (DOFs):
============================================================
  [ 0] left_hip_pitch_joint
  [ 1] right_hip_pitch_joint
  [ 2] waist_yaw_joint
  ...

============================================================
G1 29DOF Body Names (Links):
============================================================
  [ 0] pelvis
  [ 1] left_hip_pitch_link
  ...
```

### 2. 验证关节顺序一致性

运行训练时观察以下指标：
- `disc_loss`: 应该 > 0
- `expert_accuracy`: 应该接近 1.0

---

## 常见问题

### Q1: ModuleNotFoundError: No module named 'isaacsim'

**原因**: 使用了错误的 Python 解释器

**解决**:
```bash
# ✅ 正确方式
/home/robot/StarBot/isaac-projects/IsaacLab/IsaacLab-main/_isaac_sim/python.sh train.py
```

### Q2: disc_loss = 0

**原因**: 关节映射配置错误

**排查**:
1. 检查 `g1_29dof.yaml` 的 `lab_dof_names` 是否与 `unitree.py` 的 `joint_sdk_names` 一致
2. 检查 `symmetry/g1.py` 的索引是否正确

### Q3: 机器人手臂乱动

**原因**: GMR 数据被错误映射到不对应的关节

**解决**: 按照文档中的关节顺序表格修正配置

### Q4: 训练启动后崩溃 (segmentation fault)

**解决**:
```bash
# 使用正确的 DISPLAY
export DISPLAY=:1  # 或 :0

# 或使用 headless 模式
unset DISPLAY
```

---

## 技术细节

### USD 模型关节顺序

G1 29DOF 的实际关节顺序（来自 USD 模型）：

| 索引 | 关节名称 | 左/右 | 部位 |
|------|---------|-------|------|
| 0 | left_hip_pitch_joint | 左 | 腿 |
| 1 | right_hip_pitch_joint | 右 | 腿 |
| 2 | waist_yaw_joint | 中 | 腰 |
| 3 | left_hip_roll_joint | 左 | 腿 |
| 4 | right_hip_roll_joint | 右 | 腿 |
| 5 | waist_roll_joint | 中 | 腰 |
| 6 | left_hip_yaw_joint | 左 | 腿 |
| 7 | right_hip_yaw_joint | 右 | 腿 |
| 8 | waist_pitch_joint | 中 | 腰 |
| 9 | left_knee_joint | 左 | 腿 |
| 10 | right_knee_joint | 右 | 腿 |
| 11 | left_shoulder_pitch_joint | 左 | 臂 |
| 12 | right_shoulder_pitch_joint | 右 | 臂 |
| 13 | left_ankle_pitch_joint | 左 | 腿 |
| 14 | right_ankle_pitch_joint | 右 | 腿 |
| 15 | left_shoulder_roll_joint | 左 | 臂 |
| 16 | right_shoulder_roll_joint | 右 | 臂 |
| 17 | left_ankle_roll_joint | 左 | 腿 |
| 18 | right_ankle_roll_joint | 右 | 腿 |
| 19 | left_shoulder_yaw_joint | 左 | 臂 |
| 20 | right_shoulder_yaw_joint | 右 | 臂 |
| 21 | left_elbow_joint | 左 | 臂 |
| 22 | right_elbow_joint | 右 | 臂 |
| 23 | left_wrist_roll_joint | 左 | 臂 |
| 24 | right_wrist_roll_joint | 右 | 臂 |
| 25 | left_wrist_pitch_joint | 左 | 臂 |
| 26 | right_wrist_pitch_joint | 右 | 臂 |
| 27 | left_wrist_yaw_joint | 左 | 臂 |
| 28 | right_wrist_yaw_joint | 右 | 臂 |

**注意**: 这个顺序是**左右交替**的，不是左腿集中！

### 关键肢体位置 (Key Body Names)

用于 AMP Discriminator 观察的 6 个关键肢体：

| 索引 | Link 名称 | 说明 |
|------|----------|------|
| 0 | left_ankle_roll_link | 左脚踝 |
| 1 | right_ankle_roll_link | 右脚踝 |
| 2 | left_wrist_yaw_link | 左手腕 |
| 3 | right_wrist_yaw_link | 右手腕 |
| 4 | left_shoulder_roll_link | 左肩 |
| 5 | right_shoulder_roll_link | 右肩 |

### 左右镜像索引

```python
# 左关节索引
left_indices = [0, 3, 6, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27]

# 右关节索引
right_indices = [1, 4, 7, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]

# Roll 关节 (需要符号翻转)
roll_indices = [3, 4, 15, 16]

# Yaw 关节 (需要符号翻转)
yaw_indices = [6, 7, 19, 20, 27, 28]

# 腰部关节 (不参与镜像)
waist_indices = [2, 5, 8]
```

---

## 维护记录

| 日期 | 修改内容 | 说明 |
|------|---------|------|
| 2026-03-23 | 初始版本 | 实现 AMP+PPO+GMR 训练框架 |
| 2026-03-23 | 修复关节映射 | 发现 USD 模型关节顺序是左右交替而非左腿集中 |
| 2026-03-23 | 上传代码 | 提交到 GitHub |

---

## 联系方式

- GitHub: [Antonxie/RLearning](git@github.com:Antonxie/RLearning.git)

## 致谢

- [Isaac Lab](https://isaac-lab.github.io/IsaacLab/) - 仿真框架
- [Unitree](https://www.unitree.com/) - 机器人硬件
- [AMP-RSL-RL](https://github.com/gbionics/amp-rsl-rl) - AMP 算法参考
