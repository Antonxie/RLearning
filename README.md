# G1 Humanoid Robot AMP+PPO Training Repository

## 项目概述

本项目是基于 Isaac Lab/Isaac Sim 平台实现的 G1 人形机器人 AMP (Adversarial Motion Priors) + PPO (Proximal Policy Optimization) 强化学习训练系统。通过结合人类运动数据、外力干扰训练和速度命令跟踪，实现具有人类行走风格、鲁棒性强的 G1 机器人运动策略。

### 核心特性

- **AMP (对抗运动先验)**: 通过对抗学习让机器人模仿人类运动风格
- **PPO (近端策略优化)**: 稳定可靠的策略优化算法
- **速度跟踪**: 训练机器人响应遥控器速度命令
- **外力干扰**: 增强机器人在外力扰动下的平衡能力
- **对称数据增强**: 左右镜像增强提高策略泛化能力

---

## 目录结构

```
legged_lab-main/
├── README.md                          # 本文档
├── docs/
│   └── AMP_TRAINING_DEPLOYMENT.md     # 详细的AMP训练部署指南
├── source/legged_lab/
│   ├── legged_lab/
│   │   ├── assets/
│   │   │   └── unitree.py             # 机器人模型定义 (USD关节顺序)
│   │   ├── rsl_rl/
│   │   │   ├── amp_runner.py          # AMP训练Runner
│   │   │   ├── amp_algorithm.py       # PPO+AMP算法实现
│   │   │   ├── amp_networks.py         # AMP判别器网络
│   │   │   └── amp_cfg.py             # AMP配置类
│   │   │   └── rl_cfg.py              # RL基础配置
│   │   └── tasks/locomotion/amp/
│   │       ├── amp_env_cfg.py         # AMP环境基础配置
│   │       └── config/g1/
│   │           ├── g1_amp_env_cfg.py  # G1专用AMP环境配置
│   │           └── agents/
│   │               └── rsl_rl_ppo_cfg.py  # PPO训练配置
│   │       └── mdp/
│   │           ├── rewards.py         # 奖励函数定义
│   │           ├── observations.py    # 观测函数定义
│   │           ├── events.py          # 事件处理
│   │           └── symmetry/
│   │               └── g1.py          # G1对称变换(关键！)
│   └── data/
│       └── MotionData/                # 人类运动数据 (Git LFS)
├── scripts/
│   ├── rsl_rl/
│   │   ├── train.py                   # 训练入口
│   │   └── play.py                    # 播放/测试入口
│   └── tools/retarget/
│       └── config/
│           └── g1_29dof.yaml          # 运动数据重定向配置(关键！)
└── scripts/rsl_rl/rsl_rl_ppo_cfg.py   # 训练超参数配置
```

---

## 1. 系统架构

### 1.1 整体架构

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

### 1.2 AMP 训练流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            AMP+PPO 训练流程                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. 人类运动数据 (MotionData)                                                │
│          │                                                                  │
│          ▼                                                                  │
│  2. GMR (高斯混合回归) 生成参考轨迹                                          │
│          │                                                                  │
│          ▼                                                                  │
│  3. AMP Discriminator (判别器)                                               │
│     - 接收 expert 演示数据 (gmr_dof_names → lab_dof_names)                  │
│     - 接收 policy 生成的运动数据                                              │
│     - 输出 style_reward 奖励信号                                              │
│          │                                                                  │
│          ▼                                                                  │
│  4. PPO 算法优化策略                                                         │
│     - 结合 style_reward + velocity_reward + 其他奖励                         │
│     - 更新策略网络                                                           │
│          │                                                                  │
│          ▼                                                                  │
│  5. Symmetry Augmentation (对称增强)                                         │
│     - 左右镜像数据增强                                                       │
│     - 提高策略泛化能力                                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 安装与设置

### 2.1 环境要求

- **Isaac Sim**: 2023.1.0 或更高版本
- **Isaac Lab**: 最新版本
- **CUDA**: 11.8 或更高版本
- **Python**: 3.10+ (Isaac Sim 内置 Python)

### 2.2 关键安装步骤

```bash
# 1. 克隆 Isaac Lab (如果尚未安装)
cd /home/robot/StarBot/isaac-projects
git clone https://github.com/isaac-sim/IsaacLab.git

# 2. 安装 legged_lab
cd legged_lab-main
pip install -e source/legged_lab/

# 3. 安装 amp-rsl-rl (用于 AMP 训练)
git clone https://github.com/gbionics/amp-rsl-rl.git
pip install -e amp-rsl-rl/
```

### 2.3 运动数据安装 (Git LFS)

AMP 训练需要人类运动数据，必须使用 Git LFS 下载：

```bash
# 初始化 Git LFS
git lfs install

# 克隆仓库 (会自动下载 LFS 文件)
git clone https://github.com/Antonxie/RLearning.git

# 验证运动数据
ls -la source/legged_lab/data/MotionData/
```

---

## 3. 训练指南

### 3.1 正确启动方式 (重要！)

**必须使用 Isaac Sim 内置的 Python 解释器，而非 conda 的 Python：**

```bash
cd /home/robot/StarBot/isaac-projects/legged_lab-main

# 使用物理显示器1
export DISPLAY=:1

# 启动训练
/home/robot/StarBot/isaac-projects/IsaacLab/IsaacLab-main/_isaac_sim/python.sh \
    scripts/rsl_rl/train.py \
    --task LeggedLab-Isaac-AMP-G1-v0 \
    --num_envs 2000 \
    --device cuda:0
```

### 3.2 后台运行

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

### 3.3 训练参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--task` | 训练任务名称 | `LeggedLab-Isaac-AMP-G1-v0` |
| `--num_envs` | 并行环境数量 | `2000` |
| `--device` | 计算设备 | `cuda:0` |
| `--headless` | 无头模式(无图形) | 默认关闭 |

---

## 4. 关键配置说明

### 4.1 关节映射关系 (最重要！)

AMP 训练涉及三个系统的关节命名映射，必须确保一致：

| 系统 | 来源 | 用途 |
|------|------|------|
| **GMR** | `g1_29dof.yaml` 中的 `gmr_dof_names` | 人类运动数据 |
| **URDF/USD** | `unitree.py` 中的 `joint_sdk_names` | 机器人实际关节 |
| **AMP** | `g1_amp_env_cfg.py` 中的 `observation groups` | 判别器输入 |

**关键规则**：
- `lab_dof_names` 必须与 `unitree.py joint_sdk_names` **完全一致**
- `gmr_dof_names` 到 `lab_dof_names` 的映射是 **retargeting** 的核心

### 4.2 G1 关节顺序 (29 DOF) - gmr_dof_names / lab_dof_names

**重要**: `gmr_dof_names` 和 `lab_dof_names` 必须有完全相同的顺序！

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

### 4.3 Key Body Names (AMP 判别器用)

用于 AMP discriminator 观察的 6 个关键肢体位置：

| 索引 | Key Body Name (Link) | 说明 |
|------|---------------------|------|
| 0 | left_ankle_roll_link | 左脚踝 |
| 1 | right_ankle_roll_link | 右脚踝 |
| 2 | left_wrist_yaw_link | 左手腕（手臂摆动） |
| 3 | right_wrist_yaw_link | 右手腕（手臂摆动） |
| 4 | left_shoulder_roll_link | 左肩（手臂摆动） |
| 5 | right_shoulder_roll_link | 右肩（手臂摆动） |

### 4.4 Symmetry 索引 (对称增强)

**重要**: 索引基于修正后的 gmr_dof_names / lab_dof_names 顺序

```python
# 左关节索引 (先左腿0-5，再左臂15-21)
left_indices = [0, 1, 2, 3, 4, 5, 15, 16, 17, 18, 19, 20, 21]

# 右关节索引 (先右腿6-11，再右臂22-28)
right_indices = [6, 7, 8, 9, 10, 11, 22, 23, 24, 25, 26, 27, 28]

# Roll 关节 (需要符号翻转)
roll_indices = [1, 4, 5, 7, 10, 11, 16, 19, 23, 26]

# Yaw 关节 (需要符号翻转)
yaw_indices = [2, 8, 17, 21, 24, 28]

# Waist 关节 (不参与镜像)
waist_indices = [12, 13, 14]
```

---

## 5. 关键配置文件清单

| 文件路径 | 用途 | 关键内容 |
|---------|------|---------|
| `scripts/tools/retarget/config/g1_29dof.yaml` | GMR→URDF 映射 | gmr_dof_names, lab_dof_names |
| `source/legged_lab/tasks/locomotion/amp/config/g1/g1_amp_env_cfg.py` | AMP环境配置 | KEY_BODY_NAMES, observation groups |
| `source/legged_lab/tasks/locomotion/amp/mdp/symmetry/g1.py` | 对称变换 | left_indices, right_indices, roll_indices |
| `source/legged_lab/assets/unitree.py` | 机器人模型 | joint_sdk_names (29 DOF) |
| `source/legged_lab/rsl_rl/amp_cfg.py` | AMP配置 | disc_learning_rate, grad_penalty_scale |

---

## 6. 奖励函数设计

### 6.1 奖励组成

| 奖励项 | 说明 | 权重 |
|--------|------|------|
| `tracking_linear_velocity` | 线性速度跟踪 | 高 |
| `tracking_angular_velocity` | 角速度跟踪 | 中 |
| `linear_velocity` | 实际线性速度 | 中 |
| `angular_velocity` | 实际角速度 | 中 |
| `style_reward` | AMP风格奖励 | 高 |
| `joint_acceleration` | 关节加速度惩罚 | 低 |
| `action_rate` | 动作变化惩罚 | 低 |
| `feet_air_time` | 脚离地时间奖励 | 中 |
| `feet_contact` | 脚接触奖励 | 中 |

### 6.2 外力干扰

训练期间会随机施加外部力干扰，提高机器人抗扰动能力：

```yaml
external_force:
  scale: 100.0          # 外力大小
  duration: 0.2         # 持续时间 (秒)
  interval: 5.0          # 施加间隔 (秒)
  range: [0.5, 1.5]     # 随机范围系数
```

---

## 7. 常见问题排查

### 7.1 ModuleNotFoundError: No module named 'isaacsim'

**原因**: 使用了错误的 Python 解释器

**解决**:
```bash
# ❌ 错误
python scripts/rsl_rl/train.py
conda activate isaaclab && python scripts/rsl_rl/train.py

# ✅ 正确
/home/robot/StarBot/isaac-projects/IsaacLab/IsaacLab-main/_isaac_sim/python.sh scripts/rsl_rl/train.py
```

### 7.2 disc_loss = 0.0

**可能原因**:
1. expert 数据未正确加载
2. lab_dof_names 与 USD joint 顺序不一致
3. MotionData 文件未下载

**排查步骤**:
```bash
# 1. 检查 MotionData 是否存在
ls -la source/legged_lab/data/MotionData/

# 2. 检查关节顺序一致性
grep -A 30 "joint_sdk_names" source/legged_lab/assets/unitree.py
```

### 7.3 机器人手臂乱动

**原因**: GMR 数据被错误映射到不对应的机器人关节

**解决**: 修正 `g1_29dof.yaml` 中的 `lab_dof_names`，确保与 USD 关节顺序一致

### 7.4 训练启动后立即崩溃 (segmentation fault)

**原因**: DISPLAY 环境变量设置问题

**解决**:
```bash
# 使用显示器1时
export DISPLAY=:1

# 或不使用 DISPLAY (headless 模式)
unset DISPLAY
```

---

## 8. 参考资源

### 8.1 相关论文

- **AMP (Adversarial Motion Priors)**: [AMP: Adversarial Motion Priors for Stylized Physics-Based Control](https://arxiv.org/abs/2104.02180)
- **PPO**: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

### 8.2 开源项目

- [Isaac Lab](https://github.com/isaac-sim/IsaacLab) - NVIDIA Isaac Lab 仿真平台
- [amp-rsl-rl](https://github.com/gbionics/amp-rsl-rl) - AMP+RSL-RL 实现
- [DeepMimic](https://github.com/xbpeng/DeepMimic) - 动作模仿学习

### 8.3 机器人模型

- [Unitree G1](https://www.unitree.com/g1) - 本项目使用的 G1 人形机器人

---

## 9. 致谢

- NVIDIA Isaac Lab 团队
- Unitree 机器人公司
- AMP-rsl-rl 开源社区

---

## 10. 许可证

本项目基于 MIT 许可证开源。
