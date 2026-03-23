# G1机器人复合训练计划：人类行走风格 + 速度跟踪 + 外力干扰抵抗

## 1. 项目概述

本训练计划旨在训练Unitree G1人形机器人实现：
- **人类-like行走风格**：通过DeepMimic动作模仿学习自然运动模式
- **速度命令跟踪**：支持遥控器/手柄的速度指令输入
- **外力干扰抵抗**：在受到外部推力时保持平衡并继续执行任务

### 1.1 技术架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     PPO强化学习训练框架                           │
├─────────────────────────────────────────────────────────────────┤
│  观测空间 (Observation)                                          │
│  ├── 机器人本体感知：关节位置、关节速度、根姿态、角速度            │
│  ├── 动作参考：参考动作的根位置/旋转、关节角度、关键body位置        │
│  └── 速度命令：当前速度命令值                                     │
├─────────────────────────────────────────────────────────────────┤
│  动作空间 (Action)                                               │
│  └── 目标关节位置（29 DOF）                                      │
├─────────────────────────────────────────────────────────────────┤
│  奖励函数 (Reward)                                               │
│  ├── 运动模仿奖励（DeepMimic）                                   │
│  │   ├── 根位置跟踪误差  weight=0.15                             │
│  │   ├── 根姿态跟踪误差  weight=0.08                              │
│  │   ├── 根速度跟踪误差  weight=0.10                              │
│  │   ├── 根角速度跟踪误差 weight=0.05                            │
│  │   ├── 关键body位置误差 weight=0.15                            │
│  │   ├── 关节位置误差   weight=0.50                              │
│  │   └── 关节速度误差   weight=0.10                              │
│  ├── 速度跟踪奖励                                              │
│  │   ├── XY线性速度跟踪 weight=0.20                              │
│  │   └── Z轴角速度跟踪  weight=0.10                              │
│  └── 正则化奖励                                                 │
│      ├── 关节扭矩L2     weight=-0.001                            │
│      ├── 关节加速度L2   weight=-0.01                             │
│      └── 动作变化率L2   weight=-0.001                            │
├─────────────────────────────────────────────────────────────────┤
│  干扰训练 (Disturbance)                                         │
│  ├── 外部力/力矩扰动：每5-10秒施加0-30N力、0-10Nm力矩              │
│  └── 机器人推动扰动：每5-10秒施加随机推动速度                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 核心技术背景

### 2.1 DeepMimic动作模仿

DeepMimic是一种基于物理的强化学习方法，通过参考动作数据（MoCap或手工动画）来训练机器人学习复杂的运动技能。

**核心思想**：
- 使用动作捕捉数据或动画作为参考轨迹
- 奖励函数衡量机器人状态与参考状态的差异
- 通过物理引擎确保机器人遵循物理定律

**参考论文**：
- DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills (SIGGRAPH 2018)

### 2.2 速度命令跟踪

速度命令跟踪使机器人能够响应外部输入（如遥控器）并跟踪期望的移动速度。

**实现方式**：
- 引入UniformVelocityCommand，持续生成随机速度命令
- 奖励函数鼓励机器人跟踪这些命令
- 支持朝向控制（heading command）

### 2.3 外力干扰抵抗

通过在训练过程中引入随机外部力，增强机器人在真实环境中的鲁棒性。

---

## 3. 训练配置详情

### 3.1 环境注册

**任务名称**：`LeggedLab-Isaac-Deepmimic-Velocity-G1-v0`

**配置文件**：
- 基础配置：`g1_deepmimic_env_cfg.py`
- 复合配置：`g1_deepmimic_env_cfg.py::G1DeepMimicVelocityEnvCfg`

### 3.2 关键参数

| 参数类别 | 参数名称 | 值 | 说明 |
|---------|---------|-----|------|
| 环境 | num_envs | 16-256 | 并行环境数量，根据GPU内存调整 |
| 环境 | episode_length_s | 10.0 | 每个episode时长 |
| 动作 | joint_pos scale | 1.0 | 关节位置控制缩放 |
| 速度命令 | lin_vel_x | (0.0, 1.0) m/s | 前进速度范围 |
| 速度命令 | lin_vel_y | (-0.3, 0.3) m/s | 侧向速度范围 |
| 速度命令 | ang_vel_z | (-1.0, 1.0) rad/s | 偏航角速度范围 |
| 外力扰动 | force_range | (0, 30) N | 外部力范围 |
| 外力扰动 | torque_range | (0, 10) Nm | 外部力矩范围 |
| 外力扰动 | interval | 5-10 s | 扰动施加间隔 |

### 3.3 奖励函数权重

| 奖励项 | 权重 | 标准差(std) |
|-------|------|------------|
| ref_track_root_pos_w_error_exp | 0.15 | 0.5 |
| ref_track_quat_error_exp | 0.08 | 0.5 |
| ref_track_root_vel_w_error_exp | 0.10 | 1.0 |
| ref_track_root_ang_vel_w_error_exp | 0.05 | 1.0 |
| ref_track_key_body_pos_b_error_exp | 0.15 | 0.3 |
| ref_track_dof_pos_error_exp | 0.50 | 2.0 |
| ref_track_dof_vel_error_exp | 0.10 | 10.0 |
| track_lin_vel_xy_exp | 0.20 | 0.5 |
| track_ang_vel_z_exp | 0.10 | 0.5 |

---

## 4. 训练方法

### 4.1 前置条件

```bash
# Isaac Sim + Isaac Lab 环境
# GPU: NVIDIA RTX 4070+ (建议12GB+显存)
# 运动数据目录: legged_lab/data/MotionData/g1_29dof/deepmimic/
```

### 4.2 启动训练（物理显示器）

```bash
# 设置显示器环境变量
export DISPLAY=:1

# 进入项目目录
cd /home/robot/StarBot/isaac-projects/legged_lab-main

# 启动训练
/home/robot/StarBot/isaac-projects/isaac_sim/python.sh scripts/rsl_rl/train.py \
    --task LeggedLab-Isaac-Deepmimic-Velocity-G1-v0 \
    --num_envs 16
```

### 4.3 启动训练（虚拟显示器）

```bash
# 启动Xvfb虚拟显示器
Xvfb :99 -screen 0 1920x1080x24 &

# 设置环境变量
export DISPLAY=:99

# 启动训练
cd /home/robot/StarBot/isaac-projects/legged_lab-main
/home/robot/StarBot/isaac-projects/isaac_sim/python.sh scripts/rsl_rl/train.py \
    --task LeggedLab-Isaac-Deepmimic-Velocity-G1-v0 \
    --num_envs 256 \
    --headless
```

### 4.4 启动训练（无显示器模式）

```bash
export DISPLAY=:99
cd /home/robot/StarBot/isaac-projects/legged_lab-main
/home/robot/StarBot/isaac-projects/isaac_sim/python.sh scripts/rsl_rl/train.py \
    --task LeggedLab-Isaac-Deepmimic-Velocity-G1-v0 \
    --num_envs 256 \
    --headless
```

---

## 5. 预期目标

### 5.1 训练阶段划分

| 阶段 | Iteration | 目标 | 预期奖励 |
|------|-----------|------|---------|
| 初期 | 0-100 | 学会保持平衡，参考动作误差降低 | 0.05-0.15 |
| 中期 | 100-1000 | 跟踪速度命令，抵抗轻微扰动 | 0.15-0.35 |
| 后期 | 1000-5000 | 流畅的人类行走模式，稳定跟踪 | 0.35-0.55 |
| 完成 | 5000-10000 | 鲁棒的速度跟踪，抗干扰能力强 | 0.55+ |

### 5.2 成功标准

- 机器人能够以类似人类的步态行走
- 能够跟踪遥控器发送的速度命令
- 受到30N外部推力后能在2秒内恢复平衡
- Episode长度维持在20步以上（不被终止）

### 5.3 关键指标监控

```
Episode_Reward/track_lin_vel_xy_exp      # 速度跟踪XY
Episode_Reward/track_ang_vel_z_exp        # 角速度跟踪Z
Episode_Reward/ref_track_dof_pos_error_exp # 关节位置跟踪
Metrics/base_velocity/error_vel_xy        # XY速度误差
Metrics/base_velocity/error_vel_yaw      # 偏航角速度误差
Episode_Termination/base_contact          # 摔倒终止率
```

---

## 6. 文件结构

```
legged_lab-main/
├── source/legged_lab/legged_lab/tasks/locomotion/deepmimic/
│   ├── deepmimic_env_cfg.py           # 基础环境配置
│   ├── config/g1/
│   │   ├── __init__.py                # 环境注册
│   │   └── g1_deepmimic_env_cfg.py   # G1特定配置
│   └── mdp/
│       ├── rewards.py                 # 奖励函数
│       └── ...
├── scripts/rsl_rl/
│   └── train.py                       # 训练入口
└── data/MotionData/g1_29dof/deepmimic/ # 运动数据
```

---

## 7. 相关文档

- [DEPLOYMENT_GUIDE.md](../DEPLOYMENT_GUIDE.md) - 部署指南
- [DeepMimic论文](https://xbpeng.github.io/projects/DeepMimic/index.html)
- [Isaac Lab文档](https://isaac-sim.github.io/IsaacLab/main/)

---

## 8. 常见问题排查

### 8.1 GPU内存不足
```bash
# 减少环境数量
--num_envs 16

# 清理残留进程
pkill -9 -f python3
nvidia-smi
```

### 8.2 显示器连接问题
```bash
# 检查可用显示器
ls /tmp/.X11-unix/

# 使用虚拟显示器
export DISPLAY=:99
```

### 8.3 训练崩溃查看
```bash
# 启用完整错误追踪
export HYDRA_FULL_ERROR=1
```
