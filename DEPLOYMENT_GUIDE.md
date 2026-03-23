# G1 机器人 DeepMimic 训练部署流程

本文档详细介绍如何在本地/服务器环境中部署和运行 G1 机器人的 DeepMimic 动作模仿学习训练。

## 环境要求

### 硬件要求
- GPU: NVIDIA GPU (建议 8GB+ 显存)
- 内存: 16GB+
- 操作系统: Ubuntu 20.04 / 22.04

### 软件要求
- Isaac Sim 5.1
- Isaac Lab 2.3.2
- Python 3.10
- CUDA 11.8+ / 12.0+

## 1. 安装 Isaac Sim 5.1

### 1.1 下载 Isaac Sim
```bash
# 从 NVIDIA Omniverse 下载 Isaac Sim 5.1
# 下载地址: https://developer.nvidia.com/isaac-sim
```

### 1.2 安装步骤
```bash
# 解压安装包
cd /home/robot/StarBot/isaac-projects
mkdir -p isaac_sim
# 将下载的 Isaac Sim 安装到该目录
```

### 1.3 设置环境变量
```bash
# 在 ~/.bashrc 中添加
export ISAAC_SIM_PATH=/home/robot/StarBot/isaac-projects/isaac_sim
export PATH=$ISAAC_SIM_PATH:$PATH

# 刷新环境
source ~/.bashrc
```

## 2. 安装 Isaac Lab 2.3.2

### 2.1 克隆仓库
```bash
cd /home/robot/StarBot/isaac-projects
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout v2.3.2
```

### 2.2 安装依赖
```bash
# 使用 Isaac Sim 的 Python 环境
cd /home/robot/StarBot/isaac-projects/isaac_sim
source ./setup_python_env.sh

# 安装 Isaac Lab
cd /home/robot/StarBot/isaac-projects/IsaacLab
pip install -e .
```

## 3. 安装 legged_lab

### 3.1 克隆仓库
```bash
cd /home/robot/StarBot/isaac-projects
git clone https://github.com/zitongbai/legged_lab.git legged_lab-main
```

### 3.2 安装 legged_lab
```bash
cd /home/robot/StarBot/isaac-projects/legged_lab-main
pip install -e .
```

## 4. 准备 G1 机器人模型

legged_lab 已包含 G1 机器人模型，位于:
```
/home/robot/StarBot/isaac-projects/legged_lab-main/source/legged_lab/legged_lab/data/Robots/Unitree/g1_29dof/usd/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd
```

## 5. 运行 DeepMimic 训练

### 5.1 训练命令

**方式一: 无头模式 (headless)**
```bash
cd /home/robot/StarBot/isaac-projects/legged_lab-main
/home/robot/StarBot/isaac-projects/isaac_sim/python.sh scripts/rsl_rl/train.py \
    --task LeggedLab-Isaac--Deepmimic-G1-v0 \
    --num_envs 64 \
    --max_iterations 10000
```

**方式二: 显示训练画面**
```bash
# 首先启动 Xvfb (无显示器环境)
Xvfb :99 -screen 0 1920x1080x24 &

# 或者使用本地显示器
export DISPLAY=:1

# 运行训练
cd /home/robot/StarBot/isaac-projects/legged_lab-main
/home/robot/StarBot/isaac-projects/isaac_sim/python.sh scripts/rsl_rl/train.py \
    --task LeggedLab-Isaac--Deepmimic-G1-v0 \
    --num_envs 64 \
    --max_iterations 10000
```

### 5.2 参数说明
| 参数 | 说明 | 默认值 |
|------|------|--------|
| --task | 任务名称 | 必填 |
| --num_envs | 并行环境数量 | 64 |
| --max_iterations | 最大迭代次数 | 50000 |
| --headless | 无头模式运行 | False |
| --device | 运行设备 | cuda:0 |

### 5.3 训练输出
训练日志默认保存在:
```
/home/robot/StarBot/isaac-projects/legged_lab-main/logs/rsl_rl/g1_deepmimic/
```

模型检查点保存在:
```
/home/robot/StarBot/isaac-projects/legged_lab-main/logs/rsl_rl/g1_deepmimic/<run_name>/models/
```

## 6. 切换训练动作

DeepMimic 支持多种动作训练。要切换动作，需要修改配置文件:

```bash
# 编辑配置文件
vim /home/robot/StarBot/isaac-projects/legged_lab-main/source/legged_lab/legged_lab/tasks/locomotion/deepmimic/config/g1/g1_deepmimic_env_cfg.py
```

找到以下行并修改:
```python
self.motion_data.motion_dataset.motion_data_weights = {
    "动作名称": 1.0,
}
```

## 7. 可用的 DeepMimic 动作列表

legged_lab 提供以下 DeepMimic 动作可用于训练:

### 7.1 行走类 (Walk)
| 文件名 | 动作描述 |
|--------|----------|
| C4_-_run_to_walk_a_stageii.pkl | 跑步→行走 |
| C18_-_run_to_hop_to_walk_stageii.pkl | 跑步→跳跃→行走 |
| C26_-_run_to_crouch_stageii.pkl | 跑步→蹲下 |

### 7.2 踢腿类 (Kick)
| 文件名 | 动作描述 |
|--------|----------|
| G5_-__back_kick_stageii.pkl | 后踢 (Back Kick) |
| G10-__roundhouse_leading_left_stageii.pkl | 侧踢 - 左侧 (Roundhouse Kick Left) |
| G12-__cresent_left_stageii.pkl | 半月踢 - 左侧 (Crescent Kick Left) |
| G13-__cresent_right_stageii.pkl | 半月踢 - 右侧 (Crescent Kick Right) |
| E5_-__hook_left_stageii.pkl | 钩踢 - 左侧 (Hook Kick Left) |

### 7.3 旋转类 (Spin)
| 文件名 | 动作描述 |
|--------|----------|
| G19-__reverse_spin_cresent_left_stageii.pkl | 反转半月踢 - 左侧 |
| G20_-__reverse_spin_cresent_right_stageii.pkl | 反转半月踢 - 右侧 |

### 7.4 转向类 (Turn)
| 文件名 | 动作描述 |
|--------|----------|
| C11_-_run_turn_left_90_stageii.pkl | 跑动左转 90° |
| C14_-_run_turn_right_90_stageii.pkl | 跑动右转 90° |

## 8. 播放训练好的模型

训练完成后，可以使用以下命令播放模型:

### 8.1 本地显示器播放
如果本地有显示器（X11转发或VNC），使用以下命令:

```bash
# 设置显示器环境变量
export DISPLAY=:1

# 进入项目目录
cd /home/robot/StarBot/isaac-projects/legged_lab-main

# 播放模型（单环境实时显示）
/home/robot/StarBot/isaac-projects/isaac_sim/python.sh scripts/rsl_rl/play.py \
    --task LeggedLab-Isaac--Deepmimic-G1-Play-v0 \
    --num_envs 1 \
    --checkpoint /home/robot/StarBot/isaac-projects/legged_lab-main/logs/rsl_rl/g1_deepmimic/2026-03-17_14-43-44/model_9999.pt
```

### 8.2 无显示器播放（生成视频）
在没有显示器的服务器上运行，使用 `--video` 参数生成训练视频:

```bash
# 设置显示器环境变量
export DISPLAY=:1

# 进入项目目录
cd /home/robot/StarBot/isaac-projects/legged_lab-main

# 生成视频（默认500帧）
/home/robot/StarBot/isaac-projects/isaac_sim/python.sh scripts/rsl_rl/play.py \
    --task LeggedLab-Isaac--Deepmimic-G1-Play-v0 \
    --num_envs 1 \
    --checkpoint /home/robot/StarBot/isaac-projects/legged_lab-main/logs/rsl_rl/g1_deepmimic/2026-03-17_14-43-44/model_9999.pt \
    --video --video_length 500 --headless
```

视频将保存在模型目录的 `videos/play` 文件夹中。

## 9. 常见问题

### 9.1 模块导入错误
如果遇到 `ModuleNotFoundError`，确保 Isaac Lab 已正确安装:
```bash
cd /home/robot/StarBot/isaac-projects/IsaacLab
pip install -e .
```

### 9.2 USD 文件找不到
确保 G1 机器人模型文件存在:
```bash
ls -la /home/robot/StarBot/isaac-projects/legged_lab-main/source/legged_lab/legged_lab/data/Robots/Unitree/g1_29dof/usd/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd
```

### 9.3 显示器相关错误
在没有显示器的服务器上运行，使用 Xvfb:
```bash
# 安装 Xvfb (如未安装)
sudo apt install xvfb

# 启动虚拟显示器
Xvfb :99 -screen 0 1920x1080x24 &

# 设置环境变量
export DISPLAY=:99
```

### 9.4 训练速度慢
- 减少并行环境数量: `--num_envs 32`
- 使用 GPU 加速: `--device cuda:0`

## 10. 训练监控

### 10.1 TensorBoard
```bash
cd /home/robot/StarBot/isaac-projects/legged_lab-main
tensorboard --logdir=logs/rsl_rl/g1_deepmimic/
```

### 10.2 查看训练日志
```bash
tail -f /tmp/deepmimic_walk_train.log
```

## 11. 当前训练状态

**正在训练:**
- 任务: DeepMimic G1 行走
- 动作: C4_-_run_to_walk_a_stageii (跑步→行走)
- 进度: 100/10000 次迭代
- 预计完成时间: 约 4 小时
- 速度: ~1300 steps/秒

---

**文档创建日期:** 2026-03-17
**legged_lab 版本:** GitHub latest
**Isaac Sim 版本:** 5.1
**Isaac Lab 版本:** 2.3.2

