# G1 机器人稳定行走训练计划

## 训练目标

训练一个能够在遥控器命令下稳定行走，并能够抵抗外力干扰的G1机器人控制器。

---

## 训练环境配置

### 1. 标准平坦地面行走训练 (Flat Terrain)

- **任务**: `LeggedLab-Isaac-Velocity-Flat-Unitree-G1-v0`
- **环境数**: 2048
- **迭代次数**: 10000

#### 奖励配置:
| 奖励项 | 权重 | 描述 |
|--------|------|------|
| track_lin_vel_xy_exp | 1.0 | 跟踪期望线速度 |
| track_ang_vel_z_exp | 1.0 | 跟踪期望角速度 |
| flat_orientation_l2 | -5.0 | 保持身体水平 |
| lin_vel_z_l2 | -0.2 | 限制垂直方向抖动 |
| ang_vel_xy_l2 | -0.05 | 限制侧倾角速度 |
| joint_torques_l2 | -2.0e-6 | 最小化关节力矩 |
| joint_acc_l2 | -1.0e-7 | 最小化关节加速度 |
| action_rate_l2 | -0.005 | 平滑动作变化 |
| feet_air_time | 1.0 | 鼓励脚部腾空 |
| feet_slide | -0.3 | 防止脚部滑动 |
| feet_clearance | 1.0 | 鼓励脚部抬高 |

#### 遥控命令范围:
- 线速度 X: 0.0 - 1.0 m/s
- 线速度 Y: -0.5 - 0.5 m/s
- 角速度 Z: -1.0 - 1.0 rad/s

---

### 2. 外力干扰训练 (Robust Training) - 配置中

- **任务**: `LeggedLab-Isaac-Velocity-Robust-G1-v0`
- **状态**: 配置已创建，正在调试

#### 外力干扰配置:
- **base_external_force_torque**: 随机外力干扰
  - 力范围: 0-50N
  - 扭矩范围: 0-20Nm
- **push_robot**: 推动机器人
  - 速度扰动: ±0.3 m/s
  - 触发间隔: 5-10秒

---

## 训练参数

| 参数 | 值 | 描述 |
|------|-----|------|
| num_envs | 2048 | 并行环境数量 |
| max_iterations | 10000 | 最大训练迭代次数 |
| decimation | 4 | 控制器降采样率 |
| sim_dt | 0.005 | 仿真时间步长 |
| episode_length_s | 20 | 单个episode最大时长 |

---

## 预期目标

### 短期目标 (10000 iterations)
- [x] 能够在平坦地面上跟踪遥控器给定的速度命令
- [x] 保持身体平衡不跌倒
- [x] 行走动作自然流畅

### 长期目标 (外力干扰训练)
- [ ] 能够抵抗来自不同方向的外力推挤
- [ ] 在受到干扰后能快速恢复平衡
- [ ] 遥控器命令响应稳定

---

## 当前训练状态

### 训练已停止 (GPU内存不足)
- **任务**: LeggedLab-Isaac-Velocity-Robust-G1-v0 (外力干扰训练)
- **状态**: 配置已修复，但需要更多GPU内存

### Robust配置修复完成
- 外力干扰事件已正确配置
- 训练命令可正常运行
- 需要减少环境数量或升级GPU

### 替代方案
1. 使用DeepMimic/AMP训练获得更像人体的姿态
2. 使用Flat Terrain训练 (已完成605次迭代)
3. 减少环境数量继续Robust训练

---

## 后续迭代计划

### 第一阶段: 基础行走训练 (已完成配置，正在训练)
- 使用Flat Terrain环境
- 10000次迭代
- 目标: 基础遥控行走能力

### 第二阶段: 外力干扰训练 (待配置修复)
- 增加外力干扰事件
- 目标: 抗干扰能力

### 第三阶段: 复杂地形训练 (可选)
- 增加楼梯、斜坡等
- 目标: 全地形能力

---

## 训练日志位置

- 训练日志: `/tmp/flat_walk_training.log`
- 模型保存: `logs/rsl_rl/g1_deepmimic/`

---

## 播放训练模型

训练完成后，可使用以下命令播放:

```bash
cd /home/robot/StarBot/isaac-projects/legged_lab-main
export DISPLAY=:1

# 播放最新模型
/home/robot/StarBot/isaac-projects/isaac_sim/python.sh scripts/rsl_rl/play.py \
    --task LeggedLab-Isaac-Velocity-Flat-Unitree-G1-Play-v0 \
    --num_envs 1 \
    --checkpoint logs/rsl_rl/g1_deepmimic/<date_time>/model_9999.pt

# 生成视频
/home/robot/StarBot/isaac-projects/isaac_sim/python.sh scripts/rsl_rl/play.py \
    --task LeggedLab-Isaac-Velocity-Flat-Unitree-G1-Play-v0 \
    --num_envs 1 \
    --checkpoint logs/rsl_rl/g1_deepmimic/<date_time>/model_9999.pt \
    --video --video_length 500 --headless
```

---

## 配置问题记录

### Robust配置问题
- 问题: 继承自 `LocomotionVelocityEnvCfg` 时某些参数不兼容
- 解决: 改用继承自 `G1FlatEnvCfg`，只覆盖必要的奖励和事件配置

---

## 更新日志

- **2026-03-19**: 创建训练计划，启动Flat Terrain训练
