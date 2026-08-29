# Marvin 力反馈策略：数据、训练与部署

本文档固定三条实现路线，并让它们共享同一个硬件契约：

```text
输入  = 三路 RGB + observation.state[8] + observation.joint_force[7]
输出  = action[8] = B 臂 7 个绝对关节位置（度）+ 夹爪位置
执行  = Marvin impedance 模式下的 set_joint_cmd_pose
```

力是策略的实时观测，不是动作标签，也不直接发送关节力矩。全零的
`observation.cart_force[6]` 不进入任何模型。

## 1. 先生成干净数据副本

原始数据共有 149 episodes / 161728 frames / 30 Hz。审计发现 12 个 episode
首帧含夹爪初始化哨兵值 `-12.467574`。脚本永不改原目录；视频优先使用硬链接，
仅复制和修复 parquet，并重算数值统计量。

```bash
cd /home/marvin/hhw/lerobot_marvin

# 只审计
conda run -n lerobot_marvin python \
  src/lerobot/scripts/lerobot_sanitize_marvin_force_dataset.py \
  --source-root=/home/marvin/hhw/peg_optical_module_0726force

# 创建训练副本（目标已存在时会拒绝覆盖）
conda run -n lerobot_marvin python \
  src/lerobot/scripts/lerobot_sanitize_marvin_force_dataset.py \
  --source-root=/home/marvin/hhw/peg_optical_module_0726force \
  --target-root=/home/marvin/hhw/peg_optical_module_0726force_clean
```

`joint_force[7]` 按通道做 mean/std 归一化。第 6 个通道（索引 5）有明显随
episode 漂移，部署前必须在空载、同一工具和同一控制模式下检查零偏。

## 2. Force-ACT（首选基线）

实现把标准化后的 `state[8]` 与 `joint_force[7]` 拼成 15D proprioception。
训练期 VAE encoder 和主 Transformer encoder 的两处 state projection 都接收
15D；action head 仍输出 8D 关节位置。部署设 `n_action_steps=1`，每个 30 Hz
控制周期重新读取力。

```bash
conda run -n lerobot_marvin lerobot-train \
  --dataset.repo_id=hukewei/peg_optical_module_0726force \
  --dataset.root=/home/marvin/hhw/peg_optical_module_0726force_clean \
  --dataset.eval_split=0.1 \
  --policy.type=act \
  --policy.force_feature_key=observation.joint_force \
  --policy.chunk_size=100 \
  --policy.n_action_steps=1 \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.use_amp=true \
  --batch_size=2 \
  --steps=100000 \
  --output_dir=outputs/train/act_joint_force_0726 \
  --job_name=act_joint_force_0726 \
  --policy.repo_id=hukewei/act_joint_force_0726 \
  --wandb.enable=true
```

## 3. Reactive Diffusion Policy（RDP）

RDP 是两阶段模型：

1. tokenizer：`action[32,8] -> latent[8,64]`，单向 GRU 用逐时刻
   `joint_force[32,7]` 重建动作；
2. diffusion：两帧慢速 RGB/state/force 条件生成 latent。部署时慢规划器约
   6 Hz 异步更新 latent，小 GRU 在每个 30 Hz tick 用最新力重新解码位置动作。

Stage 1：

```bash
conda run -n lerobot_marvin lerobot-train \
  --dataset.repo_id=hukewei/peg_optical_module_0726force \
  --dataset.root=/home/marvin/hhw/peg_optical_module_0726force_clean \
  --dataset.eval_split=0.1 \
  --policy.type=rdp \
  --policy.training_stage=tokenizer \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.use_amp=true \
  --batch_size=16 \
  --steps=100000 \
  --output_dir=outputs/train/rdp_tokenizer_0726 \
  --job_name=rdp_tokenizer_0726 \
  --policy.repo_id=hukewei/rdp_tokenizer_0726 \
  --wandb.enable=true
```

Stage 2（加载 Stage 1 的完整 checkpoint；前 500 batch 自动校准 latent 统计量）：

```bash
conda run -n lerobot_marvin lerobot-train \
  --dataset.repo_id=hukewei/peg_optical_module_0726force \
  --dataset.root=/home/marvin/hhw/peg_optical_module_0726force_clean \
  --dataset.eval_split=0.1 \
  --policy.path=outputs/train/rdp_tokenizer_0726/checkpoints/last/pretrained_model \
  --policy.training_stage=diffusion \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.use_amp=true \
  --batch_size=2 \
  --steps=100000 \
  --output_dir=outputs/train/rdp_diffusion_0726 \
  --job_name=rdp_diffusion_0726 \
  --policy.repo_id=hukewei/rdp_joint_force_0726 \
  --wandb.enable=true
```

未完成 latent 校准的 checkpoint 会拒绝部署。RDP 使用 feature-specific
timestamps，避免把三路 480x640 图像各解码 32 帧。

## 4. ForceVLA-JointForce

这是 ForceVLA 的 Marvin 变体，不是论文中原始 6D TCP wrench 配置：

```text
state32 = state8 || joint_force7 || zeros17
三路 RGB 全部启用
Pi0 + force token + LIMoE -> 50 x action32 -> 裁成 50 x action8
```

官方显存要求为 inference >8 GB、LoRA >22.5 GB、full >70 GB；Marvin 主机的
RTX 5060 Laptop 只有 8 GB，不能可靠训练或本地推理。因此使用：

```text
外部 >=24 GB GPU：/home/marvin/hhw/ForceVLA_marvin（训练 + WebSocket server）
Marvin 主机：本仓库薄客户端（采 RGB/state/force，执行 8D position action）
```

外部 GPU（48 GB 更稳）：

```bash
cd /home/marvin/hhw/ForceVLA_marvin
uv sync
uv run scripts/compute_norm_stats.py --config-name forcevla_marvin_lora
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
  forcevla_marvin_lora --exp-name=peg_0726_joint_force --overwrite

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config=forcevla_marvin_lora \
  --policy.dir=checkpoints/forcevla_marvin_lora/peg_0726_joint_force/50000 \
  --port=8000 \
  --default-prompt='Insert the optical peg module'
```

Marvin 客户端只额外需要轻量协议依赖：

```bash
conda run -n lerobot_marvin pip install websockets msgpack
conda run -n lerobot_marvin python examples/forcevla/deploy_marvin.py \
  --server_host=<外部GPU的TAILSCALE_IP> \
  --server_port=8000 \
  --duration_s=10 \
  --dry_run=true \
  --robot.type=marvin \
  --robot.use_arm=B \
  --robot.control_mode=impedance \
  --robot.use_gripper=true \
  --robot.use_force_feedback=true \
  --robot.force_feedback_types='["joint_force"]' \
  --robot.cameras="$MARVIN_CAMERAS"
```

先确认 10 秒 dry-run 的输入形状、延迟、force age 和动作范围，再显式改为
`--dry_run=false`。客户端每 3 步异步重规划，网络超时、坏 shape、NaN、过期
response 或超力时保持当前姿态并清空旧 chunk。

## 5. ACT / RDP 的 Marvin 部署

`$MARVIN_CAMERAS` 必须使用采集时完全相同的三个 key：
`up_cam`、`left_close`、`right_close`。设备路径由现场相机映射填写。

Force-ACT：

```bash
conda run -n lerobot_marvin python -m lerobot.scripts.lerobot_rollout \
  --strategy.type=base \
  --inference.type=sync \
  --policy.path=outputs/train/act_joint_force_0726/checkpoints/last/pretrained_model \
  --robot.type=marvin \
  --robot.use_arm=B \
  --robot.control_mode=impedance \
  --robot.use_gripper=true \
  --robot.use_force_feedback=true \
  --robot.force_feedback_types='["joint_force"]' \
  --robot.cameras="$MARVIN_CAMERAS" \
  --fps=30 --duration=10 --task='Insert the optical peg module'
```

RDP 只把 inference backend 改为异步快慢环：

```bash
conda run -n lerobot_marvin python -m lerobot.scripts.lerobot_rollout \
  --strategy.type=base \
  --inference.type=rdp \
  --policy.path=outputs/train/rdp_diffusion_0726/checkpoints/last/pretrained_model \
  --robot.type=marvin \
  --robot.use_arm=B \
  --robot.control_mode=impedance \
  --robot.use_gripper=true \
  --robot.use_force_feedback=true \
  --robot.force_feedback_types='["joint_force"]' \
  --robot.cameras="$MARVIN_CAMERAS" \
  --fps=30 --duration=10 --task='Insert the optical peg module'
```

共同硬件保护：非有限动作拒绝、关节单 tick 最大变化 2 度、夹爪范围
0.3–1.4、按 0726 数据范围设置的逐关节 force envelope，以及夹爪约 6 Hz
限频。阈值不是安全认证值；更换末端工具或负载后必须重标定。

## 6. 建议验收顺序

1. 离线跑测试和 checkpoint 单 batch forward；
2. RGB+state ACT 与 RGB+state+force ACT 做严格消融；
3. force 置零、跨 episode 打乱，确认模型确实使用力；
4. 实机只连接观测做 10 秒检查；
5. 低速、空载、限位外有人急停，执行 10 秒；
6. 记录 fast-loop P99、slow inference P95、force age、超力次数和成功率。

力反馈策略仍是学习到的位置控制器，不能替代机器人底层实时安全回路。
