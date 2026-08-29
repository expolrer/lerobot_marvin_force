# LeRobot Marvin Force

[中文](README.md) | [English](README.en.md)

面向 Marvin 机械臂的力反馈模仿学习仓库，基于 LeRobot `0.6.1`，主分支完整支持 Force-conditioned ACT、Reactive Diffusion Policy（RDP）、ImplicitRDP 和 ForceVLA。仓库保留现有 Marvin 实际控制链路，把旧版 `lerobot-record + policy` 部署迁移到 `lerobot-rollout`，并统一提供环境、数据审查、训练、部署四个阶段的一键入口。

> 本仓库从 `/home/marvin/hhw/lerobot_marvin` 的工作内容受控复制后独立创建，不包含旧仓库的 `.git`，不会在旧仓库中提交或推送任何内容。

## 支持的模型

| 模型 | 工作区 | Conda 环境 | 训练 | rollout 推理 |
|---|---|---|---|---|
| Force-conditioned ACT | `workspaces/fcact` | `lerobot_marvin_fcact` | 单阶段 ACT | `sync` |
| RDP | `workspaces/rdp` | `lerobot_marvin_rdp` | tokenizer + diffusion 两阶段 | `rdp` |
| ImplicitRDP | `workspaces/irdp` | `lerobot_marvin_implicitrdp` | 单阶段联合优化 | `rdp` |
| ForceVLA | `workspaces/fvla` | `lerobot_marvin_forcevla` | PI0 基座力 token 微调 | `sync` |

`main` 同时包含四种模型；`fcact`、`rdp`、`implicitrdp`、`forcevla` 四个分支分别提供对应模型的独立完整版本。

### 四种模型的力反馈结构

- Force-conditioned ACT：保留 ACT 的 CVAE、Transformer 和动作 chunk，将当前 `state` 与独立七维力特征同时送入 VAE 编码器和 ACT 主干。训练可预测长度为 100 的 chunk，但部署固定 `n_action_steps=1`，每个控制周期都会重新读取力反馈。
- RDP：低频视觉 diffusion 负责长时规划，tokenizer/decoder 在每个控制周期结合最新力反馈解码位置动作。训练依次执行 tokenizer 和 diffusion 两阶段。
- ImplicitRDP：沿用 RDP 的反应式部署语义，以隐式/联合目标同时训练动作 latent、力条件 decoder 和视觉 diffusion，不需要单独准备 tokenizer checkpoint。
- ForceVLA：本仓库实现的是 Marvin 适配的 PI0 力 token 版本。三路 RGB、任务文本和 `state[8]` 进入 PI0 路径，七维关节力经独立投影成为条件 token。它不是基于六维 TCP wrench 的论文逐行复现。

四种模型均只把力作为观测，输出仍为 B 臂位置动作：

```text
三路 RGB + state[8] + 实时 force[7] + 可选任务文本
                         │
                         ▼
                       Policy
                         │
                         ▼
       action[8] = B 臂 7 个目标关节位置 + 夹爪目标位置
```

Policy 不输出关节力矩。真实运动继续由 Marvin 的位置/阻抗控制器执行，力反馈只用于调整下一控制周期的位置目标。

## 仓库结构

```text
lerobot_marvin_force/
├── config/deploy.yaml                 # 主入口选择模型和权重
├── scripts/                           # 主入口与配置检查
├── workspaces/
│   ├── fcact/                         # 四阶段独立工作区
│   ├── rdp/
│   ├── irdp/
│   └── fvla/
├── src/lerobot/policies/              # 四种策略
├── src/lerobot/robots/marvin/         # Marvin SDK 和位置 rollout
├── src/lerobot/rollout/               # sync/RDP 推理和 episodic 录制
├── assets/data/                       # 首页曲线
├── artifacts/data_audit/              # 可复现的数据审查结果
└── run.sh                             # 统一一键入口
```

每个工作区均可独立完成四阶段：

```text
workspaces/<缩写>/
├── config/
│   ├── environment.yaml
│   ├── data.yaml
│   ├── train.yaml
│   └── deploy.yaml
├── scripts/
│   ├── setup_environment.sh
│   ├── audit_dataset.py
│   ├── train.py
│   └── deploy.py
└── run.sh
```

所有 YAML 参数前都有中文注释；可用 `python scripts/check_yaml_comments.py` 自动检查。

## 阶段一：环境配置

前置条件：Ubuntu、Conda/Miniconda、NVIDIA 驱动与合适的 CUDA、Marvin 控制网段、SDK 动态库和三路相机。ForceVLA 首次初始化还需要可访问 PI0 基座权重。

```bash
cd /home/marvin/hhw/lerobot_marvin_force
chmod +x run.sh workspaces/*/run.sh workspaces/*/scripts/setup_environment.sh

# 交互式输入 1/2/3/4
./run.sh env

# 或直接选择模型
./run.sh env fcact
./run.sh env rdp
./run.sh env implicitrdp
./run.sh env forcevla

# 只预览安装操作
./run.sh env fcact --dry-run
```

选择对应关系：

```text
1  Force-conditioned ACT
2  RDP
3  ImplicitRDP
4  ForceVLA
```

脚本使用对应 `environment.yaml` 创建 Python `3.12` Conda 环境并对本仓库执行 editable install。复制时虽然源码已是 `0.6.1`，仍必须重新安装，才能生成新版 `lerobot-rollout` 入口。

## 阶段二：数据审查

默认数据集：

```text
/home/marvin/hhw/peg_optical_module_0726force
```

该目录是 LeRobot Dataset `v3.0`，包含 149 个 episode、161,728 帧、30 FPS，约 89.85 分钟。这里应称“第一个 episode”，而不是训练中的“第一个 epoch”。Policy 与数据均为 Marvin B 单臂契约，即使某些硬件配置允许 `AB`，学习动作仍不是双臂 16 维。

### 完整字段表

| 字段 | dtype | shape | 来源或含义 |
|---|---|---:|---|
| `observation.images.up_cam` | `video` | `[480,640,3]` | 顶部 RGB，AV1，30 FPS |
| `observation.images.left_close` | `video` | `[480,640,3]` | 左侧近景 RGB，AV1，30 FPS |
| `observation.images.right_close` | `video` | `[480,640,3]` | 右侧近景 RGB，AV1，30 FPS |
| `observation.state` | `float32` | `[8]` | B 臂七关节当前位置 + 夹爪当前位置 |
| `observation.joint_vel` | `float32` | `[7]` | `m_FB_Joint_Vel` |
| `observation.joint_torque` | `float32` | `[7]` | `m_FB_Joint_SToq`，控制器反馈关节扭矩 |
| `observation.joint_force` | `float32` | `[7]` | `m_EST_Joint_Force`，SDK 估算关节外力 |
| `observation.cart_force` | `float32` | `[6]` | `m_EST_Cart_FN`；当前全部为零，禁止作为有效条件 |
| `action` | `float32` | `[8]` | B 臂七关节目标位置 + 夹爪目标位置 |
| `timestamp` | `float32` | `[1]` | episode 内时间戳 |
| `frame_index` | `int64` | `[1]` | episode 内帧索引 |
| `episode_index` | `int64` | `[1]` | episode 索引 |
| `index` | `int64` | `[1]` | 全数据集帧索引 |
| `task_index` | `int64` | `[1]` | 任务索引 |

状态和动作顺序固定为 `joint_1 ... joint_7, gripper`；两种关节力字段顺序固定为 `joint_1 ... joint_7`。

### `joint_torque` 与 `joint_force` 不能混称

| 字段 | SDK 变量 | 默认模型输入 | 说明 |
|---|---|---:|---|
| `observation.joint_torque` | `m_FB_Joint_SToq` | 可选 | 实际控制器反馈扭矩，包含重力、负载、驱动和控制项；仅凭接口不能断定其底层一定来自独立扭矩传感器 |
| `observation.joint_force` | `m_EST_Joint_Force` | 是 | SDK 估算的外部关节扰动力，现有四模型默认使用 |
| `observation.cart_force` | `m_EST_Cart_FN` | 否 | 理论上是三维力与三维力矩，但 0726 数据全部为零 |

若要改用 `m_FB_Joint_SToq`，必须同时把训练 YAML 与部署 YAML 中的 `policy.force_feature_key` 改为 `observation.joint_torque`。rollout 已支持读取两种字段，但训练与实时推理必须使用同一 key、关节顺序、单位、符号和零点。

仅有 `float32[7]` 不代表其他机器人数据可以直接训练；还要一致地满足相机、状态、动作、关节顺序、采样频率、单位、标定与实时 SDK 语义。

### 跨 episode 一致性

149 个 episode 的 `observation.joint_force` 并非逐帧相同，但 episode 均值处在相近区间，并保留接触过程动态。七个通道的 episode 均值范围为：

```text
J1  0.150 ～  0.514 Nm
J2 -0.943 ～ -0.497 Nm
J3  0.329 ～  0.637 Nm
J4 -0.266 ～  0.083 Nm
J5  0.038 ～  0.245 Nm
J6 -0.317 ～ -0.040 Nm
J7 -0.282 ～ -0.092 Nm
```

episode 均值标准差为：

```text
[0.0785, 0.0850, 0.0689, 0.0768, 0.0455, 0.0778, 0.0375] Nm
```

结论是整体基线较稳定但不完全一致。部署时仍必须保持相同的负载、工具参数、力估算、零点和控制模式。

### 执行审查并生成曲线

```bash
# 只验证，不写报告
./run.sh data fcact --dry-run

# 生成报告、逐 episode 统计、CSV 与首页 PNG
./run.sh data fcact
```

输出：

```text
artifacts/data_audit/<模型>/dataset_report.json
artifacts/data_audit/<模型>/feature_table.md
artifacts/data_audit/<模型>/episode_force_summary.csv
artifacts/data_audit/<模型>/episode_000_force.csv
assets/data/peg_optical_module_0726force_episode_000_force.png
```

episode 0 包含 1,141 帧、约 38 秒。上图是 `m_FB_Joint_SToq`，下图是 `m_EST_Joint_Force`：

![peg_optical_module_0726force 第一个 episode 的反馈关节扭矩和估算外力](assets/data/peg_optical_module_0726force_episode_000_force.png)

## 阶段三：训练

训练包装只验证 YAML 与数据契约，然后直接调用 LeRobot `0.6.1` 的 `lerobot-train`。ACT 已恢复官方 `EpisodeAwareSampler`，不存在关键帧标签、关键帧权重或自定义关键帧 sampler。

```bash
# 先检查将执行的命令
./run.sh train fcact --dry-run
./run.sh train rdp --dry-run
./run.sh train implicitrdp --dry-run
./run.sh train forcevla --dry-run

# 实际训练
./run.sh train fcact
./run.sh train rdp
./run.sh train implicitrdp
./run.sh train forcevla
```

参数分别位于 `workspaces/<模型缩写>/config/train.yaml`。RDP 包装会先训练 tokenizer，再从 `outputs/train/rdp_tokenizer_peg_0726/checkpoints/last/pretrained_model` 启动 diffusion 阶段。

默认 `policy.push_to_hub=false`；W&B 默认启用。开始训练前检查数据路径、模型 ID、W&B 登录、GPU 显存以及选定的 `force_feature_key`。

## 阶段四：部署

先编辑主选择文件 `config/deploy.yaml`：

```yaml
# 选择 fcact、rdp、implicitrdp 或 forcevla。
model_name: fcact

# 训练输出的 pretrained_model 目录或 Hugging Face 模型 ID。
weight_path: /home/marvin/hhw/lerobot_marvin_force/outputs/train/fcact_peg_0726/checkpoints/last/pretrained_model
```

机器人、相机、episodic 数据录制和推理参数在对应 `workspaces/<模型>/config/deploy.yaml`。

```bash
# 只进行环境、权重、力字段、相机和推理类型预检并打印命令
./run.sh deploy --dry-run

# 经人工核对后连接真实 Marvin
./run.sh deploy
```

### 从旧 `lerobot-record` 迁移

| 旧设置 | 新设置 |
|---|---|
| `lerobot-record` 加载 policy | `lerobot-rollout` |
| `--policy.path=...` | 根 `config/deploy.yaml` 的 `weight_path` |
| 固定数量评估 episode | `strategy.type: episodic` |
| ACT / ForceVLA | `inference.type: sync` |
| RDP / ImplicitRDP | `inference.type: rdp` |
| `--dataset.episode_time_s=600` | `dataset.episode_time_s: 600` |
| `--dataset.num_episodes=20` | `dataset.num_episodes: 20` |
| `--dataset.single_task=...` | `runtime.task` 与 `dataset.single_task` |
| `--resume=true` | `runtime.resume: true`，仅用于已存在且 schema 一致的数据集 |

新版 rollout 新建数据集名称必须以 `rollout_` 开头，并自动附加时间戳。默认 `resume=false`、`dataset.root=null`，避免误写旧数据；恢复已有数据时应同时填写该数据集的确切带时间戳 repo ID、root，并改为 `resume=true`。

旧设备 `/dev/cam_up`、`/dev/cam_down`、`/dev/cam_top` 在配置中直接使用训练字段名 `up_cam`、`left_close`、`right_close`。首次上机必须验证两个近景相机的物理视角确实对应，不能只按名字推断。

当前默认 `robot.use_arm=B`，因为 policy 始终只输出 B 臂 `action[8]`。若改为 `AB`，SDK 会额外配置 A 臂，但不会让模型变成双臂 policy。

### 上机安全

- 必须先执行 `./run.sh deploy --dry-run`。
- 核对 IP、阻抗模式、`tool_mass_b=1.04`、工具重心/偏移和实时力的单位/符号。
- 首次测试缩短 episode、降低速度，并保持急停和人工监控。
- 默认限制每周期关节步长为 2°，并配置了基于 0726 数据的逐关节外力阈值；这些值不是安全认证，负载变化后必须重新标定。
- 默认不在 episode 间或退出时自动返回启动姿态，避免未经确认的自动运动。
- 出现 NaN/Inf、相机断流、异常接触或力越界时立即停止。

## ForceVLA 显存限制

ForceVLA 的 PI0 主干明显大于另外三种模型。默认采用 `batch_size=1`、`bfloat16` 和 gradient checkpointing，但不保证 8 GB 显存可完成训练或实时 30 Hz 推理。建议先用小步数验证；初步实验建议至少 24 GB，完整训练更适合 48 GB 或更大显存。

## 常用命令

```bash
./run.sh --help
./run.sh env                         # 交互选择 1/2/3/4
./run.sh data fcact
./run.sh train fcact --dry-run
./run.sh train fcact
./run.sh deploy --dry-run
./run.sh deploy
```
