# LeRobot Marvin Force：ManiFeel USB

[中文](README.md) | [英文](README.en.md)

这是 `lerobot_marvin_force` 的 `manifeel_usb` 独立分支，面向带触觉力反馈的 USB 插接仿真消融实验。分支基于 LeRobot `0.6.1`，把 ManiFeel 官方单臂 USB 数据转换为 LeRobot Dataset，并统一支持 Force-conditioned ACT、RDP、ImplicitRDP 与 ForceVLA 的训练和官方仿真成功率评测。

选择 ManiFeel USB 的原因很直接：在已逐文件核验的候选中，它是体积最小且无需构造力代理的数据集。官方 ZIP 为 `3,991,771,935` 字节，包含 50 个完整 episode、5,976 帧，以及右指 `10×14×3` 触觉力场。该任务是单臂，因此本分支建立在 `lerobot_marvin_force`，不是双臂 `lerobot_vlahost_force`。

> 这是仿真专项分支。模型输出 `action[6]` 相对末端位姿增量，不是 Marvin 真机的 `action[8]` 关节位置，也不输出力矩。未经动作空间、状态、相机和触觉模态重新映射，不得把本分支 checkpoint 直接发送给真实 Marvin。

## 可复现来源

| 项目 | 固定版本 |
|---|---|
| ManiFeel 代码 | `purdue-mars/manifeel@ebfa9e1784848903f642ba1b5233f42511a6de62` |
| ManiFeel 仿真 | `purdue-mars/manifeel-isaacgymenvs@1a61da683cf1485f3307684c740771ea5e842b39` |
| 数据仓库 revision | `purdue-mars/manifeel@d2f3bd1fa7eb38ee807d4f3df2c2a3a3821371ea` |
| 官方文件 | `data/usb_quan_Aug05.zip` |
| 文件大小 | `3,991,771,935` 字节 |
| 文件 SHA256 | `25e7912dec28282a2294adc34f819be59a01cebc73ec21501d938dc78f64cb00` |
| 许可证 | MIT |

下载脚本固定到上述 revision，并同时验证长度和 SHA256；不会把分支漂移的 `main` 文件当作同一输入。

## 四个模型如何使用力反馈

| 模型 | 力反馈路径 | 训练方式 | 在线动作 |
|---|---|---|---|
| Force-conditioned ACT | 将 `state[7]` 与 `force[420]` 分别投影后送入 CVAE 和 Transformer | 单阶段 | 每次重观测后执行 1 个 6 维动作 |
| RDP | 慢速视觉 diffusion 规划 latent，快速 decoder 在 chunk 内读取最新力场 | tokenizer、diffusion 两阶段 | 反应式 6 维相对末端动作 |
| ImplicitRDP | 将 RDP 的动作 latent、视觉 diffusion 和力条件 decoder 联合优化 | 单阶段 | 反应式 6 维相对末端动作 |
| ForceVLA | 保留 PI0 状态投影，将 420 维力场单独投影为 force token | PI0 基座微调 | 每次重观测后执行 1 个 6 维动作 |

ForceVLA 的力向量不再塞进 PI0 固定 32 维状态槽；它通过独立线性层生成 force token，因此 420 维 TacFF 不会改变预训练状态投影的形状。

四个模型的公共闭环接口为：

```text
wrist RGB + EEF pose[7] + 当前右指 TacFF[420]
                         │
                         ▼
                       Policy
                         │
                         ▼
 action[6] = 归一化 [Δx, Δy, Δz, axis-angle Δrx, Δry, Δrz]
                         │
                         ▼
          ManiFeel 执行动作，再采集新的 TacFF
```

位置分量在仿真中乘 `0.01 m`，旋转分量乘 `0.05 rad`。USB 夹爪固定闭合到 `0.0145`，所以动作中没有夹爪通道。

## 工作区

```text
workspaces/manifeel_usb/
├── config/
│   ├── environment.yaml
│   ├── data.yaml
│   ├── train_fcact.yaml
│   ├── train_rdp.yaml
│   ├── train_implicitrdp.yaml
│   ├── train_forcevla.yaml
│   ├── train_vision.yaml
│   └── evaluate.yaml
├── scripts/
│   ├── setup_environment.sh
│   ├── download_dataset.py
│   ├── convert_manifeel_to_lerobot.py
│   ├── audit_dataset.py
│   ├── plot_episode_force.py
│   ├── train.py
│   └── serve_lerobot_act.py
├── manifeel_adapter/
├── tests/
└── run.sh
```

所有可配置 YAML 参数都有中文注释。四个阶段都可以仅依赖本分支完成，不需要切换到其他分支。

## 阶段一：环境配置

56 服务器训练路径约定为：

```text
仓库       /ssd/force/repos/lerobot_marvin_force_manifeel_usb
环境       /ssd/force/envs/lerobot_marvin_manifeel_usb
原始数据   /ssd/force/datasets/manifeel_usb/source
转换数据   /ssd/force/datasets/manifeel_usb/lerobot
输出       /ssd/force/outputs/lerobot_marvin_force/manifeel_usb
日志       /ssd/force/logs/manifeel_usb
```

```bash
git clone --branch manifeel_usb --single-branch \
  git@github.com:expolrer/lerobot_marvin_force.git \
  /ssd/force/repos/lerobot_marvin_force_manifeel_usb
cd /ssd/force/repos/lerobot_marvin_force_manifeel_usb

# 安装或校验 LeRobot 训练环境
./run.sh manifeel-usb env
```

安装器支持配置好的 conda-pack 离线归档，也支持显式的 Conda 安装方式。它只安装 LeRobot 训练、Zarr 转换、绘图与 ZMQ 桥接所需依赖。PI0、PaliGemma 和 ResNet 权重必须位于 YAML 指定缓存；脚本会先检查，不会在训练中静默下载不确定版本。

官方成功率环境使用 Python `3.8`、定制 IsaacGym/TacSL；它与 LeRobot `0.6.1` 的 Python `3.12` 环境严格分开。H100 用于离线训练，旧 IsaacGym 相机评测应在兼容的 RTX 4090 主机运行。

## 阶段二：数据下载、转换与审查

### 原始 Zarr 完整字段

| 原始字段 | dtype | shape | 含义 |
|---|---|---:|---|
| `data/state` | `float32` | `[5976,7]` | 世界坐标系 EEF：`x,y,z,qx,qy,qz,qw` |
| `data/action` | `float32` | `[5976,6]` | 归一化相对 EEF 平移与 axis-angle 旋转 |
| `data/tactile_force_field_right` | `float32` | `[5976,10,14,3]` | 右指局部坐标的 `normal,shear_x,shear_y` |
| `data/tactile_depth_right` | `float32` | `[5976,10,14]` | 右指触觉深度 |
| `data/left_tactile_camera_taxim` | `float32` | `[5976,320,240,3]` | 左指触觉 RGB |
| `data/right_tactile_camera_taxim` | `float32` | `[5976,320,240,3]` | 右指触觉 RGB |
| `data/wrist` | `float32` | `[5976,256,256,3]` | 默认腕部 RGB |
| `data/wrist_2` | `float32` | `[5976,256,256,3]` | 第二腕部 RGB |
| `data/front` | `float32` | `[5976,256,256,3]` | 前视 RGB |
| `data/side` | `float32` | `[5976,256,256,3]` | 侧视 RGB |
| `meta/episode_ends` | `int64` | `[50]` | 50 个 episode 的累积结束索引 |

远程 ZIP 的 12,218 个条目和所有理论 Zarr chunk 已逐一核验。episode 长度为 76–168 帧，第一集 103 帧。数据未存时间戳；由物理步长 `0.016667 s` 与 `controlFrequencyInv=4` 得到采样率 `15 FPS`。官方 runner 的 `10 FPS` 只用于视频编码，不能写入数据集元数据。

### LeRobot 最小映射

| LeRobot 字段 | dtype | shape | 转换规则 |
|---|---|---:|---|
| `observation.images.wrist` | `video` | `[3,256,256]` | `float32[0,1]` 转 RGB 视频 |
| `observation.state` | `float32` | `[7]` | 原样保留 EEF 位姿 |
| `observation.tactile_force` | `float32` | `[420]` | 原始 `[10,14,3]` 按 HWC C-order 展平 |
| `action` | `float32` | `[6]` | 原样保留当前观测对应的相对动作 |

展平次序必须保持每个触点的 `normal,shear_x,shear_y` 相邻；不能先转 CHW 再展平。

```bash
# 依次执行固定版本下载、可恢复转换、严格审查和首集曲线
./run.sh manifeel-usb data all

# 也可逐阶段运行
./run.sh manifeel-usb data download
./run.sh manifeel-usb data convert
./run.sh manifeel-usb data audit
```

下载支持断点续传和自动重试；转换只在一集完整 `save_episode/finalize` 后原子更新进度。恢复前会校验源文件 SHA、已完成 episode 前缀、帧数和 schema，避免把两份不同数据拼接到一起。

审查至少检查：50 集、5,976 帧、episode 边界、dtype/shape、NaN/Inf、TacFF C-order、动作配对、视频可解码、源 manifest 和转换后统计。输出在 `/ssd/force/reports/manifeel_usb`。

第一集 normal、两个 shear 分量与净力范数曲线：

![ManiFeel USB 第一个 episode 的右指触觉力曲线](assets/data/manifeel_usb_episode_000_force.png)

## 阶段三：四模型可续训训练

四组训练使用同一数据、划分、随机种子和力字段。56 服务器默认并行分配：

| GPU | 训练 |
|---:|---|
| 0 | Force-conditioned ACT |
| 1 | RDP tokenizer，随后 RDP diffusion |
| 2 | ImplicitRDP |
| 3 | ForceVLA |

```bash
# 先做数据、权重、GPU、命令行 schema 和续训契约预检
./run.sh manifeel-usb train all --dry-run

# 并行启动四组训练（前台汇总输出；服务器上可由 nohup 托管）
./run.sh manifeel-usb train all

# 单独启动或恢复某个模型
./run.sh manifeel-usb train fcact
./run.sh manifeel-usb train rdp
./run.sh manifeel-usb train implicitrdp
./run.sh manifeel-usb train forcevla
```

在正式任务前可运行隔离的真实短训；它会完成 DataLoader、前向、反向与 checkpoint，产物只写入各输出目录的 `smoke/`，不会污染正式续训点：

```bash
./run.sh manifeel-usb train all --smoke-test
# 再将目标从 2 步提高到 3 步，可验证自动从 last checkpoint 续训
./run.sh manifeel-usb train all --smoke-test --smoke-steps 3
```

每次启动都会查找对应输出目录中的：

```text
checkpoints/last/pretrained_model/train_config.json
```

存在时自动使用该配置并追加 `--resume=true`，恢复模型、优化器、调度器、随机状态和训练步数。YAML 的 `steps` 表示目标总步数，不是本次追加步数。若改变数据集、world size、batch、force key 或模型阶段，启动契约会拒绝不安全续训。

RDP 是两阶段模型：tokenizer 未完成时只恢复第一阶段；第一阶段完成且 checkpoint 完整后才进入或恢复 diffusion。W&B 和 Hub 上传默认关闭，避免无凭据时阻塞。

ForceVLA 的 PI0 主干预测 50 步动作且不会在 loss 中屏蔽 episode 边界 padding，因此配置固定 `drop_n_last_frames=49`；训练和留出集 loss 都只使用具有完整未来动作块的 anchor。

视觉消融使用同一数据集和 ACT 参数，仅令 `policy.force_feature_key=null`：

```bash
./run.sh manifeel-usb train vision
```

不要删除数据中的力列来做消融；这样才能保持完全相同的 episode、划分与图像输入。

## 阶段四：官方仿真成功率评测

评测保持 ManiFeel 官方 USB 环境、50 个固定 seed、最多 500 步和原成功判据，只增加跨 Python 环境的本机 ZMQ adapter：

```text
ManiFeel Python 3.8 / IsaacGym             LeRobot Python 3.12
state + wrist + TacFF ── localhost ZMQ ──> processor + policy
6D relative EEF action <────────────────── postprocessor
```

```bash
# LeRobot 环境中启动 checkpoint 服务
./run.sh manifeel-usb eval serve fcact \
  --checkpoint /path/to/checkpoints/last/pretrained_model

# ManiFeel 环境中先运行单环境 smoke test
./run.sh manifeel-usb eval sim fcact --num-envs 1

# 再运行固定 50 环境评测
./run.sh manifeel-usb eval sim fcact --num-envs 50
```

USB 成功条件是四组 plug/socket keypoint 对应欧氏距离的均值严格小于 `0.0079916 m`。官方 wrapper 将成功产生的 `reset=1` 映射为 reward，最终：

```text
success_rate = mean(max reward over time for each environment)
```

评测时固定 `n_action_steps=1`，每执行一个动作就重新读取实时 TacFF。Force 和 vision 消融必须使用相同 `test_start_seed`、环境数与最大步数。

## 引用与边界

- ManiFeel 官方仓库：<https://github.com/purdue-mars/manifeel>
- ManiFeel 官方数据：<https://huggingface.co/datasets/purdue-mars/manifeel>
- 本分支调用 LeRobot 原生 `lerobot-train`；不是把 ManiFeel 的 Diffusion Policy 训练脚本改名。
- 56 服务器 H100 只负责离线训练。旧 IsaacGym/TacSL 的相机闭环评测应在兼容 GPU 上运行并单独记录驱动、CUDA、仿真 commit 和 seed。
