# LeRobot Marvin Force · Reactive Diffusion Policy

[Chinese](README.md) | [English](README.en.md)

> Standalone `rdp` branch. Primary workspace: `workspaces/rdp`. Conda environment: `lerobot_marvin_rdp`.
>
> Low-rate visual diffusion performs long-horizon planning while a high-rate force-conditioned decoder emits a position action from the newest force[7].
>
> This branch remains self-contained: environment setup, dataset audit, training, and physical rollout do not depend on another branch.

A force-conditioned imitation-learning repository for the Marvin robot, based on LeRobot `0.6.1`. The `main` branch supports Force-conditioned ACT, Reactive Diffusion Policy (RDP), ImplicitRDP, and ForceVLA. It preserves the existing Marvin position-control rollout path, migrates policy deployment from `lerobot-record` to `lerobot-rollout`, and provides one-command entry points for environment setup, dataset audit, training, and deployment.

> This is an independent, controlled copy of the working content in `/home/marvin/hhw/lerobot_marvin`. It contains none of the old repository's `.git` history, and no commit or push is made to the old repository.

## Models

| Model | Workspace | Conda environment | Training | Rollout inference |
|---|---|---|---|---|
| Force-conditioned ACT | `workspaces/fcact` | `lerobot_marvin_fcact` | Single-stage ACT | `sync` |
| RDP | `workspaces/rdp` | `lerobot_marvin_rdp` | Tokenizer then diffusion | `rdp` |
| ImplicitRDP | `workspaces/irdp` | `lerobot_marvin_implicitrdp` | Single-stage joint optimization | `rdp` |
| ForceVLA | `workspaces/fvla` | `lerobot_marvin_forcevla` | PI0 force-token fine-tuning | `sync` |

The `main` branch contains all four models. The `fcact`, `rdp`, `implicitrdp`, and `forcevla` branches each provide a complete standalone version for one model.

### How force enters each architecture

- Force-conditioned ACT retains ACT's CVAE, Transformer, and action chunks. The current state and a separate seven-dimensional force feature enter both the VAE encoder and the ACT backbone. Training may predict a chunk of 100 actions, while rollout uses `n_action_steps=1` so the next control tick reads force again.
- RDP separates low-rate visual diffusion planning from a force-conditioned tokenizer/decoder that generates a position action from the newest force at every control tick. Training runs tokenizer and diffusion stages in order.
- ImplicitRDP keeps RDP's reactive deployment semantics while jointly optimizing the action latent, force-conditioned decoder, and visual diffusion objective in one stage.
- ForceVLA is a Marvin-adapted PI0 force-token implementation. Three RGB streams, task text, and `state[8]` use the PI0 path, while a seven-dimensional joint-force vector is projected as a separate conditioning token. This is not a line-by-line reproduction of a six-dimensional TCP-wrench model.

All four policies use force as an observation and still output arm-B position commands:

```text
3 RGB streams + state[8] + live force[7] + optional task text
                              │
                              ▼
                            Policy
                              │
                              ▼
       action[8] = 7 arm-B target joint positions + gripper target
```

No policy outputs joint torque. The Marvin position or impedance controller executes the position targets; force only changes the next policy decision.

## Layout

```text
lerobot_marvin_force/
├── config/deploy.yaml
├── scripts/
├── workspaces/{fcact,rdp,irdp,fvla}/
├── src/lerobot/policies/
├── src/lerobot/robots/marvin/
├── src/lerobot/rollout/
├── assets/data/
├── artifacts/data_audit/
└── run.sh
```

Every workspace is self-contained:

```text
workspaces/<model>/
├── config/{environment,data,train,deploy}.yaml
├── scripts/{setup_environment.sh,audit_dataset.py,train.py,deploy.py}
└── run.sh
```

Every YAML parameter has a Chinese inline explanation for operators on the target machine. Validate this contract with `python scripts/check_yaml_comments.py`.

## Stage 1: environment

Requirements are Ubuntu, Conda or Miniconda, a compatible NVIDIA driver/CUDA stack, access to the Marvin control network and SDK libraries, and the three cameras. Initial ForceVLA setup also needs access to the PI0 base weights.

```bash
cd /home/marvin/hhw/lerobot_marvin_force
chmod +x run.sh workspaces/*/run.sh workspaces/*/scripts/setup_environment.sh

# Interactive selection: 1, 2, 3, or 4
./run.sh env

# Direct selection
./run.sh env fcact
./run.sh env rdp
./run.sh env implicitrdp
./run.sh env forcevla

# Preview without installing
./run.sh env fcact --dry-run
```

The choices are `1` Force-conditioned ACT, `2` RDP, `3` ImplicitRDP, and `4` ForceVLA. Each installer creates the exact Python `3.12` Conda environment declared in `environment.yaml` and performs an editable install of this repository. Reinstallation is required after copying so the new `lerobot-rollout` entry point is generated.

## Stage 2: dataset audit

Default dataset:

```text
/home/marvin/hhw/peg_optical_module_0726force
```

It is a LeRobot Dataset `v3.0` with 149 episodes, 161,728 frames at 30 FPS, and approximately 89.85 minutes of data. “First episode” is the correct dataset term; an epoch is a training concept. The learned interface controls Marvin arm B, not a bimanual 16-dimensional action.

### Complete feature table

| Feature | dtype | shape | Source or meaning |
|---|---|---:|---|
| `observation.images.up_cam` | `video` | `[480,640,3]` | Upper RGB, AV1, 30 FPS |
| `observation.images.left_close` | `video` | `[480,640,3]` | Left close-view RGB, AV1, 30 FPS |
| `observation.images.right_close` | `video` | `[480,640,3]` | Right close-view RGB, AV1, 30 FPS |
| `observation.state` | `float32` | `[8]` | Seven arm-B joint positions and gripper position |
| `observation.joint_vel` | `float32` | `[7]` | `m_FB_Joint_Vel` |
| `observation.joint_torque` | `float32` | `[7]` | `m_FB_Joint_SToq`, controller feedback joint torque |
| `observation.joint_force` | `float32` | `[7]` | `m_EST_Joint_Force`, estimated external joint force |
| `observation.cart_force` | `float32` | `[6]` | `m_EST_Cart_FN`; entirely zero and invalid as a condition |
| `action` | `float32` | `[8]` | Seven arm-B target joint positions and gripper target |
| `timestamp` | `float32` | `[1]` | Time within the episode |
| `frame_index` | `int64` | `[1]` | Frame within the episode |
| `episode_index` | `int64` | `[1]` | Episode index |
| `index` | `int64` | `[1]` | Global frame index |
| `task_index` | `int64` | `[1]` | Task index |

State and action channel order is `joint_1 ... joint_7, gripper`. Both joint-force features use `joint_1 ... joint_7`.

### Torque and estimated external force are different signals

| Feature | SDK variable | Default input | Meaning |
|---|---|---:|---|
| `observation.joint_torque` | `m_FB_Joint_SToq` | Optional | Controller feedback torque, including gravity, payload, drive, and control components. The interface alone does not prove an independent torque-sensor source. |
| `observation.joint_force` | `m_EST_Joint_Force` | Yes | SDK-estimated external joint disturbance and the default conditioning signal. |
| `observation.cart_force` | `m_EST_Cart_FN` | No | Nominal Cartesian force/moment, but all values are zero in this dataset. |

To train on `m_FB_Joint_SToq`, change `policy.force_feature_key` to `observation.joint_torque` in both training and deployment YAML. Rollout can expose both signals, but the selected key, order, unit, sign, and offset must match training exactly.

A different robot dataset is not compatible merely because it contains `float32[7]`; cameras, state/action definitions, joint order, rate, units, calibration, and live SDK semantics must also match.

### Cross-episode consistency

The force sequences are not identical. Their episode-level baselines are close while insertion contact dynamics remain present. Ranges of episode means for `observation.joint_force` are:

```text
J1  0.150 to  0.514 Nm
J2 -0.943 to -0.497 Nm
J3  0.329 to  0.637 Nm
J4 -0.266 to  0.083 Nm
J5  0.038 to  0.245 Nm
J6 -0.317 to -0.040 Nm
J7 -0.282 to -0.092 Nm
```

The standard deviations of those episode means are `[0.0785, 0.0850, 0.0689, 0.0768, 0.0455, 0.0778, 0.0375] Nm`. Training and deployment should therefore keep payload, tool parameters, force estimation, zero point, and control mode consistent.

```bash
# Validate without writing artifacts
./run.sh data fcact --dry-run

# Generate reports, per-episode statistics, CSV files, and the homepage plot
./run.sh data fcact
```

Artifacts are written to:

```text
artifacts/data_audit/<model>/dataset_report.json
artifacts/data_audit/<model>/feature_table.md
artifacts/data_audit/<model>/episode_force_summary.csv
artifacts/data_audit/<model>/episode_000_force.csv
assets/data/peg_optical_module_0726force_episode_000_force.png
```

Episode 0 contains 1,141 frames and lasts about 38 seconds. The upper plot is `m_FB_Joint_SToq`; the lower plot is `m_EST_Joint_Force`.

![Feedback joint torque and estimated external joint force in the first episode](assets/data/peg_optical_module_0726force_episode_000_force.png)

## Stage 3: training

The wrapper validates YAML and the dataset contract, then invokes the stock LeRobot `0.6.1` `lerobot-train` command. ACT uses the stock `EpisodeAwareSampler`; no keyframe labels, weights, or custom keyframe sampler remain.

```bash
# Preview
./run.sh train fcact --dry-run
./run.sh train rdp --dry-run
./run.sh train implicitrdp --dry-run
./run.sh train forcevla --dry-run

# Train
./run.sh train fcact
./run.sh train rdp
./run.sh train implicitrdp
./run.sh train forcevla
```

Parameters live in `workspaces/<model>/config/train.yaml`. RDP first trains the tokenizer and then loads `outputs/train/rdp_tokenizer_peg_0726/checkpoints/last/pretrained_model` for diffusion training. `policy.push_to_hub` defaults to `false`; W&B defaults to enabled. Check paths, IDs, W&B authentication, GPU memory, and the selected force key first.

## Stage 4: deployment

Select the model and checkpoint in `config/deploy.yaml`:

```yaml
# Select fcact, rdp, implicitrdp, or forcevla.
model_name: fcact

# A pretrained_model directory or Hugging Face model ID.
weight_path: /home/marvin/hhw/lerobot_marvin_force/outputs/train/fcact_peg_0726/checkpoints/last/pretrained_model
```

Robot, camera, episodic-recording, and inference settings live in the selected workspace's `config/deploy.yaml`.

```bash
# Validate the environment, weights, force key, cameras, and inference type
./run.sh deploy --dry-run

# Connect to the physical Marvin only after manual review
./run.sh deploy
```

### Migration from `lerobot-record`

| Old concept | New configuration |
|---|---|
| Policy loaded by `lerobot-record` | `lerobot-rollout` |
| `--policy.path=...` | Root `config/deploy.yaml` `weight_path` |
| Fixed evaluation episodes | `strategy.type: episodic` |
| ACT / ForceVLA | `inference.type: sync` |
| RDP / ImplicitRDP | `inference.type: rdp` |
| `--dataset.episode_time_s=600` | `dataset.episode_time_s: 600` |
| `--dataset.num_episodes=20` | `dataset.num_episodes: 20` |
| `--dataset.single_task=...` | `runtime.task` and `dataset.single_task` |
| `--resume=true` | `runtime.resume: true`, only for an existing compatible dataset |

New rollout dataset names must start with `rollout_` and receive a timestamp. Defaults are `resume=false` and `dataset.root=null` to prevent accidental writes into old data. To resume, provide the exact stamped repo ID and root and set `resume=true`.

The old `/dev/cam_up`, `/dev/cam_down`, and `/dev/cam_top` devices are exposed directly under the training keys `up_cam`, `left_close`, and `right_close`. Verify the physical close-camera views before rollout.

`robot.use_arm=B` is the safe default because the policy only emits arm-B `action[8]`. Setting `AB` also configures arm A in the SDK, but it does not make the policy bimanual.

### Physical safety

- Always run `./run.sh deploy --dry-run` first.
- Verify the IP, impedance mode, `tool_mass_b=1.04`, center of mass/tool offset, and live-force unit and sign.
- Shorten the first episode, reduce speed, keep the emergency stop available, and maintain human supervision.
- The default 2-degree per-tick joint step and per-joint force envelopes are task-derived safeguards, not certified safety limits; recalibrate them after any payload change.
- Automatic return to the startup pose is disabled by default to avoid unreviewed motion.
- Stop immediately on NaN/Inf, camera loss, abnormal contact, or a force-limit violation.

## ForceVLA GPU note

The PI0 backbone is substantially larger than the other policies. Defaults use `batch_size=1`, `bfloat16`, and gradient checkpointing, but do not guarantee training or real-time 30 Hz inference on an 8 GB GPU. Validate with a short run first. At least 24 GB is recommended for initial experiments; 48 GB or more is better suited to full training.

## Common commands

```bash
./run.sh --help
./run.sh env
./run.sh data fcact
./run.sh train fcact --dry-run
./run.sh train fcact
./run.sh deploy --dry-run
./run.sh deploy
```
