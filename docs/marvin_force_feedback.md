# Marvin Robot: Force Feedback Configuration

## Overview

The Marvin robot now supports optional force/torque sensor data collection during recording and policy inference. This feature allows you to record rich interaction data including joint velocities, joint torques, end-effector forces, and joint external forces.

## Configuration Parameters

### Layer 1: Enable Force Feedback

```bash
--robot.use_force_feedback=true
```

- **Default**: `false` (only record joint positions)
- **When enabled**: Collects additional force/torque sensor data

### Layer 2: Select Force Feedback Types

```bash
--robot.force_feedback_types='["joint_vel", "joint_torque", "cart_force", "joint_force"]'
```

Available options:

| Type | Description | Fields Added | Units |
|------|-------------|--------------|-------|
| `joint_vel` | Joint velocities | `joint_1.vel` ~ `joint_7.vel` | deg/s |
| `joint_torque` | Joint sensor torques | `joint_1.torque` ~ `joint_7.torque` | Nm |
| `cart_force` | End-effector Cartesian forces | `cart_force.fx/fy/fz/mx/my/mz` | N, Nm |
| `joint_force` | Joint space external forces | `joint_1.force` ~ `joint_7.force` | Nm |

**Note**: Only B arm (follower/execution arm) data is recorded. A arm is the teleoperation device.

## Usage Examples

### Example 1: Default (Position Only)

```bash
lerobot-record \
  --robot.type=marvin \
  --robot.ip=192.168.1.190 \
  --robot.cameras="{cam_top: {type: opencv, index_or_path: /dev/cam_top, width: 640, height: 480, fps: 30}}" \
  --dataset.repo_id=your_username/your_dataset \
  --dataset.num_episodes=10
```

**Dataset features**: 7 joint positions + gripper + cameras

---

### Example 2: Position + Joint Velocity

```bash
lerobot-record \
  --robot.type=marvin \
  --robot.ip=192.168.1.190 \
  --robot.use_force_feedback=true \
  --robot.force_feedback_types='["joint_vel"]' \
  --robot.cameras="{cam_top: {type: opencv, index_or_path: /dev/cam_top, width: 640, height: 480, fps: 30}}" \
  --dataset.repo_id=your_username/your_dataset \
  --dataset.num_episodes=10
```

**Dataset features**: 7 joint positions + 7 joint velocities + gripper + cameras

---

### Example 3: Position + End-Effector Forces

```bash
lerobot-record \
  --robot.type=marvin \
  --robot.ip=192.168.1.190 \
  --robot.use_force_feedback=true \
  --robot.force_feedback_types='["cart_force"]' \
  --robot.cameras="{cam_top: {type: opencv, index_or_path: /dev/cam_top, width: 640, height: 480, fps: 30}}" \
  --dataset.repo_id=your_username/your_dataset \
  --dataset.num_episodes=10
```

**Dataset features**: 7 joint positions + 6 Cartesian forces + gripper + cameras

---

### Example 4: All Force Feedback Types

```bash
lerobot-record \
  --robot.type=marvin \
  --robot.ip=192.168.1.190 \
  --robot.use_force_feedback=true \
  --robot.force_feedback_types='["joint_vel", "joint_torque", "cart_force", "joint_force"]' \
  --robot.cameras="{cam_top: {type: opencv, index_or_path: /dev/cam_top, width: 640, height: 480, fps: 30}}" \
  --dataset.repo_id=your_username/your_dataset \
  --dataset.num_episodes=10
```

**Dataset features**: 7 positions + 7 velocities + 7 torques + 7 external forces + 6 Cartesian forces + gripper + cameras (total: 34 motor features)

---

## Dataset Field Naming

All force feedback data is stored in the dataset with clear, distinguishable field names:

### Observation Fields

```
observation.joint_1.pos        # Joint 1 position (always included)
observation.joint_1.vel        # Joint 1 velocity (if joint_vel enabled)
observation.joint_1.torque     # Joint 1 sensor torque (if joint_torque enabled)
observation.joint_1.force      # Joint 1 external force (if joint_force enabled)
...
observation.joint_7.pos
observation.joint_7.vel
observation.joint_7.torque
observation.joint_7.force

observation.cart_force.fx      # End-effector force X (if cart_force enabled)
observation.cart_force.fy      # End-effector force Y
observation.cart_force.fz      # End-effector force Z
observation.cart_force.mx      # End-effector moment X
observation.cart_force.my      # End-effector moment Y
observation.cart_force.mz      # End-effector moment Z

observation.gripper.pos        # Gripper position (if use_gripper=true)
observation.cam_top            # Camera image (if cameras configured)
```

### Action Fields

```
action.joint_1.pos             # Always joint positions only
...
action.joint_7.pos
action.gripper.pos             # If use_gripper=true
```

**Note**: Actions are always position-based. Force feedback data is observation-only.

---

## Policy Inference Mode

When using a policy for inference:

```bash
lerobot-record \
  --robot.type=marvin \
  --robot.use_force_feedback=true \
  --robot.force_feedback_types='["joint_vel", "cart_force"]' \
  --policy.path=your_username/your_policy \
  --dataset.repo_id=your_username/eval_dataset \
  --dataset.num_episodes=5
```

**Behavior**:
- Force feedback data is **collected and stored** in the dataset
- However, force data is **not yet passed to the policy model** (network architecture not finalized)
- The policy still receives only position data for inference

This allows you to record force-rich evaluation datasets while the model architecture is being developed.

---

## Important Notes

### 1. Dataset Compatibility

Datasets recorded with different force feedback configurations will have different `features`:

- A dataset with `use_force_feedback=false` has 7 joint positions
- A dataset with `force_feedback_types=["joint_vel"]` has 7 positions + 7 velocities

**Training requirement**: The policy configuration must match the dataset features.

### 2. Dataset Size

Enabling force feedback increases dataset size:

| Configuration | Motor Features | Approx. Size per Frame |
|--------------|----------------|----------------------|
| Position only | 7 | ~28 bytes |
| + joint_vel | 14 | ~56 bytes |
| + cart_force | 13 | ~52 bytes |
| + All types | 34 | ~136 bytes |

For 1000 episodes × 60s × 30fps = 1.8M frames, enabling all force types adds ~200MB to the dataset (excluding videos).

### 3. Teleoperation vs Policy Mode

- **Teleoperation mode**: Force data is recorded during human demonstrations
- **Policy mode**: Force data is recorded during policy execution (but not used by policy yet)

### 4. Backward Compatibility

The default configuration (`use_force_feedback=false`) maintains full backward compatibility with existing datasets and training scripts.

---

## Verification

To verify your configuration is working:

```python
from lerobot.datasets import LeRobotDataset

# Load your dataset
dataset = LeRobotDataset("your_username/your_dataset")

# Check features
print("Observation features:", dataset.meta.observation_features)
print("Action features:", dataset.meta.action_features)

# Check a sample
sample = dataset[0]
print("Sample keys:", sample.keys())
```

Expected output with `force_feedback_types=["joint_vel", "cart_force"]`:

```
Observation features: {
    'observation.joint_1.pos': float,
    'observation.joint_1.vel': float,
    ...
    'observation.cart_force.fx': float,
    'observation.cart_force.fy': float,
    'observation.cart_force.fz': float,
    'observation.cart_force.mx': float,
    'observation.cart_force.my': float,
    'observation.cart_force.mz': float,
    'observation.cam_top': (480, 640, 3)
}
```

---

## Future Work

- Add support for filtering force feedback in preprocessing pipeline
- Design policy architectures that can leverage force/torque information
- Add visualization tools for force data in Rerun viewer
