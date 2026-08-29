# Marvin Robot Data Collection Flow

## Overview

This document describes the complete data flow for Marvin robot data collection, including all SDK operations involved.

---

## 1. Initialization Phase

### 1.1 Robot Connection (`robot.connect()`)

**File**: [marvin_robot.py:112-157](src/lerobot/robots/marvin/marvin_robot.py#L112-L157)

```python
def connect(self, calibrate: bool = True, skip_configure: bool = False) -> None:
    self._sdk, self._sdk_lock, already_connected = get_sdk(self.config.ip)
    
    if not already_connected:
        with self._sdk_lock:
            # SDK Operation 1: Connect to robot controller
            self._sdk.connect(self.config.ip)
            
            # SDK Operation 2: Disable SDK logs
            self._sdk.log_switch('0')
            self._sdk.local_log_switch('0')
            
            # SDK Operation 3: Clear errors
            self._sdk.clear_error("A")
            self._sdk.clear_error("B")
            
            # SDK Operation 4: Verify connection (read frames)
            self._verify_connection()  # Calls subscribe() 10 times
    
    # SDK Operation 5: Configure arms (if not skip_configure)
    if not already_connected and not skip_configure:
        with self._sdk_lock:
            for arm in self._arms:
                self._configure_arm(arm)
```

**SDK calls during connection**:
1. `self._sdk.connect(ip)` — Establish UDP connection
2. `self._sdk.log_switch('0')` — Disable remote logging
3. `self._sdk.local_log_switch('0')` — Disable local logging
4. `self._sdk.clear_error("A/B")` — Clear error flags
5. `self._sdk.subscribe()` × 10 — Verify frame updates (connection check)

---

### 1.2 Arm Configuration (`_configure_arm()`)

**File**: [marvin_robot.py:200-251](src/lerobot/robots/marvin/marvin_robot.py#L200-L251)

```python
def _configure_arm(self, arm: str) -> None:
    # SDK Operation 6: Read current position
    data = self._sdk.subscribe()
    current_joints = data["outputs"][arm_idx]["fb_joint_pos"]
    
    # SDK Operation 7: Set tool gravity compensation (if enabled)
    if self.config.enable_gravity_compensation:
        self._sdk.clear_set()
        ret = self._sdk.set_tool(arm=arm, kineParams=tool_kine, dynamicParams=tool_dyn)
        self._sdk.send_cmd()
        time.sleep(0.1)
    
    # SDK Operation 8: Configure control mode
    if self.config.control_mode == "position":
        self._sdk.clear_set()
        self._sdk.set_state(arm=arm, state=1)  # Position control
        self._sdk.set_vel_acc(arm=arm, velRatio=..., AccRatio=...)
        self._sdk.set_joint_cmd_pose(arm=arm, joints=current_joints)
        self._sdk.send_cmd()
    else:  # impedance mode
        self._sdk.clear_set()
        self._sdk.set_state(arm=arm, state=3)  # Impedance control
        self._sdk.set_impedance_type(arm=arm, type=1)
        self._sdk.set_vel_acc(arm=arm, velRatio=..., AccRatio=...)
        self._sdk.set_joint_kd_params(arm=arm, K=..., D=...)
        self._sdk.set_joint_cmd_pose(arm=arm, joints=current_joints)
        self._sdk.send_cmd()
    time.sleep(0.3)
```

**SDK calls during configuration (per arm)**:
1. `self._sdk.subscribe()` — Read current joint positions
2. `self._sdk.clear_set()` — Clear command buffer
3. `self._sdk.set_tool()` — Set tool gravity compensation (if enabled)
4. `self._sdk.send_cmd()` — Send tool parameters
5. `self._sdk.clear_set()` — Clear command buffer again
6. `self._sdk.set_state()` — Set control mode (position=1 or impedance=3)
7. `self._sdk.set_vel_acc()` — Set velocity/acceleration limits
8. `self._sdk.set_impedance_type()` — Set impedance type (impedance mode only)
9. `self._sdk.set_joint_kd_params()` — Set K/D parameters (impedance mode only)
10. `self._sdk.set_joint_cmd_pose()` — Set initial target position
11. `self._sdk.send_cmd()` — Send configuration commands

---

## 2. Data Collection Loop

### 2.1 Get Observation (`robot.get_observation()`)

**File**: [marvin_robot.py:280-339](src/lerobot/robots/marvin/marvin_robot.py#L280-L339)

**Called at**: 30 Hz (dataset fps)

```python
def get_observation(self) -> RobotObservation:
    # SDK Operation 9: Read all sensor data
    with self._sdk_lock:
        data = self._sdk.subscribe()
    
    obs = {}
    out = data["outputs"][_ARM_OUT_IDX['B']]  # B arm (follower)
    
    # Extract joint positions (always)
    for i in range(7):
        obs[f"joint_{i+1}.pos"] = float(out["fb_joint_pos"][i])
    
    # Extract force feedback data (if enabled)
    if self.config.use_force_feedback:
        for i in range(7):
            if "joint_vel" in self.config.force_feedback_types:
                obs[f"joint_{i+1}.vel"] = float(out["fb_joint_vel"][i])
            
            if "joint_torque" in self.config.force_feedback_types:
                obs[f"joint_{i+1}.torque"] = float(out["fb_joint_sToq"][i])
            
            if "joint_force" in self.config.force_feedback_types:
                obs[f"joint_{i+1}.force"] = float(out["est_joint_force"][i])
        
        if "cart_force" in self.config.force_feedback_types:
            cart_fn = out["est_cart_fn"]
            obs["cart_force.fx"] = float(cart_fn[0])
            obs["cart_force.fy"] = float(cart_fn[1])
            obs["cart_force.fz"] = float(cart_fn[2])
            obs["cart_force.mx"] = float(cart_fn[3])
            obs["cart_force.my"] = float(cart_fn[4])
            obs["cart_force.mz"] = float(cart_fn[5])
    
    # Extract gripper position (if enabled)
    if self.config.use_gripper and self._gripper is not None:
        self._gripper.recv()  # SDK Operation 10: Read gripper feedback
        right_gripper_pos = float(self._motor_right.getPosition())
        obs["gripper.pos"] = right_gripper_pos
    
    # Store data for shared access (teleop alignment)
    set_latest_data(self.config.ip, data)
    
    # Read camera images
    for cam_key, cam in self.cameras.items():
        obs[cam_key] = cam.read_latest()
    
    return obs
```

**SDK calls during observation (every frame at 30Hz)**:
1. `self._sdk.subscribe()` — Read all real-time data from robot controller

**Data read from `subscribe()` (B arm only)**:
- `fb_joint_pos[7]` — Joint positions (always read)
- `fb_joint_vel[7]` — Joint velocities (if `joint_vel` enabled)
- `fb_joint_sToq[7]` — Joint sensor torques (if `joint_torque` enabled)
- `est_joint_force[7]` — Joint external forces (if `joint_force` enabled)
- `est_cart_fn[6]` — End-effector Cartesian forces (if `cart_force` enabled)

**Gripper data (if enabled)**:
2. `self._gripper.recv()` — Read gripper motor feedback via 485/CAN

---

### 2.2 Send Action (`robot.send_action()`)

**File**: [marvin_robot.py:341-421](src/lerobot/robots/marvin/marvin_robot.py#L341-L421)

**Called at**: 30 Hz (dataset fps)

```python
def send_action(self, action: RobotAction) -> RobotAction:
    already_connected = is_shared(self.config.ip)  # Check if teleop is active
    
    # Gripper control (if enabled)
    if self.config.use_gripper and self._gripper is not None:
        if self._gripper_update_counter % gripper_update_divisor == 0:
            if already_connected:
                # Teleoperation mode: Left gripper draggable, right follows
                m1_pos = self._motor_left.getPosition()
                m2_target = m1_pos + 0.1
                
                # SDK Operation 11: Control gripper motors
                self._gripper.controlMIT(self._motor_left, kp, kd, m1_pos, ...)
                self._gripper.controlMIT(self._motor_right, kp, kd, m2_target, ...)
            else:
                # Policy mode: Control right gripper only
                gripper_target = action.get("gripper.pos")
                self._gripper.controlMIT(self._motor_right, kp, kd, gripper_target, ...)
    
    # Arm control (policy mode only)
    if not already_connected:
        with self._sdk_lock:
            # SDK Operation 12: Send joint position commands
            self._sdk.clear_set()
            joints = [action[f"joint_{i+1}.pos"] for i in range(7)]
            self._sdk.set_joint_cmd_pose(arm='B', joints=joints)
            self._sdk.send_cmd()
    
    return action
```

**SDK calls during action sending (every frame at 30Hz)**:

**Policy inference mode** (`already_connected=False`):
1. `self._sdk.clear_set()` — Clear command buffer
2. `self._sdk.set_joint_cmd_pose(arm='B', joints=...)` — Set B arm target positions
3. `self._sdk.send_cmd()` — Send commands to robot
4. `self._gripper.controlMIT()` — Control right gripper motor (if enabled)

**Teleoperation mode** (`already_connected=True`):
1. `self._gripper.controlMIT()` × 2 — Control both gripper motors (if enabled)
2. **No arm commands sent** — B arm follows A arm via CYR (hardware-level)

---

## 3. Data Flow Summary

### 3.1 Complete SDK Operation Sequence (per frame at 30Hz)

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Read Sensor Data                                         │
│    self._sdk.subscribe()                                    │
│    ├─ Joint positions (7 values)                            │
│    ├─ Joint velocities (7 values, optional)                 │
│    ├─ Joint torques (7 values, optional)                    │
│    ├─ Joint external forces (7 values, optional)            │
│    └─ End-effector Cartesian forces (6 values, optional)    │
│                                                              │
│ 2. Read Gripper Feedback (if enabled)                       │
│    self._gripper.recv()                                     │
│    └─ Gripper motor positions                               │
│                                                              │
│ 3. Read Camera Images                                       │
│    cam.read_latest()                                        │
│    └─ Camera frames (OpenCV/RealSense)                      │
│                                                              │
│ 4. Send Control Commands (policy mode only)                 │
│    self._sdk.clear_set()                                    │
│    self._sdk.set_joint_cmd_pose(arm='B', joints=...)        │
│    self._sdk.send_cmd()                                     │
│                                                              │
│ 5. Control Gripper (if enabled)                             │
│    self._gripper.controlMIT(motor, kp, kd, target, ...)    │
└─────────────────────────────────────────────────────────────┘
```

---

### 3.2 SDK Data Structure

From `self._sdk.subscribe()`, we get:

```python
data = {
    "states": [
        {"cur_state": int, "cmd_state": int, "err_code": int},  # A arm
        {"cur_state": int, "cmd_state": int, "err_code": int}   # B arm
    ],
    "outputs": [
        {  # A arm (leader, not recorded)
            "frame_serial": int,
            "fb_joint_pos": [float × 7],      # ✅ Joint positions
            "fb_joint_vel": [float × 7],      # ✅ Joint velocities
            "fb_joint_sToq": [float × 7],     # ✅ Joint sensor torques
            "est_joint_force": [float × 7],   # ✅ Joint external forces
            "est_cart_fn": [float × 6],       # ✅ End-effector Cartesian forces
            "fb_joint_posE": [float × 7],     # External encoder positions
            "fb_joint_cmd": [float × 7],      # Command positions
            "fb_joint_cToq": [float × 7],     # Command torques
            "fb_joint_them": [float × 7],     # Joint temperatures
            "est_joint_firc": [float × 7],    # Friction estimates
            "est_joint_firc_dot": [float × 7],# Friction derivatives
            ...
        },
        {  # B arm (follower, recorded)
            # Same structure as A arm
        }
    ],
    "inputs": [...],  # Real-time input commands
}
```

**We only record B arm data** (follower/execution arm).

---

## 4. Performance Considerations

### 4.1 SDK Call Frequency

| Operation | Frequency | Notes |
|-----------|-----------|-------|
| `subscribe()` | 30 Hz | Every frame during recording |
| `send_cmd()` | 30 Hz | Every frame in policy mode, 0 Hz in teleop mode |
| `gripper.recv()` | 30 Hz | If gripper enabled |
| `gripper.controlMIT()` | 6 Hz | Rate-limited to prevent EtherCAT loss |

### 4.2 Data Bandwidth

**Without force feedback** (position only):
- 7 joint positions × 4 bytes = 28 bytes/frame
- At 30 fps: ~840 bytes/second

**With all force feedback enabled**:
- 7 positions + 7 velocities + 7 torques + 7 forces + 6 cart forces = 34 values
- 34 × 4 bytes = 136 bytes/frame
- At 30 fps: ~4 KB/second

**Total bandwidth (including cameras)**:
- 1 camera (640×480 RGB) = ~900 KB/frame (uncompressed)
- At 30 fps: ~27 MB/second (before video compression)

---

## 5. Key SDK Functions

### 5.1 Data Reading Functions

| Function | Purpose | Return Value |
|----------|---------|--------------|
| `subscribe()` | Read all real-time sensor data | `dict` with states/outputs/inputs |
| `gripper.recv()` | Read gripper motor feedback | Updates internal motor state |
| `motor.getPosition()` | Get gripper motor position | `float` (radians) |

### 5.2 Command Sending Functions

| Function | Purpose | Parameters |
|----------|---------|------------|
| `clear_set()` | Clear command buffer | None |
| `set_joint_cmd_pose()` | Set target joint positions | `arm`, `joints[7]` (degrees) |
| `send_cmd()` | Send all buffered commands | None |
| `gripper.controlMIT()` | Control gripper motor | `motor`, `kp`, `kd`, `target`, `vel`, `torque` |

### 5.3 Configuration Functions

| Function | Purpose | Parameters |
|----------|---------|------------|
| `connect()` | Connect to robot controller | `ip` |
| `set_state()` | Set control mode | `arm`, `state` (1=position, 3=impedance) |
| `set_vel_acc()` | Set velocity/acceleration limits | `arm`, `velRatio`, `AccRatio` |
| `set_tool()` | Set tool gravity compensation | `arm`, `kineParams[6]`, `dynamicParams[10]` |
| `set_impedance_type()` | Set impedance type | `arm`, `type` (1=joint, 2=cart, 3=force) |
| `set_joint_kd_params()` | Set impedance K/D parameters | `arm`, `K[7]`, `D[7]` |

---

## 6. Error Handling

### 6.1 Connection Verification

After `connect()`, the SDK reads 10 consecutive frames to verify:
- Frame serial number is updating (non-zero and incrementing)
- At least 3 different frame IDs observed within 1 second

If verification fails → `ConnectionError` is raised.

### 6.2 SDK Lock

All SDK operations are protected by `self._sdk_lock` (threading.Lock) to prevent:
- Race conditions when robot and teleop share the same SDK connection
- Concurrent UDP packet corruption
- Command buffer conflicts

---

## 7. Teleoperation vs Policy Mode

### 7.1 Teleoperation Mode (`skip_configure=True`)

**Initialization**:
- Robot connects but **does not configure** arms
- Teleop connects and configures arms + enables CYR mode

**Data Collection Loop**:
- `get_observation()`: Reads B arm sensor data via `subscribe()`
- `send_action()`: **Does not send arm commands** (CYR handles it)
- B arm follows A arm via hardware-level CYR control

### 7.2 Policy Mode (`skip_configure=False`)

**Initialization**:
- Robot connects and **configures both arms** (if `use_arm="AB"`)
- Sets tool gravity compensation + impedance parameters

**Data Collection Loop**:
- `get_observation()`: Reads B arm sensor data via `subscribe()`
- `send_action()`: **Sends joint position commands** to B arm via `send_cmd()`
- B arm executes policy predictions

---

## 8. Summary

**Total SDK operations per frame (30 Hz)**:

| Operation | Teleop Mode | Policy Mode |
|-----------|-------------|-------------|
| `subscribe()` | 1× | 1× |
| `gripper.recv()` | 1× (if enabled) | 1× (if enabled) |
| `gripper.controlMIT()` | 2× (rate-limited) | 1× (rate-limited) |
| `clear_set()` | 0× | 1× |
| `set_joint_cmd_pose()` | 0× | 1× |
| `send_cmd()` | 0× | 1× |

**Data read per frame**:
- Minimum: 7 joint positions (28 bytes)
- Maximum: 34 motor values (136 bytes) + camera images (~900 KB)

**Critical sections**:
- All SDK calls are locked via `self._sdk_lock`
- Gripper control is rate-limited to 6 Hz (30 Hz / 5)
- Camera reading happens outside the SDK lock

---

## References

- SDK Implementation: [fx_robot.py](src/lerobot/Marvin_sdk/fx_robot.py)
- Robot Implementation: [marvin_robot.py](src/lerobot/robots/marvin/marvin_robot.py)
- Recording Script: [lerobot_record.py](src/lerobot/scripts/lerobot_record.py)
