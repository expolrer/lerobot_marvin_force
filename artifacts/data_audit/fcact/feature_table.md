| Feature | dtype | shape | Names / meaning |
|---|---:|---:|---|
| `action` | `float32` | `[8]` | joint_1.pos, joint_2.pos, joint_3.pos, joint_4.pos, joint_5.pos, joint_6.pos, joint_7.pos, gripper.pos |
| `observation.state` | `float32` | `[8]` | joint_1.pos, joint_2.pos, joint_3.pos, joint_4.pos, joint_5.pos, joint_6.pos, joint_7.pos, gripper.pos |
| `observation.joint_vel` | `float32` | `[7]` | joint_1.vel, joint_2.vel, joint_3.vel, joint_4.vel, joint_5.vel, joint_6.vel, joint_7.vel |
| `observation.joint_torque` | `float32` | `[7]` | joint_1.torque, joint_2.torque, joint_3.torque, joint_4.torque, joint_5.torque, joint_6.torque, joint_7.torque |
| `observation.joint_force` | `float32` | `[7]` | joint_1.force, joint_2.force, joint_3.force, joint_4.force, joint_5.force, joint_6.force, joint_7.force |
| `observation.cart_force` | `float32` | `[6]` | cart_force.fx, cart_force.fy, cart_force.fz, cart_force.mx, cart_force.my, cart_force.mz |
| `observation.images.up_cam` | `video` | `[480, 640, 3]` | height, width, channels |
| `observation.images.left_close` | `video` | `[480, 640, 3]` | height, width, channels |
| `observation.images.right_close` | `video` | `[480, 640, 3]` | height, width, channels |
| `timestamp` | `float32` | `[1]` | - |
| `frame_index` | `int64` | `[1]` | - |
| `episode_index` | `int64` | `[1]` | - |
| `index` | `int64` | `[1]` | - |
| `task_index` | `int64` | `[1]` | - |
