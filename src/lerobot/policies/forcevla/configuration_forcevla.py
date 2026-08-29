#!/usr/bin/env python

"""LeRobot-native ForceVLA configuration for the single-arm Marvin policy."""

from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.utils.constants import OBS_STATE


@PreTrainedConfig.register_subclass("forcevla")
@dataclass
class ForceVLAConfig(PI0Config):
    """PI0-based VLA with a distinct estimated-joint-force token.

    The reference dataset and the standalone Marvin rollout both expose only
    the follower B arm: seven joint positions plus one gripper position, seven
    estimated external joint-force channels, and eight position actions.
    """

    force_feature_key: str = "observation.joint_force"
    proprio_dim: int = 8
    force_dim: int = 7
    n_action_steps: int = 1

    def validate_features(self) -> None:
        super().validate_features()
        if OBS_STATE not in self.input_features:
            raise ValueError("ForceVLA requires observation.state")
        if self.force_feature_key not in self.input_features:
            raise ValueError(f"ForceVLA requires {self.force_feature_key}")
        state_shape = self.input_features[OBS_STATE].shape
        force_shape = self.input_features[self.force_feature_key].shape
        if state_shape != (self.proprio_dim,):
            raise ValueError(
                f"Expected state[{self.proprio_dim}], got {state_shape}"
            )
        if force_shape != (self.force_dim,):
            raise ValueError(
                f"Expected joint_force[{self.force_dim}], got {force_shape}"
            )
        if self.proprio_dim + self.force_dim > self.max_state_dim:
            raise ValueError("max_state_dim cannot hold state and force transport vector")
