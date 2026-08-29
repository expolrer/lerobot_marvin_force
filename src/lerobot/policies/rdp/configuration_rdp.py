#!/usr/bin/env python

"""Native LeRobot configuration for force-reactive diffusion policy (RDP)."""

from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig
from lerobot.utils.constants import ACTION, OBS_STATE


@PreTrainedConfig.register_subclass("rdp")
@dataclass
class RDPConfig(PreTrainedConfig):
    """Two-stage RDP adapted to Marvin joint-position control.

    Stage ``tokenizer`` learns an action latent with a force-conditioned GRU
    decoder. Stage ``diffusion`` freezes that tokenizer and learns a slow visual
    latent diffusion model. A stage-2 checkpoint contains both halves.
    """

    training_stage: str = "tokenizer"
    force_feature_key: str = "observation.joint_force"
    dtype: str = "bfloat16"

    horizon: int = 32
    force_history_pre: int = 3
    slow_obs_steps: int = 2
    slow_obs_stride: int = 2
    control_hz: int = 30
    slow_hz: int = 6
    drop_n_last_frames: int = 28

    latent_horizon: int = 8
    latent_dim: int = 64
    tokenizer_hidden_dim: int = 128
    condition_dim: int = 256
    unet_base_dim: int = 64

    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    resize_shape: tuple[int, int] = (240, 320)

    num_train_timesteps: int = 100
    num_inference_steps: int = 10
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    tokenizer_kl_weight: float = 1e-6
    latent_stats_steps: int = 500

    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-6

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.QUANTILES,
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.training_stage not in {"tokenizer", "diffusion"}:
            raise ValueError("training_stage must be 'tokenizer' or 'diffusion'")
        if self.dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("dtype must be bfloat16, float16, or float32")
        if self.horizon < 4 or self.horizon % 4:
            raise ValueError("horizon must be divisible by 4")
        if self.latent_horizon != self.horizon // 4:
            raise ValueError("latent_horizon must equal horizon // 4 for the action encoder")
        if self.latent_horizon < 4 or self.latent_horizon % 4:
            raise ValueError("latent_horizon must be at least 4 and divisible by 4 for the U-Net")
        if self.force_history_pre >= self.horizon:
            raise ValueError("force_history_pre must be smaller than horizon")
        if self.drop_n_last_frames < self.horizon - self.force_history_pre - 1:
            raise ValueError("drop_n_last_frames must exclude the furthest future action target")
        if self.slow_obs_steps != 2:
            raise ValueError("The current native RDP encoder supports exactly two slow observations")
        if self.control_hz <= 0 or self.slow_hz <= 0 or self.control_hz % self.slow_hz:
            raise ValueError("control_hz must be a positive integer multiple of slow_hz")
        if len(self.resize_shape) != 2 or min(self.resize_shape) <= 0:
            raise ValueError("resize_shape must contain two positive integers")

    @property
    def slow_interval(self) -> int:
        return self.control_hz // self.slow_hz

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if not self.robot_state_feature:
            raise ValueError("RDP requires observation.state")
        if self.force_feature_key not in self.input_features:
            raise ValueError(f"RDP requires {self.force_feature_key}")
        force_shape = self.input_features[self.force_feature_key].shape
        if len(force_shape) != 1:
            raise ValueError(f"RDP force feature must be a vector, got {force_shape}")
        if not self.image_features:
            raise ValueError("RDP requires at least one RGB image")

    def delta_indices_for_feature(self, key: str) -> list[int] | None:
        """Return feature-specific time indices without decoding image horizons."""
        sequence = list(range(-self.force_history_pre, self.horizon - self.force_history_pre))
        if key == ACTION:
            return sequence
        if self.training_stage == "tokenizer":
            return sequence if key == self.force_feature_key else None
        if key == self.force_feature_key or key == OBS_STATE or key.startswith("observation.images."):
            return [-(self.slow_obs_steps - 1) * self.slow_obs_stride, 0]
        return None

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(-self.force_history_pre, self.horizon - self.force_history_pre))

    @property
    def reward_delta_indices(self) -> None:
        return None
