#!/usr/bin/env python

"""End-to-end visual/joint-force diffusion configuration for Marvin."""

from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.policies.rdp.configuration_rdp import RDPConfig
from lerobot.utils.constants import ACTION, OBS_STATE


@PreTrainedConfig.register_subclass("implicitrdp")
@dataclass
class ImplicitRDPConfig(RDPConfig):
    """No-auxiliary ImplicitRDP adaptation using joint-force observations.

    Unlike two-stage RDP, the action tokenizer, reactive decoder, visual
    conditioner, and latent diffusion model are optimized together.
    """

    training_stage: str = "diffusion"
    reconstruction_weight: float = 1.0
    diffusion_weight: float = 1.0
    latent_stats_momentum: float = 0.01

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.training_stage != "diffusion":
            raise ValueError("ImplicitRDP is a single-stage end-to-end policy")
        if self.reconstruction_weight <= 0 or self.diffusion_weight <= 0:
            raise ValueError("ImplicitRDP loss weights must be positive")
        if not 0.0 < self.latent_stats_momentum <= 1.0:
            raise ValueError("latent_stats_momentum must be in (0, 1]")

    def delta_indices_for_feature(self, key: str) -> list[int] | None:
        sequence = list(range(-self.force_history_pre, self.horizon - self.force_history_pre))
        if key in {ACTION, self.force_feature_key}:
            return sequence
        if key == OBS_STATE or key.startswith("observation.images."):
            return [-(self.slow_obs_steps - 1) * self.slow_obs_stride, 0]
        return None
