#!/usr/bin/env python

"""End-to-end ImplicitRDP adaptation for Marvin joint-force feedback."""

import torch
import torch.nn.functional as F
from torch import Tensor

from lerobot.policies.rdp.modeling_rdp import RDPPolicy
from lerobot.utils.constants import ACTION

from .configuration_implicitrdp import ImplicitRDPConfig


class ImplicitRDPPolicy(RDPPolicy):
    config_class = ImplicitRDPConfig
    name = "implicitrdp"

    def __init__(self, config: ImplicitRDPConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)
        # RDP's diffusion stage freezes this module; ImplicitRDP is end-to-end.
        self.tokenizer.requires_grad_(True)

    def _slow_condition_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        slow_batch = {
            key: batch[key] for key in [*self.config.image_features, "observation.state"]
        }
        force = batch[self.config.force_feature_key]
        if force.ndim != 3 or force.shape[1] != self.config.horizon:
            raise ValueError(
                f"ImplicitRDP expects force [B,{self.config.horizon},F], got {tuple(force.shape)}"
            )
        current = self.config.force_history_pre
        previous = current - self.config.slow_obs_stride
        if previous < 0:
            raise ValueError("force_history_pre must cover the previous slow observation")
        slow_batch[self.config.force_feature_key] = force[:, [previous, current]]
        return slow_batch

    @torch.no_grad()
    def _track_encoder_latent_stats(self, mu: Tensor) -> None:
        """Calibrate initially, then track the evolving encoder with an EMA."""
        if not bool(self.latent_calibrated):
            self._update_latent_stats(mu)
            return
        values = mu.reshape(-1, mu.shape[-1]).float()
        batch_mean = values.mean(dim=0)
        batch_std = values.std(dim=0, unbiased=False).clamp_min(1e-4)
        momentum = self.config.latent_stats_momentum
        self.latent_mean.lerp_(batch_mean, momentum)
        self.latent_std.lerp_(batch_std, momentum)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        action = batch[ACTION]
        force = batch[self.config.force_feature_key]
        latent, mu, logvar = self.tokenizer.encode(action, sample=True)
        reconstructed = self.tokenizer.decode(latent, force)
        reconstruction = self._masked_action_l1(
            reconstructed, action, batch.get("action_is_pad")
        )
        kl = (-0.5 * (1.0 + logvar - mu.square() - logvar.exp())).sum(-1).mean()
        tokenizer_loss = reconstruction + self.config.tokenizer_kl_weight * kl

        # Track deterministic encoder means only during optimization. An EMA
        # after initial calibration follows the jointly trained encoder without
        # letting validation data or VAE sampling noise mutate deployment stats.
        if self.training:
            self._track_encoder_latent_stats(mu.detach())
        metrics = {
            "reconstruction_loss": reconstruction.item(),
            "kld_loss": kl.item(),
            "latent_calibrated": float(self.latent_calibrated.item()),
        }
        if not bool(self.latent_calibrated):
            return self.config.reconstruction_weight * tokenizer_loss, metrics

        normalized = (latent - self.latent_mean) / self.latent_std
        timestep = torch.randint(
            0, self.config.num_train_timesteps, (action.shape[0],), device=action.device
        )
        noise = torch.randn_like(normalized)
        alpha_bar = self.alpha_bar[timestep].view(-1, 1, 1)
        noisy = alpha_bar.sqrt() * normalized + (1.0 - alpha_bar).sqrt() * noise
        condition = self.conditioner(
            self._slow_condition_batch(batch), self.config.force_feature_key
        )
        predicted_noise = self.denoiser(noisy, timestep, condition)
        diffusion = F.mse_loss(predicted_noise, noise)
        loss = (
            self.config.reconstruction_weight * tokenizer_loss
            + self.config.diffusion_weight * diffusion
        )
        metrics["diffusion_loss"] = diffusion.item()
        return loss, metrics
