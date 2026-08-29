#!/usr/bin/env python

"""Force-reactive latent diffusion policy for Marvin.

The implementation follows RDP's slow/fast split: a slow visual diffusion
model predicts a latent action chunk, while a small force-conditioned GRU
re-decodes that latent on every control tick. The output remains an absolute
joint-position action; force is an observation, never a torque command.
"""

from __future__ import annotations

import math
from collections import deque

import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn

from lerobot.utils.constants import ACTION, OBS_STATE

from ..pretrained import PreTrainedPolicy
from .configuration_rdp import RDPConfig


class ActionTokenizer(nn.Module):
    """Compress action chunks and reactively decode them with a force sequence."""

    def __init__(self, action_dim: int, force_dim: int, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(action_dim, hidden_dim, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 5, stride=2, padding=2),
            nn.SiLU(),
        )
        self.to_mu = nn.Conv1d(hidden_dim, latent_dim, 1)
        self.to_logvar = nn.Conv1d(hidden_dim, latent_dim, 1)
        self.decoder = nn.GRU(latent_dim + force_dim, hidden_dim, batch_first=True)
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def encode(self, action: Tensor, sample: bool = True) -> tuple[Tensor, Tensor, Tensor]:
        encoded = self.encoder(action.transpose(1, 2))
        mu = self.to_mu(encoded).transpose(1, 2)
        logvar = self.to_logvar(encoded).transpose(1, 2).clamp(-10.0, 10.0)
        latent = mu
        if sample:
            latent = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return latent, mu, logvar

    def decode(self, latent: Tensor, force: Tensor) -> Tensor:
        if force.ndim != 3:
            raise ValueError(f"Expected force [B,T,F], got {tuple(force.shape)}")
        latent_steps = F.interpolate(
            latent.transpose(1, 2), size=force.shape[1], mode="linear", align_corners=False
        ).transpose(1, 2)
        decoded, _ = self.decoder(torch.cat((latent_steps, force), dim=-1))
        return self.action_head(decoded)


class SlowConditionEncoder(nn.Module):
    """Shared ResNet encoder for two slow observations from every camera."""

    def __init__(self, config: RDPConfig, state_dim: int, force_dim: int) -> None:
        super().__init__()
        # A saved RDP checkpoint already contains the complete backbone. Avoid
        # a network/cache dependency before state_dict loading during offline
        # deployment or stage-2 resume. Fresh stage-1 training may initialize
        # from the configured torchvision weights.
        backbone_weights = None if config.pretrained_path else config.pretrained_backbone_weights
        backbone = getattr(torchvision.models, config.vision_backbone)(weights=backbone_weights)
        self.vision_dim = backbone.fc.in_features
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.image_keys = list(config.image_features)
        self.slow_obs_steps = config.slow_obs_steps
        self.resize_shape = config.resize_shape
        flat_dim = (
            len(self.image_keys) * self.slow_obs_steps * self.vision_dim
            + self.slow_obs_steps * (state_dim + force_dim)
        )
        self.output = nn.Sequential(
            nn.Linear(flat_dim, config.condition_dim),
            nn.LayerNorm(config.condition_dim),
            nn.SiLU(),
            nn.Linear(config.condition_dim, config.condition_dim),
        )

    def _ensure_time(self, value: Tensor) -> Tensor:
        if value.ndim in {2, 4}:
            value = value.unsqueeze(1)
        if value.shape[1] == 1:
            value = value.expand(-1, self.slow_obs_steps, *value.shape[2:])
        if value.shape[1] != self.slow_obs_steps:
            raise ValueError(
                f"Expected {self.slow_obs_steps} slow observations, got {tuple(value.shape)}"
            )
        return value

    def forward(self, batch: dict[str, Tensor], force_key: str) -> Tensor:
        features: list[Tensor] = []
        for key in self.image_keys:
            image = self._ensure_time(batch[key])
            batch_size, steps, channels, height, width = image.shape
            image = F.interpolate(
                image.reshape(batch_size * steps, channels, height, width),
                size=self.resize_shape,
                mode="bilinear",
                align_corners=False,
            )
            encoded = self.backbone(image).flatten(1).reshape(batch_size, steps, -1)
            features.append(encoded.flatten(1))
        state = self._ensure_time(batch[OBS_STATE])
        force = self._ensure_time(batch[force_key])
        features.extend((state.flatten(1), force.flatten(1)))
        return self.output(torch.cat(features, dim=-1))


def _time_embedding(timestep: Tensor, dim: int) -> Tensor:
    half = dim // 2
    scale = math.log(10000.0) / max(half - 1, 1)
    frequencies = torch.exp(-scale * torch.arange(half, device=timestep.device, dtype=torch.float32))
    angles = timestep.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
    return F.pad(embedding, (0, dim - embedding.shape[1]))


class ConditionalBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, condition_dim: int) -> None:
        super().__init__()
        groups = min(8, out_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.film = nn.Linear(condition_dim, out_channels * 2)
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, value: Tensor, condition: Tensor) -> Tensor:
        hidden = F.silu(self.norm1(self.conv1(value)))
        scale, bias = self.film(condition).chunk(2, dim=-1)
        hidden = hidden * (1.0 + scale.unsqueeze(-1)) + bias.unsqueeze(-1)
        hidden = self.norm2(self.conv2(F.silu(hidden)))
        return F.silu(hidden + self.skip(value))


class ConditionalLatentUNet(nn.Module):
    """Compact 1D U-Net over the low-rate latent action sequence."""

    def __init__(self, latent_dim: int, base_dim: int, condition_dim: int) -> None:
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(condition_dim, condition_dim), nn.SiLU(), nn.Linear(condition_dim, condition_dim)
        )
        self.input = nn.Conv1d(latent_dim, base_dim, 1)
        self.block0 = ConditionalBlock1D(base_dim, base_dim, condition_dim)
        self.down1 = nn.Conv1d(base_dim, base_dim * 2, 4, stride=2, padding=1)
        self.block1 = ConditionalBlock1D(base_dim * 2, base_dim * 2, condition_dim)
        self.down2 = nn.Conv1d(base_dim * 2, base_dim * 4, 4, stride=2, padding=1)
        self.mid = ConditionalBlock1D(base_dim * 4, base_dim * 4, condition_dim)
        self.up1 = nn.ConvTranspose1d(base_dim * 4, base_dim * 2, 4, stride=2, padding=1)
        self.block_up1 = ConditionalBlock1D(base_dim * 4, base_dim * 2, condition_dim)
        self.up2 = nn.ConvTranspose1d(base_dim * 2, base_dim, 4, stride=2, padding=1)
        self.block_up2 = ConditionalBlock1D(base_dim * 2, base_dim, condition_dim)
        self.output = nn.Conv1d(base_dim, latent_dim, 1)
        self.condition_dim = condition_dim

    def forward(self, noisy_latent: Tensor, timestep: Tensor, global_condition: Tensor) -> Tensor:
        condition = global_condition + self.time_mlp(_time_embedding(timestep, self.condition_dim))
        start = self.block0(self.input(noisy_latent.transpose(1, 2)), condition)
        down = self.block1(self.down1(start), condition)
        middle = self.mid(self.down2(down), condition)
        up = self.up1(middle)
        up = self.block_up1(torch.cat((up, down), dim=1), condition)
        up = self.up2(up)
        up = self.block_up2(torch.cat((up, start), dim=1), condition)
        return self.output(up).transpose(1, 2)


class RDPPolicy(PreTrainedPolicy):
    config_class = RDPConfig
    name = "rdp"

    def __init__(self, config: RDPConfig, **kwargs) -> None:
        super().__init__(config)
        config.validate_features()
        self.config = config
        action_dim = config.action_feature.shape[0]
        force_dim = config.input_features[config.force_feature_key].shape[0]
        state_dim = config.robot_state_feature.shape[0]

        self.tokenizer = ActionTokenizer(
            action_dim=action_dim,
            force_dim=force_dim,
            latent_dim=config.latent_dim,
            hidden_dim=config.tokenizer_hidden_dim,
        )
        self.conditioner = SlowConditionEncoder(config, state_dim=state_dim, force_dim=force_dim)
        self.denoiser = ConditionalLatentUNet(
            latent_dim=config.latent_dim,
            base_dim=config.unet_base_dim,
            condition_dim=config.condition_dim,
        )

        beta = torch.linspace(config.beta_start, config.beta_end, config.num_train_timesteps)
        alpha_bar = torch.cumprod(1.0 - beta, dim=0)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("latent_mean", torch.zeros(config.latent_dim))
        self.register_buffer("latent_std", torch.ones(config.latent_dim))
        self.register_buffer("latent_m2", torch.zeros(config.latent_dim))
        self.register_buffer("latent_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("latent_calibration_batches", torch.zeros((), dtype=torch.long))
        self.register_buffer("latent_calibrated", torch.zeros((), dtype=torch.bool))

        if config.training_stage == "tokenizer":
            self.conditioner.requires_grad_(False)
            self.denoiser.requires_grad_(False)
        else:
            self.tokenizer.requires_grad_(False)
        self.reset()

    def get_optim_params(self):
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def _save_pretrained(self, save_directory, state_dict=None) -> None:
        """Clone cuDNN GRU storage views before safetensors serialization.

        After a CUDA GRU forward, PyTorch may flatten its parameters into one
        shared storage. ``safetensors.save_model`` intentionally rejects those
        partial-storage views. A checkpoint is infrequent and this model is
        small, so materializing independent tensors is the safest portable
        representation and also works for an FSDP-gathered state dict.
        """
        source = self.state_dict() if state_dict is None else state_dict
        portable = {name: value.detach().clone() for name, value in source.items()}
        super()._save_pretrained(save_directory, state_dict=portable)

    def reset(self) -> None:
        self._online_tick = 0
        self._online_observations: deque[dict[str, Tensor]] = deque(maxlen=self.config.slow_obs_steps)
        self._online_force: deque[Tensor] = deque(maxlen=self.config.horizon)
        self._active_force: deque[Tensor] = deque(maxlen=self.config.horizon)
        self._active_latent: Tensor | None = None
        self._active_plan_tick = -1

    def _masked_action_l1(self, prediction: Tensor, target: Tensor, is_pad: Tensor | None) -> Tensor:
        error = (prediction - target).abs()
        if is_pad is None:
            return error.mean()
        valid = (~is_pad).unsqueeze(-1)
        return (error * valid).sum() / (valid.sum() * error.shape[-1]).clamp_min(1)

    @torch.no_grad()
    def _update_latent_stats(self, latent: Tensor) -> None:
        values = latent.reshape(-1, latent.shape[-1]).float()
        batch_count = values.shape[0]
        batch_mean = values.mean(dim=0)
        batch_m2 = ((values - batch_mean) ** 2).sum(dim=0)
        old_count = int(self.latent_count.item())
        new_count = old_count + batch_count
        delta = batch_mean - self.latent_mean
        if old_count == 0:
            self.latent_mean.copy_(batch_mean)
            self.latent_m2.copy_(batch_m2)
        else:
            self.latent_mean.add_(delta * (batch_count / new_count))
            self.latent_m2.add_(batch_m2 + delta.square() * old_count * batch_count / new_count)
        self.latent_count.fill_(new_count)
        self.latent_calibration_batches.add_(1)
        if self.latent_calibration_batches.item() >= self.config.latent_stats_steps:
            self.latent_std.copy_(torch.sqrt(self.latent_m2 / max(new_count, 1)).clamp_min(1e-4))
            self.latent_calibrated.fill_(True)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        action = batch[ACTION]
        if self.config.training_stage == "tokenizer":
            force = batch[self.config.force_feature_key]
            latent, mu, logvar = self.tokenizer.encode(action, sample=True)
            prediction = self.tokenizer.decode(latent, force)
            l1 = self._masked_action_l1(prediction, action, batch.get("action_is_pad"))
            kl = (-0.5 * (1.0 + logvar - mu.square() - logvar.exp())).sum(-1).mean()
            loss = l1 + self.config.tokenizer_kl_weight * kl
            return loss, {"l1_loss": l1.item(), "kld_loss": kl.item()}

        with torch.no_grad():
            latent, _, _ = self.tokenizer.encode(action, sample=False)
        if not bool(self.latent_calibrated):
            if self.training:
                self._update_latent_stats(latent)
            zero = sum(parameter.sum() for parameter in self.denoiser.parameters()) * 0.0
            return zero, {
                "latent_calibration_batches": float(self.latent_calibration_batches.item()),
                "latent_calibrated": float(self.latent_calibrated.item()),
            }

        normalized = (latent - self.latent_mean) / self.latent_std
        timestep = torch.randint(
            0, self.config.num_train_timesteps, (action.shape[0],), device=action.device
        )
        noise = torch.randn_like(normalized)
        alpha_bar = self.alpha_bar[timestep].view(-1, 1, 1)
        noisy = alpha_bar.sqrt() * normalized + (1.0 - alpha_bar).sqrt() * noise
        condition = self.conditioner(batch, self.config.force_feature_key)
        prediction = self.denoiser(noisy, timestep, condition)
        loss = F.mse_loss(prediction, noise)
        return loss, {"diffusion_loss": loss.item()}

    def _require_deployable(self) -> None:
        if self.config.training_stage != "diffusion":
            raise RuntimeError("RDP deployment requires a stage-2 diffusion checkpoint")
        if not bool(self.latent_calibrated):
            raise RuntimeError("RDP latent statistics are not calibrated; finish stage-2 calibration first")

    @torch.no_grad()
    def plan_latent(self, slow_batch: dict[str, Tensor]) -> Tensor:
        self._require_deployable()
        condition = self.conditioner(slow_batch, self.config.force_feature_key)
        sample = torch.randn(
            condition.shape[0],
            self.config.latent_horizon,
            self.config.latent_dim,
            device=condition.device,
            dtype=condition.dtype,
        )
        timesteps = torch.linspace(
            self.config.num_train_timesteps - 1,
            0,
            self.config.num_inference_steps,
            device=condition.device,
        ).round().long()
        for index, timestep_value in enumerate(timesteps):
            timestep = timestep_value.expand(condition.shape[0])
            predicted_noise = self.denoiser(sample, timestep, condition)
            alpha_now = self.alpha_bar[timestep_value]
            alpha_prev = (
                self.alpha_bar[timesteps[index + 1]]
                if index + 1 < len(timesteps)
                else torch.ones((), device=sample.device, dtype=sample.dtype)
            )
            predicted_clean = (sample - (1.0 - alpha_now).sqrt() * predicted_noise) / alpha_now.sqrt()
            sample = alpha_prev.sqrt() * predicted_clean + (1.0 - alpha_prev).sqrt() * predicted_noise
        return sample * self.latent_std + self.latent_mean

    @torch.no_grad()
    def decode_reactive_action(
        self, latent: Tensor, force_history: Tensor, plan_age: int = 0
    ) -> Tensor:
        """Decode a fixed-horizon plan and select the action aligned to now.

        The latent represents offsets ``[-force_history_pre, ..., future]``.
        Only forces through the current tick are observed; missing left history
        is filled with the earliest sample and unknown future force is filled
        with the latest sample. Decoding a short sequence and taking its final
        element would incorrectly stretch the whole plan onto a few ticks.
        """
        if force_history.ndim != 3 or force_history.shape[1] == 0:
            raise ValueError(
                f"Expected non-empty force history [B,T,F], got {tuple(force_history.shape)}"
            )
        max_age = self.config.horizon - self.config.force_history_pre - 1
        aligned_age = max(0, min(int(plan_age), max_age))
        observed_steps = self.config.force_history_pre + aligned_age + 1
        force_history = force_history[:, -observed_steps:]
        if force_history.shape[1] < observed_steps:
            missing = observed_steps - force_history.shape[1]
            left_pad = force_history[:, :1].expand(-1, missing, -1)
            force_history = torch.cat((left_pad, force_history), dim=1)
        future_steps = self.config.horizon - force_history.shape[1]
        if future_steps:
            right_pad = force_history[:, -1:].expand(-1, future_steps, -1)
            force_history = torch.cat((force_history, right_pad), dim=1)
        actions = self.tokenizer.decode(latent, force_history)
        return actions[:, self.config.force_history_pre + aligned_age]

    def _online_slow_batch(self) -> dict[str, Tensor]:
        observations = list(self._online_observations)
        while len(observations) < self.config.slow_obs_steps:
            observations.insert(0, observations[0])
        result: dict[str, Tensor] = {}
        for key in [*self.config.image_features, OBS_STATE, self.config.force_feature_key]:
            result[key] = torch.stack([item[key] for item in observations], dim=1)
        return result

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Synchronous fallback. Use ``--inference.type=rdp`` for nonblocking deployment."""
        self.eval()
        current = {
            key: batch[key]
            for key in [*self.config.image_features, OBS_STATE, self.config.force_feature_key]
        }
        self._online_observations.append(current)
        current_force = batch[self.config.force_feature_key]
        self._online_force.append(current_force)
        if self._active_latent is None or self._online_tick % self.config.slow_interval == 0:
            self._active_latent = self.plan_latent(self._online_slow_batch())
            self._active_plan_tick = self._online_tick
            self._active_force = deque(
                list(self._online_force)[-(self.config.force_history_pre + 1) :],
                maxlen=self.config.horizon,
            )
        elif len(self._active_force) < self.config.horizon:
            self._active_force.append(current_force)
        force_history = torch.stack(list(self._active_force), dim=1)
        action = self.decode_reactive_action(
            self._active_latent,
            force_history,
            plan_age=self._online_tick - self._active_plan_tick,
        )
        self._online_tick += 1
        return action

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        latent = self.plan_latent(batch)
        force = batch[self.config.force_feature_key]
        if force.ndim == 2:
            force = force.unsqueeze(1)
        latest_force = force[:, -1:].expand(-1, self.config.horizon, -1)
        return self.tokenizer.decode(latent, latest_force)
