#!/usr/bin/env python

"""PI0-based ForceVLA with a separate, dataset-sized force token."""

from pathlib import Path

import torch
from torch import Tensor, nn

from lerobot.configs import PreTrainedConfig
from lerobot.policies.pi0.modeling_pi0 import PI0Policy, PI0Pytorch
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import OBS_STATE
from lerobot.utils.import_utils import require_package

from .configuration_forcevla import ForceVLAConfig


class ForceVLAPytorch(PI0Pytorch):
    """Insert a learned force token between the PI0 state and action tokens."""

    def __init__(self, config: ForceVLAConfig, rtc_processor=None):
        super().__init__(config, rtc_processor=rtc_processor)
        self.force_proj = nn.Linear(config.force_dim, self.state_proj.out_features)

    def embed_suffix(self, state, noisy_actions, timestep):
        expected_dim = self.config.proprio_dim + self.config.force_dim
        if state.ndim != 2 or state.shape[-1] != expected_dim:
            raise ValueError(
                "ForceVLA internal state must contain proprioception followed by force, "
                f"expected [B,{expected_dim}], got {tuple(state.shape)}"
            )
        # Keep PI0's pretrained state projection at max_state_dim.  The force
        # vector travels beside it and is projected by force_proj, so even a
        # dense tactile field does not resize or invalidate pretrained weights.
        proprio = state.new_zeros((state.shape[0], self.config.max_state_dim))
        proprio[:, : self.config.proprio_dim] = state[:, : self.config.proprio_dim]
        force = state[
            :,
            self.config.proprio_dim : self.config.proprio_dim + self.config.force_dim,
        ]
        embs, pad_masks, att_masks, adarms_cond = super().embed_suffix(
            proprio, noisy_actions, timestep
        )
        if self.force_proj.weight.dtype == torch.float32:
            force = force.to(torch.float32)
        force_emb = self._apply_checkpoint(self.force_proj, force)[:, None, :]
        force_mask = torch.ones(
            force_emb.shape[0], 1, dtype=torch.bool, device=force_emb.device
        )
        force_attention = torch.ones(
            force_emb.shape[0], 1, dtype=att_masks.dtype, device=att_masks.device
        )
        embs = torch.cat((embs[:, :1], force_emb, embs[:, 1:]), dim=1)
        pad_masks = torch.cat((pad_masks[:, :1], force_mask, pad_masks[:, 1:]), dim=1)
        att_masks = torch.cat(
            (att_masks[:, :1], force_attention, att_masks[:, 1:]), dim=1
        )
        return embs, pad_masks, att_masks, adarms_cond


class ForceVLAPolicy(PI0Policy):
    config_class = ForceVLAConfig
    name = "forcevla"

    def __init__(self, config: ForceVLAConfig, **kwargs):
        require_package("transformers", extra="pi")
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = ForceVLAPytorch(config, rtc_processor=self.rtc_processor)
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.model.to(config.device)
        self.reset()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path,
        *,
        config=None,
        force_download: bool = False,
        resume_download=None,
        proxies=None,
        token=None,
        cache_dir=None,
        local_files_only: bool = False,
        revision=None,
        strict: bool | None = None,
        **kwargs,
    ):
        """Load PI0 base or ForceVLA weights and fail closed on any mismatch.

        A PI0 base checkpoint may omit exactly the two new force projection
        tensors. A trained ForceVLA checkpoint must contain every tensor. The
        upstream PI0 loader returns a random model after some load failures;
        that behavior is unsafe for physical rollout and is intentionally not
        inherited here.
        """
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )
        if not isinstance(config, ForceVLAConfig):
            raise TypeError(
                "ForceVLAPolicy requires ForceVLAConfig; provide policy.type=forcevla "
                "when initializing from a PI0 base checkpoint"
            )
        model = cls(config, **kwargs)
        model_id = str(pretrained_name_or_path)
        local_model = Path(model_id).expanduser() / "model.safetensors"
        try:
            if Path(model_id).expanduser().is_dir():
                if not local_model.is_file():
                    raise FileNotFoundError(f"Missing checkpoint file: {local_model}")
                resolved_file = str(local_model)
            else:
                from transformers.utils import cached_file

                resolved_file = cached_file(
                    model_id,
                    "model.safetensors",
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    token=token,
                    revision=revision,
                    local_files_only=local_files_only,
                )
                if resolved_file is None:
                    raise FileNotFoundError(f"model.safetensors was not resolved for {model_id}")
            from safetensors.torch import load_file

            original_state = load_file(resolved_file)
        except Exception as error:
            raise RuntimeError(f"ForceVLA checkpoint load failed for {model_id}: {error}") from error

        fixed_state = model._fix_pytorch_state_dict_keys(original_state, model.config)
        remapped_state = {
            key if key.startswith("model.") else f"model.{key}": value
            for key, value in fixed_state.items()
        }
        missing, unexpected = model.load_state_dict(remapped_state, strict=False)
        force_keys = {"model.force_proj.weight", "model.force_proj.bias"}
        source_has_force_adapter = bool(force_keys & set(remapped_state))
        allowed_missing = set() if source_has_force_adapter else force_keys
        invalid_missing = set(missing) - allowed_missing
        if invalid_missing or unexpected or set(missing) != allowed_missing:
            raise RuntimeError(
                "ForceVLA checkpoint is incompatible: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        if strict is True and missing:
            raise RuntimeError(
                "Strict ForceVLA loading rejected a PI0 base checkpoint without force_proj"
            )
        model.to(config.device)
        model.eval()
        return model

    def prepare_state(self, batch: dict[str, Tensor]) -> Tensor:
        state = batch[OBS_STATE]
        force = batch[self.config.force_feature_key]
        if state.ndim != 2 or force.ndim != 2:
            raise ValueError(
                "ForceVLA expects current state/force vectors, got "
                f"{state.shape}, {force.shape}"
            )
        if not torch.isfinite(force).all():
            raise ValueError("ForceVLA received non-finite effort")
        if state.shape[-1] != self.config.proprio_dim:
            raise ValueError(
                f"ForceVLA expected state[{self.config.proprio_dim}], got {tuple(state.shape)}"
            )
        if force.shape[-1] != self.config.force_dim:
            raise ValueError(
                f"ForceVLA expected force[{self.config.force_dim}], got {tuple(force.shape)}"
            )
        return torch.cat((state, force), dim=-1)
