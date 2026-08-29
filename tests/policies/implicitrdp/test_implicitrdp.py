import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.implicitrdp.configuration_implicitrdp import ImplicitRDPConfig
from lerobot.policies.implicitrdp.modeling_implicitrdp import ImplicitRDPPolicy
from lerobot.utils.constants import ACTION, OBS_STATE


def _config() -> ImplicitRDPConfig:
    return ImplicitRDPConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
            "observation.joint_force": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
            "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(8,))},
        horizon=16,
        latent_horizon=4,
        latent_dim=8,
        tokenizer_hidden_dim=16,
        condition_dim=32,
        unet_base_dim=8,
        resize_shape=(32, 32),
        pretrained_backbone_weights=None,
        num_train_timesteps=8,
        num_inference_steps=2,
        latent_stats_steps=1,
        drop_n_last_frames=12,
    )


def test_end_to_end_loss_updates_all_modules():
    policy = ImplicitRDPPolicy(_config())
    batch = {
        ACTION: torch.randn(2, 16, 8),
        OBS_STATE: torch.randn(2, 2, 8),
        "observation.joint_force": torch.randn(2, 16, 7),
        "observation.images.top": torch.randn(2, 2, 3, 32, 32),
        "action_is_pad": torch.zeros(2, 16, dtype=torch.bool),
    }
    loss, metrics = policy(batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert "reconstruction_loss" in metrics and "diffusion_loss" in metrics
    assert any(parameter.grad is not None for parameter in policy.tokenizer.parameters())
    assert any(parameter.grad is not None for parameter in policy.conditioner.parameters())
    assert any(parameter.grad is not None for parameter in policy.denoiser.parameters())


def test_eval_does_not_mutate_latent_calibration_buffers():
    policy = ImplicitRDPPolicy(_config()).eval()
    batch = {
        ACTION: torch.randn(2, 16, 8),
        OBS_STATE: torch.randn(2, 2, 8),
        "observation.joint_force": torch.randn(2, 16, 7),
        "observation.images.top": torch.randn(2, 2, 3, 32, 32),
        "action_is_pad": torch.zeros(2, 16, dtype=torch.bool),
    }
    before = {
        name: value.clone()
        for name, value in policy.named_buffers()
        if name.startswith("latent_")
    }
    with torch.no_grad():
        policy(batch)
    for name, expected in before.items():
        torch.testing.assert_close(dict(policy.named_buffers())[name], expected)
