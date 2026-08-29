import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.rdp.configuration_rdp import RDPConfig
from lerobot.policies.rdp.modeling_rdp import RDPPolicy
from lerobot.utils.constants import ACTION, OBS_STATE


def _config(stage: str) -> RDPConfig:
    return RDPConfig(
        training_stage=stage,
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
            "observation.joint_force": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
            "observation.images.up_cam": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
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


def test_tokenizer_loss_and_force_causality():
    policy = RDPPolicy(_config("tokenizer"))
    action = torch.randn(2, 16, 8)
    force = torch.randn(2, 16, 7)
    loss, metrics = policy(
        {ACTION: action, "observation.joint_force": force, "action_is_pad": torch.zeros(2, 16, dtype=torch.bool)}
    )
    assert torch.isfinite(loss) and "l1_loss" in metrics

    latent, _, _ = policy.tokenizer.encode(action, sample=False)
    altered = force.clone()
    altered[:, 5:] += 100
    original_actions = policy.tokenizer.decode(latent, force)
    altered_actions = policy.tokenizer.decode(latent, altered)
    torch.testing.assert_close(original_actions[:, :5], altered_actions[:, :5])


def test_diffusion_calibrates_then_trains():
    policy = RDPPolicy(_config("diffusion"))
    batch = {
        ACTION: torch.randn(2, 16, 8),
        OBS_STATE: torch.randn(2, 2, 8),
        "observation.joint_force": torch.randn(2, 2, 7),
        "observation.images.up_cam": torch.randn(2, 2, 3, 32, 32),
        "action_is_pad": torch.zeros(2, 16, dtype=torch.bool),
    }
    calibration_loss, calibration_metrics = policy(batch)
    assert calibration_loss == 0
    assert calibration_metrics["latent_calibrated"] == 1
    diffusion_loss, metrics = policy(batch)
    assert torch.isfinite(diffusion_loss) and "diffusion_loss" in metrics


def test_reactive_decode_uses_fixed_horizon_and_plan_age():
    policy = RDPPolicy(_config("diffusion"))
    captured = {}

    def fake_decode(latent, force):
        captured["force"] = force.clone()
        steps = torch.arange(force.shape[1], dtype=force.dtype).view(1, -1, 1)
        return steps.expand(force.shape[0], -1, 8)

    policy.tokenizer.decode = fake_decode
    force_history = torch.stack((torch.zeros(1, 7), torch.ones(1, 7)), dim=1)
    latent = torch.zeros(1, 4, 8)
    action_now = policy.decode_reactive_action(latent, force_history, plan_age=0)
    assert captured["force"].shape == (1, policy.config.horizon, 7)
    torch.testing.assert_close(
        action_now, torch.full((1, 8), policy.config.force_history_pre, dtype=torch.float32)
    )
    action_two_ticks_later = policy.decode_reactive_action(latent, force_history, plan_age=2)
    torch.testing.assert_close(
        action_two_ticks_later,
        torch.full((1, 8), policy.config.force_history_pre + 2, dtype=torch.float32),
    )


def test_diffusion_eval_does_not_calibrate_latent_stats():
    policy = RDPPolicy(_config("diffusion")).eval()
    batch = {
        ACTION: torch.randn(2, 16, 8),
        OBS_STATE: torch.randn(2, 2, 8),
        "observation.joint_force": torch.randn(2, 2, 7),
        "observation.images.up_cam": torch.randn(2, 2, 3, 32, 32),
        "action_is_pad": torch.zeros(2, 16, dtype=torch.bool),
    }
    with torch.no_grad():
        policy(batch)
    assert policy.latent_count.item() == 0
    assert not policy.latent_calibrated.item()
