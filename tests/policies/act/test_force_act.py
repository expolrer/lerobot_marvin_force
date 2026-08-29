import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE


def test_force_act_concatenates_force_in_both_state_projections():
    force_key = "observation.joint_force"
    config = ACTConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
            force_key: PolicyFeature(type=FeatureType.STATE, shape=(7,)),
            OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(2,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(8,))},
        force_feature_key=force_key,
        chunk_size=4,
        n_action_steps=1,
        dim_model=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_vae_encoder_layers=1,
        pretrained_backbone_weights=None,
    )
    policy = ACTPolicy(config)
    assert policy.model.encoder_robot_state_input_proj.in_features == 15
    assert policy.model.vae_encoder_robot_state_input_proj.in_features == 15

    batch = {
        OBS_STATE: torch.randn(2, 8),
        force_key: torch.randn(2, 7),
        OBS_ENV_STATE: torch.randn(2, 2),
        ACTION: torch.randn(2, 4, 8),
        "action_is_pad": torch.zeros(2, 4, dtype=torch.bool),
    }
    loss, _ = policy(batch)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_force_act_fails_closed_when_force_is_missing():
    force_key = "observation.joint_force"
    config = ACTConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
            force_key: PolicyFeature(type=FeatureType.STATE, shape=(7,)),
            OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(2,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(8,))},
        force_feature_key=force_key,
        chunk_size=2,
        n_action_steps=1,
        dim_model=16,
        n_heads=4,
        dim_feedforward=32,
        n_encoder_layers=1,
        n_vae_encoder_layers=1,
    )
    policy = ACTPolicy(config).eval()
    try:
        policy.predict_action_chunk({OBS_STATE: torch.randn(1, 8), OBS_ENV_STATE: torch.randn(1, 2)})
    except KeyError as error:
        assert force_key in str(error)
    else:
        raise AssertionError("Missing force must not be silently ignored")
