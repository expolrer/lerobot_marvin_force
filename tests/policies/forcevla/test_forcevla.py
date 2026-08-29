import torch
from torch import nn

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.forcevla.configuration_forcevla import ForceVLAConfig
from lerobot.policies.forcevla.modeling_forcevla import ForceVLAPolicy, ForceVLAPytorch
from lerobot.utils.constants import ACTION, OBS_STATE


def _config() -> ForceVLAConfig:
    return ForceVLAConfig(
        paligemma_variant="gemma_300m",
        action_expert_variant="gemma_300m",
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
            "observation.joint_force": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
            "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(8,))},
        chunk_size=4,
        n_action_steps=1,
        device="cpu",
    )


def _lightweight_core(config: ForceVLAConfig) -> ForceVLAPytorch:
    # Exercise the real suffix implementation without allocating PI0's 2B VLM.
    model = ForceVLAPytorch.__new__(ForceVLAPytorch)
    nn.Module.__init__(model)
    model.config = config
    model.rtc_processor = None
    model.gradient_checkpointing_enabled = False
    model.state_proj = nn.Linear(config.max_state_dim, 8)
    model.force_proj = nn.Linear(config.force_dim, 8)
    model.action_in_proj = nn.Linear(config.max_action_dim, 8)
    model.action_out_proj = nn.Linear(8, config.max_action_dim)
    model.action_time_mlp_in = nn.Linear(16, 8)
    model.action_time_mlp_out = nn.Linear(8, 8)
    return model


def test_force_token_changes_suffix_without_changing_action_shape():
    model = _lightweight_core(_config())
    state = torch.zeros(2, 32)
    actions = torch.randn(2, 4, 32)
    timestep = torch.rand(2)
    baseline, baseline_pad, baseline_att, _ = model.embed_suffix(state, actions, timestep)
    state[:, 8:15] = 1.0
    changed, changed_pad, changed_att, _ = model.embed_suffix(state, actions, timestep)
    assert baseline.shape == changed.shape == (2, 6, model.state_proj.out_features)
    assert baseline_pad.shape == changed_pad.shape == (2, 6)
    assert baseline_att.shape == changed_att.shape == (2, 6)
    torch.testing.assert_close(baseline[:, 0], changed[:, 0])
    assert not torch.allclose(baseline[:, 1], changed[:, 1])


def test_prepare_state_preserves_state_and_force_channel_order():
    config = _config()
    policy = ForceVLAPolicy.__new__(ForceVLAPolicy)
    nn.Module.__init__(policy)
    policy.config = config
    state = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    force = torch.arange(14, dtype=torch.float32).reshape(2, 7) + 100
    prepared = policy.prepare_state(
        {OBS_STATE: state, "observation.joint_force": force}
    )
    assert prepared.shape == (2, config.max_state_dim)
    torch.testing.assert_close(prepared[:, :8], state)
    torch.testing.assert_close(prepared[:, 8:15], force)


def test_prepare_state_rejects_non_finite_force():
    config = _config()
    policy = ForceVLAPolicy.__new__(ForceVLAPolicy)
    nn.Module.__init__(policy)
    policy.config = config
    force = torch.zeros(1, 7)
    force[0, 0] = float("nan")
    try:
        policy.prepare_state({OBS_STATE: torch.zeros(1, 8), "observation.joint_force": force})
    except ValueError as error:
        assert "non-finite" in str(error)
    else:
        raise AssertionError("ForceVLA accepted non-finite force")
