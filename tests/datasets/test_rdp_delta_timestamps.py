from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies.rdp.configuration_rdp import RDPConfig


class _Metadata:
    fps = 30
    features = {
        "action": {},
        "observation.state": {},
        "observation.joint_force": {},
        "observation.images.up_cam": {},
    }


def test_rdp_tokenizer_uses_force_horizon_without_image_horizon():
    config = RDPConfig(training_stage="tokenizer")
    timestamps = resolve_delta_timestamps(config, _Metadata())
    assert len(timestamps["action"]) == 32
    assert timestamps["action"] == timestamps["observation.joint_force"]
    assert "observation.images.up_cam" not in timestamps


def test_rdp_diffusion_uses_two_slow_observations():
    config = RDPConfig(training_stage="diffusion")
    timestamps = resolve_delta_timestamps(config, _Metadata())
    assert len(timestamps["action"]) == 32
    assert timestamps["observation.state"] == [-2 / 30, 0]
    assert timestamps["observation.joint_force"] == [-2 / 30, 0]
    assert timestamps["observation.images.up_cam"] == [-2 / 30, 0]
