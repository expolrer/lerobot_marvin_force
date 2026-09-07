from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE))

from manifeel_adapter.protocol import ProtocolError, decode_message, encode_message  # noqa: E402
from manifeel_adapter.zmq_policy import ZmqLeRobotPolicy  # noqa: E402


def test_protocol_roundtrip_and_size_validation() -> None:
    source = np.arange(12, dtype=np.float32).reshape(2, 6)
    frames = encode_message(command="act_result", arrays={"action": source})
    header, arrays = decode_message([bytes(frame) for frame in frames], max_message_bytes=4096)
    assert header["command"] == "act_result"
    np.testing.assert_array_equal(arrays["action"], source)

    corrupted = [bytes(frame) for frame in frames]
    corrupted[1] = corrupted[1][:-4]
    with pytest.raises(ProtocolError, match="Byte count mismatch"):
        decode_message(corrupted, max_message_bytes=4096)


def test_simulator_chw_force_is_restored_to_training_hwc_order() -> None:
    policy = ZmqLeRobotPolicy.__new__(ZmqLeRobotPolicy)
    policy.config = {
        "wrist_source_key": "wrist",
        "tactile_source_key": "tactile_force_field_right",
        "wrist_target_key": "observation.images.wrist",
        "state_target_key": "observation.state",
        "tactile_target_key": "observation.tactile_force",
        "image_channels": 3,
        "image_height": 256,
        "image_width": 256,
        "state_dim": 7,
        "tactile_dim": 420,
        "action_dim": 6,
    }
    policy.model = {"use_force": True}

    captured: dict[str, np.ndarray] = {}

    def fake_request(command, arrays=None, fields=None):
        assert command == "act"
        assert fields == {"batch_size": 1}
        captured.update(arrays)
        return {"command": "act_result"}, {"action": np.zeros((1, 6), dtype=np.float32)}

    policy._request = fake_request
    force_hwc = np.arange(420, dtype=np.float32).reshape(10, 14, 3)
    force_chw = np.transpose(force_hwc, (2, 0, 1))
    observations = {
        "wrist": torch.zeros(1, 1, 3, 256, 256),
        "state": torch.zeros(1, 1, 7),
        "tactile_force_field_right": torch.from_numpy(force_chw.copy())[None, None],
    }

    result = policy.predict_action(observations)

    assert result["action"].shape == (1, 1, 6)
    np.testing.assert_array_equal(
        captured["observation.tactile_force"], force_hwc.reshape(1, 420)
    )
