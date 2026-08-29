import numpy as np

from lerobot.transport.openpi_websocket_client import packb, unpackb


def test_msgpack_numpy_roundtrip():
    source = {
        "state": np.arange(8, dtype=np.float32),
        "joint_force": np.arange(7, dtype=np.float32),
        "image": np.zeros((8, 8, 3), dtype=np.uint8),
    }
    decoded = unpackb(packb(source))
    for key, value in source.items():
        np.testing.assert_array_equal(decoded[key], value)
