"""Policy-shaped ZMQ proxy consumed by ManiFeel's official runner."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import zmq

from .protocol import ProtocolError, configure_socket, request


class ZmqLeRobotPolicy:
    """Expose ``predict_action`` while inference runs in the LeRobot environment."""

    def __init__(self, config: dict[str, Any], model_name: str):
        self.config = config
        self.model_name = model_name
        try:
            self.model = config["models"][model_name]
        except KeyError as error:
            raise ValueError(f"Unknown evaluation model: {model_name}") from error
        # The official runner only uses these to move observations before
        # predict_action.  Keeping the proxy on CPU avoids coupling IsaacGym's
        # CUDA context to the separate LeRobot inference process.
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self._context = zmq.Context.instance()
        self._socket = self._new_socket()
        self.server_info, _ = self._request("ping")
        if self.server_info.get("command") != "pong":
            raise ProtocolError(f"Unexpected ping response: {self.server_info}")
        if self.server_info.get("policy_type") != self.model["policy_type"]:
            raise ProtocolError(
                "Policy server type does not match evaluate.yaml: "
                f"{self.server_info.get('policy_type')!r} != {self.model['policy_type']!r}"
            )
        if bool(self.server_info.get("use_force")) != bool(self.model["use_force"]):
            raise ProtocolError("Policy server force-input contract does not match evaluate.yaml")
        if int(self.server_info.get("action_dim", -1)) != int(self.config["action_dim"]):
            raise ProtocolError("Policy server action dimension does not match evaluate.yaml")

    @property
    def endpoint(self) -> str:
        return f"tcp://{self.config['bind_host']}:{int(self.model['port'])}"

    def _new_socket(self) -> zmq.Socket:
        socket = self._context.socket(zmq.REQ)
        configure_socket(
            socket,
            timeout_ms=int(self.config["timeout_ms"]),
            max_message_bytes=int(self.config["max_message_bytes"]),
        )
        socket.connect(self.endpoint)
        return socket

    def _request(
        self,
        command: str,
        arrays: dict[str, np.ndarray] | None = None,
        fields: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        try:
            return request(
                self._socket,
                command=command,
                arrays=arrays,
                fields={"model": self.model_name, **(fields or {})},
                max_message_bytes=int(self.config["max_message_bytes"]),
            )
        except zmq.ZMQError as error:
            # REQ sockets enter an unusable state after a timeout. Recreate it
            # so a subsequent explicit retry or cleanup is well-defined.
            self._socket.close(linger=0)
            self._socket = self._new_socket()
            raise RuntimeError(f"ZMQ request {command!r} failed at {self.endpoint}: {error}") from error

    @staticmethod
    def _numpy(value: Any, name: str) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        array = np.asarray(value, dtype=np.float32)
        if not np.isfinite(array).all():
            raise ValueError(f"Observation {name} contains NaN or infinity")
        return np.ascontiguousarray(array)

    @staticmethod
    def _latest(array: np.ndarray, name: str, sample_rank: int) -> np.ndarray:
        if array.ndim == sample_rank + 2:  # [B,T,...]
            if array.shape[1] < 1:
                raise ValueError(f"Observation history for {name} is empty")
            return array[:, -1]
        if array.ndim == sample_rank + 1:  # [B,...]
            return array
        raise ValueError(
            f"Observation {name} must be [B,T,...] or [B,...], got {array.shape}"
        )

    def reset(self) -> None:
        header, _ = self._request("reset")
        if header.get("command") != "reset_ok":
            raise ProtocolError(f"Unexpected reset response: {header}")

    def predict_action(self, obs_dict: dict[str, Any]) -> dict[str, torch.Tensor]:
        wrist_key = self.config["wrist_source_key"]
        state_key = "state"
        if wrist_key not in obs_dict or state_key not in obs_dict:
            raise KeyError(f"Runner observation needs {wrist_key!r} and 'state': {sorted(obs_dict)}")
        wrist = self._latest(self._numpy(obs_dict[wrist_key], wrist_key), wrist_key, 3)
        state = self._latest(self._numpy(obs_dict[state_key], state_key), state_key, 1)
        expected_wrist = (
            int(self.config["image_channels"]),
            int(self.config["image_height"]),
            int(self.config["image_width"]),
        )
        if tuple(wrist.shape[1:]) != expected_wrist:
            raise ValueError(f"Wrist RGB expected [B,{expected_wrist}], got {wrist.shape}")
        if state.shape[1:] != (int(self.config["state_dim"]),):
            raise ValueError(f"State expected [B,{self.config['state_dim']}], got {state.shape}")
        if wrist.shape[0] != state.shape[0]:
            raise ValueError("Wrist and state batch sizes differ")
        # ManiFeel publishes normalized RGB.  A small tolerance accommodates
        # interpolation roundoff while still detecting uint8 or corrupted data.
        if wrist.size and (float(wrist.min()) < -1e-4 or float(wrist.max()) > 1.0001):
            raise ValueError("Wrist RGB must be float32 in [0,1]")

        arrays = {
            self.config["wrist_target_key"]: wrist,
            self.config["state_target_key"]: state,
        }
        if bool(self.model["use_force"]):
            force_key = self.config["tactile_source_key"]
            if force_key not in obs_dict:
                raise KeyError(f"Force-conditioned model requires runner observation {force_key!r}")
            force = self._latest(self._numpy(obs_dict[force_key], force_key), force_key, 3)
            expected_force = (state.shape[0], 3, 10, 14)
            if force.shape != expected_force:
                raise ValueError(
                    f"Force field must be {expected_force} "
                    f"({self.config['tactile_dim']} values), got {force.shape}"
                )
            # The simulator wrapper returns CHW, while the source Zarr and the
            # converter use HWC. Restore HWC before C-order flattening so online
            # observations exactly match training feature order.
            force_hwc = np.transpose(force, (0, 2, 3, 1))
            arrays[self.config["tactile_target_key"]] = np.ascontiguousarray(
                force_hwc.reshape(force.shape[0], int(self.config["tactile_dim"])),
                dtype=np.float32,
            )

        header, response = self._request(
            "act", arrays=arrays, fields={"batch_size": int(state.shape[0])}
        )
        if header.get("command") != "act_result" or "action" not in response:
            raise ProtocolError(f"Malformed act response: {header}")
        action = response["action"]
        expected_action = (state.shape[0], int(self.config["action_dim"]))
        if action.shape != expected_action or not np.isfinite(action).all():
            raise ProtocolError(f"Expected finite action {expected_action}, got {action.shape}")
        # MultiStepWrapper expects [B,n_action_steps,6].  Values remain the
        # official normalized 6-D EEF delta; no joint-space conversion occurs.
        return {"action": torch.from_numpy(action[:, None, :].copy())}

    def close(self) -> None:
        self._socket.close(linger=0)
