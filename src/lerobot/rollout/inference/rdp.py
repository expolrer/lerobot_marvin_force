#!/usr/bin/env python

"""Nonblocking slow/fast inference engine for RDP."""

from __future__ import annotations

import logging
import time
import traceback
from collections import deque
from contextlib import nullcontext
from copy import copy
from threading import Event, Lock, Thread

import torch

from lerobot.policies.rdp.modeling_rdp import RDPPolicy
from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline

from .base import InferenceEngine

logger = logging.getLogger(__name__)


class RDPInferenceEngine(InferenceEngine):
    """Plan a latent chunk slowly, then react with the latest force at 30 Hz."""

    def __init__(
        self,
        policy: RDPPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        dataset_features: dict,
        ordered_action_keys: list[str],
        task: str,
        device: str | None,
        robot_type: str,
    ) -> None:
        if not isinstance(policy, RDPPolicy):
            raise TypeError("--inference.type=rdp requires a policy.type=rdp checkpoint")
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._dataset_features = dataset_features
        self._ordered_action_keys = ordered_action_keys
        self._task = task
        self._device = torch.device(device or "cpu")
        self._robot_type = robot_type

        self._shutdown = Event()
        self._active = Event()
        self._plan_requested = Event()
        self._failed = Event()
        self._plan_lock = Lock()
        self._latent_lock = Lock()
        self._pending_plan: tuple[int, dict[str, torch.Tensor]] | None = None
        self._published_plan: tuple[int, torch.Tensor] | None = None
        self._thread: Thread | None = None
        self._slow_stream = torch.cuda.Stream(device=self._device) if self._device.type == "cuda" else None
        self.reset()

    @property
    def failed(self) -> bool:
        return self._failed.is_set()

    def start(self) -> None:
        self._shutdown.clear()
        self._active.set()
        self._thread = Thread(target=self._planner_loop, daemon=True, name="RDPPlanner")
        self._thread.start()
        logger.info("RDP slow planner started; fast force decoder remains on the control thread")

    def stop(self) -> None:
        self._shutdown.set()
        self._active.clear()
        self._plan_requested.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            if self._thread.is_alive():
                logger.warning("RDP planner did not stop within 3 seconds")
            self._thread = None

    def pause(self) -> None:
        self._active.clear()

    def resume(self) -> None:
        self._active.set()

    def reset(self) -> None:
        self._policy.reset()
        self._preprocessor.reset()
        self._postprocessor.reset()
        self._tick = 0
        self._slow_history: deque[dict[str, torch.Tensor]] = deque(
            maxlen=self._policy.config.slow_obs_steps
        )
        self._force_history: deque[tuple[int, torch.Tensor]] = deque(
            maxlen=self._policy.config.horizon + self._policy.config.force_history_pre
        )
        self._active_latent: torch.Tensor | None = None
        self._active_plan_tick = -1
        with self._plan_lock:
            self._pending_plan = None
        with self._latent_lock:
            self._published_plan = None

    def notify_observation(self, obs: dict) -> None:
        # The current dataset frame is processed in get_action so the normalizer
        # and force history are advanced exactly once per executed control tick.
        return None

    def _autocast(self):
        return (
            torch.autocast(device_type="cuda")
            if self._device.type == "cuda" and self._policy.config.use_amp
            else nullcontext()
        )

    def _snapshot(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        keys = [
            *self._policy.config.image_features,
            "observation.state",
            self._policy.config.force_feature_key,
        ]
        return {key: batch[key].detach() for key in keys}

    def _build_slow_batch(self) -> dict[str, torch.Tensor]:
        history = list(self._slow_history)
        while len(history) < self._policy.config.slow_obs_steps:
            history.insert(0, history[0])
        keys = history[0]
        return {key: torch.stack([item[key] for item in history], dim=1) for key in keys}

    def _request_plan(self) -> None:
        if not self._slow_history:
            return
        with self._plan_lock:
            if self._pending_plan is not None:
                return
            self._pending_plan = (self._tick, self._build_slow_batch())
        self._plan_requested.set()

    def _consume_published_plan(self) -> None:
        with self._latent_lock:
            published = self._published_plan
            self._published_plan = None
        if published is not None:
            self._active_plan_tick, self._active_latent = published

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        if obs_frame is None or self.failed or not self._active.is_set():
            return None
        observation = copy(obs_frame)
        with torch.inference_mode(), self._autocast():
            observation = prepare_observation_for_inference(
                observation, self._device, self._task, self._robot_type
            )
            batch = self._preprocessor(observation)
            snapshot = self._snapshot(batch)
            self._slow_history.append(snapshot)
            force = batch[self._policy.config.force_feature_key].detach()
            self._force_history.append((self._tick, force))
            self._consume_published_plan()

            should_plan = (
                self._active_latent is None
                or self._tick % self._policy.config.slow_interval == 0
                or self._tick - self._active_plan_tick
                >= self._policy.config.horizon - self._policy.config.force_history_pre
            )
            if should_plan:
                self._request_plan()

            self._tick += 1
            if self._active_latent is None:
                return None

            first_force_tick = self._active_plan_tick - self._policy.config.force_history_pre
            forces = [value for tick, value in self._force_history if tick >= first_force_tick]
            if not forces:
                return None
            forces = forces[-self._policy.config.horizon :]
            force_history = torch.stack(forces, dim=1)
            plan_age = max(0, self._tick - 1 - self._active_plan_tick)
            normalized_action = self._policy.decode_reactive_action(
                self._active_latent, force_history, plan_age=plan_age
            )
            action = self._postprocessor(normalized_action).squeeze(0).cpu()

        if not torch.isfinite(action).all():
            logger.error("RDP produced a non-finite action; holding the robot")
            return None
        action_dict = make_robot_action(action, self._dataset_features)
        return torch.tensor([action_dict[key] for key in self._ordered_action_keys])

    def _planner_loop(self) -> None:
        try:
            while not self._shutdown.is_set():
                self._plan_requested.wait(timeout=0.1)
                self._plan_requested.clear()
                if self._shutdown.is_set():
                    return
                if not self._active.is_set():
                    continue
                with self._plan_lock:
                    pending = self._pending_plan
                    self._pending_plan = None
                if pending is None:
                    continue
                plan_tick, slow_batch = pending
                started = time.perf_counter()
                with torch.inference_mode(), self._autocast():
                    stream_context = (
                        torch.cuda.stream(self._slow_stream)
                        if self._slow_stream is not None
                        else nullcontext()
                    )
                    with stream_context:
                        latent = self._policy.plan_latent(slow_batch)
                    if self._slow_stream is not None:
                        self._slow_stream.synchronize()
                with self._latent_lock:
                    self._published_plan = (plan_tick, latent.detach())
                logger.debug("RDP slow plan latency %.3fs", time.perf_counter() - started)
        except Exception as error:
            logger.error("Fatal RDP planner error: %s", error)
            logger.error(traceback.format_exc())
            self._failed.set()
