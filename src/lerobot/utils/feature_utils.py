#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Lightweight feature-manipulation utilities.

These functions are intentionally kept free of heavy dependencies (e.g. the
HuggingFace ``datasets`` library) so that they can be imported from anywhere
in the codebase – including modules that are part of the *minimal* install –
without triggering the ``lerobot.datasets`` package guard.
"""

from typing import Any

import numpy as np

from lerobot.configs import FeatureType, PolicyFeature

from .constants import ACTION, DEFAULT_FEATURES, OBS_ENV_STATE, OBS_STR


def _validate_feature_names(features: dict[str, dict]) -> None:
    """Validate that feature names do not contain invalid characters.

    Args:
        features (dict): The LeRobot features dictionary.

    Raises:
        ValueError: If any feature name contains '/'.
    """
    invalid_features = {name: ft for name, ft in features.items() if "/" in name}
    if invalid_features:
        raise ValueError(f"Feature names should not contain '/'. Found '/' in '{invalid_features}'.")


def hw_to_dataset_features(
    hw_features: dict[str, type | tuple], prefix: str, use_video: bool = True
) -> dict[str, dict]:
    """Convert hardware-specific features to a LeRobot dataset feature dictionary.

    This function takes a dictionary describing hardware outputs (like joint states
    or camera image shapes) and formats it into the standard LeRobot feature
    specification. Single-channel cameras (shape ``(H, W, 1)``) are flagged as depth
    maps via ``info["is_depth_map"] = True``; three-channel cameras ``(H, W, 3)`` are
    treated as RGB.

    Args:
        hw_features (dict): Dictionary mapping feature names to their type (float for
            joints) or shape (tuple for images).
        prefix (str): The prefix to add to the feature keys (e.g., "observation"
            or "action").
        use_video (bool): If True, image features are marked as "video", otherwise "image".

    Returns:
        dict: A LeRobot features dictionary. Depth cameras carry ``info["is_depth_map"] = True``.
    """
    features = {}
    joint_fts = {
        key: ftype
        for key, ftype in hw_features.items()
        if ftype is float or (isinstance(ftype, PolicyFeature) and ftype.type != FeatureType.VISUAL)
    }
    # TODO(CarolinePascal): we should not rely on the shape to determine if a feature is a camera !
    cam_fts = {key: shape for key, shape in hw_features.items() if isinstance(shape, tuple)}

    if joint_fts and prefix == ACTION:
        features[prefix] = {
            "dtype": "float32",
            "shape": (len(joint_fts),),
            "names": list(joint_fts),
        }

    if joint_fts and prefix == OBS_STR:
        # Marvin adaptation: Separate state features (joint positions + gripper) from force feedback
        force_feedback_patterns = [".vel", ".torque", ".force"]
        force_feedback_cart_patterns = ["cart_force."]

        state_fts = {}
        force_fts = {}

        for key, ftype in joint_fts.items():
            # Check if this is a force feedback field
            is_force = any(pattern in key for pattern in force_feedback_patterns) or \
                       any(key.startswith(pattern) for pattern in force_feedback_cart_patterns)

            if is_force:
                force_fts[key] = ftype
            else:
                state_fts[key] = ftype

        # Create observation.state for joint positions and gripper
        if state_fts:
            features[f"{prefix}.state"] = {
                "dtype": "float32",
                "shape": (len(state_fts),),
                "names": list(state_fts),
            }

        # Create separate observation fields for each force feedback type
        if force_fts:
            # Group by force feedback type
            joint_vel = [k for k in force_fts if ".vel" in k]
            joint_torque = [k for k in force_fts if ".torque" in k]
            joint_force = [k for k in force_fts if ".force" in k]
            cart_force = [k for k in force_fts if k.startswith("cart_force.")]

            if joint_vel:
                features[f"{prefix}.joint_vel"] = {
                    "dtype": "float32",
                    "shape": (len(joint_vel),),
                    "names": joint_vel,
                }

            if joint_torque:
                features[f"{prefix}.joint_torque"] = {
                    "dtype": "float32",
                    "shape": (len(joint_torque),),
                    "names": joint_torque,
                }

            if joint_force:
                features[f"{prefix}.joint_force"] = {
                    "dtype": "float32",
                    "shape": (len(joint_force),),
                    "names": joint_force,
                }

            if cart_force:
                features[f"{prefix}.cart_force"] = {
                    "dtype": "float32",
                    "shape": (len(cart_force),),
                    "names": cart_force,
                }

    for key, shape in cam_fts.items():
        # Upstream 0.6.1: Enhanced camera support with depth map detection
        dtype = "video" if use_video else "image"
        if len(shape) == 3 and shape[2] in (1, 3):
            features[f"{prefix}.images.{key}"] = {
                "dtype": dtype,
                "shape": shape,
                "names": ["height", "width", "channels"],
                "info": {"is_depth_map": shape[2] == 1},
            }
        else:
            raise ValueError(
                f"Camera feature '{key}' has shape {shape}. "
                f"Expected a 3-tuple (H, W, C), e.g. (480, 640, 3) for RGB or (480, 640, 1) for depth."
            )

    _validate_feature_names(features)
    return features


def build_dataset_frame(
    ds_features: dict[str, dict], values: dict[str, Any], prefix: str
) -> dict[str, np.ndarray]:
    """Construct a single data frame from raw values based on dataset features.

    A "frame" is a dictionary containing all the data for a single timestep,
    formatted as numpy arrays according to the feature specification.

    Args:
        ds_features (dict): The LeRobot dataset features dictionary.
        values (dict): A dictionary of raw values from the hardware/environment.
        prefix (str): The prefix to filter features by (e.g., "observation"
            or "action").

    Returns:
        dict: A dictionary representing a single frame of data.
    """
    frame = {}
    for key, ft in ds_features.items():
        if key in DEFAULT_FEATURES or not key.startswith(prefix):
            continue
        elif ft["dtype"] == "float32" and len(ft["shape"]) == 1:
            frame[key] = np.array([values[name] for name in ft["names"]], dtype=np.float32)
        elif ft["dtype"] in ["image", "video"]:
            frame[key] = values[key.removeprefix(f"{prefix}.images.")]

    return frame


def dataset_to_policy_features(features: dict[str, dict]) -> dict[str, PolicyFeature]:
    """Convert dataset features to policy features.

    This function transforms the dataset's feature specification into a format
    that a policy can use, classifying features by type (e.g., visual, state,
    action) and ensuring correct shapes (e.g., channel-first for images).

    Args:
        features (dict): The LeRobot dataset features dictionary.

    Returns:
        dict: A dictionary mapping feature keys to `PolicyFeature` objects.

    Raises:
        ValueError: If an image feature does not have a 3D shape.
    """
    # TODO(aliberts): Implement "type" in dataset features and simplify this
    policy_features = {}
    for key, ft in features.items():
        shape = ft["shape"]
        if ft["dtype"] in ["image", "video"]:
            type = FeatureType.VISUAL
            if len(shape) != 3:
                raise ValueError(f"Number of dimensions of {key} != 3 (shape={shape})")
            else:
                names = ft["names"]
                # Backward compatibility for "channel" which is an error introduced in LeRobotDataset v2.0 for ported datasets.
                if names[2] in ["channel", "channels"]:  # (h, w, c) -> (c, h, w)
                    shape = (shape[2], shape[0], shape[1])
        elif key == OBS_ENV_STATE:
            type = FeatureType.ENV
        elif key.startswith(OBS_STR):
            type = FeatureType.STATE
        elif key.startswith(ACTION):
            type = FeatureType.ACTION
        else:
            continue

        policy_features[key] = PolicyFeature(
            type=type,
            shape=shape,
        )

    return policy_features


def combine_feature_dicts(*dicts: dict) -> dict:
    """Merge LeRobot grouped feature dicts.

    - For 1D numeric specs (dtype not image/video/string) with "names": we merge the names and recompute the shape.
    - For others (e.g. `observation.images.*`), the last one wins (if they are identical).

    Args:
        *dicts: A variable number of LeRobot feature dictionaries to merge.

    Returns:
        dict: A single merged feature dictionary.

    Raises:
        ValueError: If there's a dtype mismatch for a feature being merged.
    """
    out: dict = {}
    for d in dicts:
        for key, value in d.items():
            if not isinstance(value, dict):
                out[key] = value
                continue

            dtype = value.get("dtype")
            shape = value.get("shape")
            is_vector = (
                dtype not in ("image", "video", "string")
                and isinstance(shape, tuple)
                and len(shape) == 1
                and "names" in value
            )

            if is_vector:
                # Initialize or retrieve the accumulating dict for this feature key
                target = out.setdefault(key, {"dtype": dtype, "names": [], "shape": (0,)})
                # Ensure consistent data types across merged entries
                if "dtype" in target and dtype != target["dtype"]:
                    raise ValueError(f"dtype mismatch for '{key}': {target['dtype']} vs {dtype}")

                # Merge feature names: append only new ones to preserve order without duplicates
                seen = set(target["names"])
                for n in value["names"]:
                    if n not in seen:
                        target["names"].append(n)
                        seen.add(n)
                # Recompute the shape to reflect the updated number of features
                target["shape"] = (len(target["names"]),)
            else:
                # For images/videos and non-1D entries: override with the latest definition
                out[key] = value
    return out
