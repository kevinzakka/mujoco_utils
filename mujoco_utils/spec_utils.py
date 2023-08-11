"""Utilities for creating dm_env specs from MuJoCo models."""

from typing import Sequence

import mujoco
import numpy as np
from dm_env import specs


def create_action_spec(physics, actuators, prefix: str = "") -> specs.BoundedArray:
    """Creates an action spec for a list of MJCF actuators."""
    num_actuators = len(actuators)
    actuator_names = [
        f"{prefix}{act.name}" if act.name else f"{prefix}{i}"
        for i, act in enumerate(actuators)
    ]
    control_range = physics.bind(actuators).ctrlrange
    is_limited = physics.bind(actuators).ctrllimited.astype(bool)
    minima = np.full(num_actuators, fill_value=-mujoco.mjMAXVAL, dtype=np.float64)
    maxima = np.full(num_actuators, fill_value=mujoco.mjMAXVAL, dtype=np.float64)
    minima[is_limited], maxima[is_limited] = control_range[is_limited].T
    return specs.BoundedArray(
        shape=(num_actuators,),
        dtype=np.float64,
        minimum=minima,
        maximum=maxima,
        name="\t".join(actuator_names),
    )


# Reference: https://github.com/deepmind/dm_robotics/blob/main/py/agentflow/spec_utils.py
def merge_specs(spec_list: Sequence[specs.BoundedArray]) -> specs.BoundedArray:
    """Merges a list of `BoundedArray` specs into one."""
    # Check all specs are flat.
    for spec in spec_list:
        if len(spec.shape) > 1:
            raise ValueError("Not merging multi-dimensional spec: {}".format(spec))

    # Filter out no-op specs with no actuators.
    spec_list = [spec for spec in spec_list if spec.shape and spec.shape[0]]
    dtype = np.result_type(*[spec.dtype for spec in spec_list])

    num_actions = 0
    name = ""
    mins = np.array([], dtype=dtype)
    maxs = np.array([], dtype=dtype)

    for i, spec in enumerate(spec_list):
        num_actions += spec.shape[0]
        if name:
            name += "\t"
        name += spec.name or f"spec_{i}"
        mins = np.concatenate([mins, spec.minimum])
        maxs = np.concatenate([maxs, spec.maximum])

    return specs.BoundedArray(
        shape=(num_actions,), dtype=dtype, minimum=mins, maximum=maxs, name=name
    )
