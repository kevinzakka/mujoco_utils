"""Physics-related utilities."""

from mujoco_utils import types

from dm_control import mjcf
import math


def compensate_gravity(model: types.MjcfRootElement) -> None:
    """Applies gravity compensation to the all bodies in the model.

    Args:
        model: The MJCF model.
    """
    for body in model.find_all("body"):
        # A value of 1.0 creates an upward force equal to the body’s weight and
        # compensates for gravity exactly.
        body.gravcomp = 1.0


def get_critical_damping_from_stiffness(
    stiffness: float, joint_name: str, model: types.MjcfRootElement
) -> float:
    """Compute the critical damping coefficient for a given stiffness.

    Args:
        stiffness: The stiffness coefficient.
        joint_name: The name of the joint to compute the critical damping for.
        model: The MJCF model.

    Returns:
        The critical damping coefficient.
    """
    physics = mjcf.Physics.from_mjcf_model(model)
    joint_id = physics.named.model.jnt_qposadr[joint_name]
    joint_mass = physics.model.dof_M0[joint_id]
    return 2 * math.sqrt(joint_mass * stiffness)
