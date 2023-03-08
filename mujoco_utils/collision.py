"""Collision-related utilities."""

from itertools import product
from typing import Sequence

from mujoco_utils import types

# Default minimum distance between two geoms for them to be considered in collision.
_DEFAULT_COLLISION_MARGIN: float = 1e-8


def has_collision(
    physics,
    collision_geom_prefix_1: Sequence[str],
    collision_geom_prefix_2: Sequence[str],
    margin: float = _DEFAULT_COLLISION_MARGIN,
) -> bool:
    for contact in physics.data.contact:
        if contact.dist > margin:
            continue

        geom1_name = physics.model.id2name(contact.geom1, "geom")
        geom2_name = physics.model.id2name(contact.geom2, "geom")

        for pair in product(collision_geom_prefix_1, collision_geom_prefix_2):
            if (geom1_name.startswith(pair[0]) and geom2_name.startswith(pair[1])) or (
                geom2_name.startswith(pair[0]) and geom1_name.startswith(pair[1])
            ):
                return True

    return False


def disable_geom_collisions(geom: types.MjcfElement) -> None:
    """Disables collisions with the given geom."""
    geom.contype = 0
    geom.conaffinity = 0


def disable_body_collisions(body: types.MjcfElement) -> None:
    """Disables collisions with the given body."""
    for geom in body.find_all("geom"):
        disable_geom_collisions(geom)
