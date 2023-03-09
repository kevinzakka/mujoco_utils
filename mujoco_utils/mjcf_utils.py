"""MJCF-related utilities."""

from typing import List

import numpy as np
from dm_control import mjcf

from mujoco_utils import types


def safe_find_all(
    root: types.MjcfRootElement,
    feature_name: str,
    immediate_children_only: bool = False,
    exclude_attachments: bool = False,
) -> List[mjcf.Element]:
    """Find all given elements or throw an error if none are found."""
    features = root.find_all(feature_name, immediate_children_only, exclude_attachments)
    if not features:
        raise ValueError(f"No {feature_name} found in the MJCF model.")
    return features


def safe_find(
    root: types.MjcfRootElement,
    namespace: str,
    identifier: str,
) -> mjcf.Element:
    """Find the given element or throw an error if not found."""
    feature = root.find(namespace, identifier)
    if feature is None:
        raise ValueError(f"{namespace} with the specified {identifier} not found.")
    return feature


def attach_entities(
    parent_entity: types.MjcfRootElement,
    child_entity: types.MjcfRootElement,
    site_name: str,
) -> None:
    physics = mjcf.Physics.from_mjcf_model(child_entity)

    attachment_site = safe_find(parent_entity, "site", site_name)

    # Expand the ctrl and qpos keyframes to account for the child hand DoFs.
    parent_key = parent_entity.find("key", "home")
    if parent_key is not None:
        child_key = child_entity.find("key", "home")
        if child_key is None:
            parent_key.ctrl = np.concatenate(
                [parent_key.ctrl, np.zeros(physics.model.nu)]
            )
            parent_key.qpos = np.concatenate(
                [parent_key.qpos, np.zeros(physics.model.nq)]
            )
        else:
            parent_key.ctrl = np.concatenate([parent_key.ctrl, child_key.ctrl])
            parent_key.qpos = np.concatenate([parent_key.qpos, child_key.qpos])

    attachment_site.attach(child_entity)
