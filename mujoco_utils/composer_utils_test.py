"""Tests for composer_utils.py."""

from absl.testing import absltest
from dm_control import mjcf

from mujoco_utils import composer_utils


class ArenaTest(absltest.TestCase):
    def test_compiles_and_steps(self) -> None:
        arena = composer_utils.Arena()
        physics = mjcf.Physics.from_mjcf_model(arena.mjcf_model)
        physics.step()


if __name__ == "__main__":
    absltest.main()
