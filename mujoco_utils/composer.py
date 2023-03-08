"""Composer-related utilities."""

from dm_control import composer
import dm_env


class StaticPhysicsEnvironment(composer.Environment):
    """A composer environment with functionality to skip physics recompilation."""

    def __init__(self, recompile_physics: bool = True, **base_kwargs) -> None:
        """Constructor.

        Args:
            recompile_physics: Whether to recompile the physics between episodes. When
                set to False, `initialize_episode_mjcf` and `after_compile` steps are
                skipped. This can be useful for speeding up training when the physics
                are not changing between episodes.
            **base_kwargs: `composer.Environment` kwargs.
        """
        super().__init__(**base_kwargs)

        self._recompile_physics_active = recompile_physics
        self._physics_recompiled_once = False

    def _reset_attempt(self) -> dm_env.TimeStep:
        if self._recompile_physics_active or not self._physics_recompiled_once:
            self._hooks.initialize_episode_mjcf(self._random_state)
            self._recompile_physics_and_update_observables()
            self._physics_recompiled_once = True

        with self._physics.reset_context():
            self._hooks.initialize_episode(self._physics_proxy, self._random_state)

        self._observation_updater.reset(self._physics_proxy, self._random_state)
        self._reset_next_step = False

        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=None,
            discount=None,
            observation=self._observation_updater.get_observation(),
        )
