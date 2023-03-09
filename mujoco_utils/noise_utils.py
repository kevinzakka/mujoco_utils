import math

import numpy as np
from dm_env import specs
from mujoco import mjMINVAL


class OrnsteinUhlenbeckNoiseProcess:
    """Generates control noise using the Ornstein-Uhlenbeck process."""

    def __init__(
        self,
        control_timestep: float,
        action_spec: specs.Array,
        noise_std: float = 0.0,
        noise_rate: float = 0.0,
    ) -> None:
        if not 0 <= noise_std <= 2:
            raise ValueError("noise_std must be in [0, 2].")
        if not 0 <= noise_rate <= 2:
            raise ValueError("noise_rate must be in [0, 2].")

        self._rate = math.exp(-control_timestep / max(noise_rate, mjMINVAL))
        self._scale = noise_std * math.sqrt(1 - self._rate**2)
        self._nu = action_spec.shape[-1]
        self._dtype = action_spec.dtype

        self.reset()

    def reset(self) -> None:
        self._ctrlnoise = np.zeros((self._nu,), dtype=self._dtype)

    def sample(self) -> np.ndarray:
        self._ctrlnoise *= self._rate
        self._ctrlnoise += self._scale * np.random.randn(self._nu)
        return self._ctrlnoise
