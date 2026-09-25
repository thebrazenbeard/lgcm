from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class CausalControlEnv:
    """Small bounded action-conditioned dynamical system for V0 experiments."""

    observation_dim = 2
    action_dim = 1

    def __init__(self, *, seed: int, noise_std: float = 0.0) -> None:
        if not np.isfinite(noise_std) or noise_std < 0:
            raise ValueError("noise_std must be finite and non-negative")
        self.seed = int(seed)
        self.noise_std = float(noise_std)
        rng = np.random.default_rng(seed)
        self._a = np.array([[0.55, 0.15], [-0.12, 0.50]], dtype=np.float64)
        self._a += rng.normal(0.0, 0.02, size=(2, 2))
        self._b = np.array([[0.48], [-0.34]], dtype=np.float64)
        self._b += rng.normal(0.0, 0.02, size=(2, 1))
        self._c = np.array([[0.7, -0.2], [0.25, 0.6]], dtype=np.float64)
        self._noise_rng = np.random.default_rng(seed + 1)
        self._state_rng = np.random.default_rng(seed + 2)
        self.state = self._state_rng.uniform(-0.25, 0.25, size=2).astype(np.float64)

    def _state_vector(self, state) -> NDArray[np.float64]:
        value = np.ascontiguousarray(state, dtype=np.float64)
        if value.shape != (2,) or not np.all(np.isfinite(value)):
            raise ValueError("state must be a finite vector with shape (2,)")
        if np.any(np.abs(value) > 1.0):
            raise ValueError("state must remain inside [-1, 1]")
        return value

    def _action_vector(self, action) -> NDArray[np.float64]:
        value = np.ascontiguousarray(action, dtype=np.float64)
        if value.shape != (1,) or not np.all(np.isfinite(value)):
            raise ValueError("action must be a finite vector with shape (1,)")
        if np.any(np.abs(value) > 1.0):
            raise ValueError("action must remain inside [-1, 1]")
        return value

    def transition(self, state, action) -> NDArray[np.float64]:
        s = self._state_vector(state)
        a = self._action_vector(action)
        interaction = np.array(
            [0.12 * s[0] * a[0], -0.10 * s[1] * a[0]],
            dtype=np.float64,
        )
        raw = self._a @ s + self._b @ a + 0.14 * np.sin(self._c @ s) + interaction
        return np.tanh(raw).astype(np.float64)

    def step(self, action) -> NDArray[np.float64]:
        next_state = self.transition(self.state, action)
        if self.noise_std:
            next_state = np.tanh(
                np.arctanh(np.clip(next_state, -0.999999, 0.999999))
                + self._noise_rng.normal(0.0, self.noise_std, size=2)
            )
        self.state = np.ascontiguousarray(next_state, dtype=np.float64)
        return self.state.copy()
