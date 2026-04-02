import numpy as np
import pytensor.tensor as pt


class TransitionPrior:
    def __init__(self, V: int):
        self.V = V
        self.beta = self._uninformative_base()

    def _uninformative_base(self) -> np.ndarray:
        return np.full((self.V, self.V), 0.01)

    def _gaussian(self, x: np.ndarray, center: float, sigma: float) -> np.ndarray:
        return np.exp(-((x - center) ** 2) / (2 * sigma ** 2))

    def _exponential_decay(self, x: np.ndarray, steepness: float) -> np.ndarray:
        return np.exp(-steepness * x)

    def _normalize(self, arr: np.ndarray, weight: float, axis: int = None) -> np.ndarray:
        if axis is not None:
            return (arr / arr.sum(axis=axis, keepdims=True)) * weight
        return (arr / arr.sum()) * weight

    def prefer_small_intervals(self, weight: float = 5.0, sigma: float = 2.0):
        i_idx = np.arange(self.V)[:, None]
        j_idx = np.arange(self.V)[None, :]

        bump = self._gaussian(i_idx - j_idx, center=0, sigma=sigma)
        self.beta += self._normalize(bump, weight, axis=1)

        return self

    def prefer_low_register(self, weight: float = 5.0, center: float = None, sigma: float = None):
        center = center if center is not None else self.V / 4
        sigma = sigma if sigma is not None else self.V / 8

        j_idx = np.arange(self.V)
        bump_1d = self._gaussian(j_idx, center=center, sigma=sigma)
        bump_1d = self._normalize(bump_1d, weight)

        self.beta += bump_1d[None, :]

        return self

    def dim_high_register(self, steepness: float = 2.0):
        i_idx = np.arange(self.V)[:, None]
        j_idx = np.arange(self.V)[None, :]

        combined_pitch_level = (i_idx + j_idx) / (2 * max(1, self.V - 1))
        mask = self._exponential_decay(combined_pitch_level, steepness)

        self.beta = (self.beta - 1.0) * mask + 1.0

        return self

    def penalize_repeated_notes(self, scale: float = 0.1):
        np.fill_diagonal(self.beta, self.beta.diagonal() * scale)
        return self

    def offset(self, offset_value: float):
        self.beta += offset_value
        return self

    def get_beta(self) -> np.ndarray:
        return self.beta.copy()

    def reset(self):
        self.beta = self._uninformative_base()
        return self

    def __repr__(self):
        return (
            f"TransitionPrior(V={self.V}, "
            f"beta_min={self.beta.min():.2f}, "
            f"beta_max={self.beta.max():.2f})"
        )