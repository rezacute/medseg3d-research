"""B1: Echo State Network (ESN) baseline.

Classical reservoir computing baseline for comparison with QRC.
"""

from typing import Any, Optional, cast

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import Ridge


class EchoStateNetwork:
    """Echo State Network for time-series forecasting.
    
    A classical reservoir computing approach using a random
    recurrent neural network with fixed weights.
    
    Attributes:
        n_reservoir: Number of reservoir neurons.
        spectral_radius: Spectral radius of reservoir weight matrix.
        leak_rate: Leaky integration coefficient.
        alpha: Ridge regression regularization.
    """
    
    def __init__(
        self,
        n_reservoir: int = 200,
        spectral_radius: float = 0.9,
        leak_rate: float = 0.3,
        alpha: float = 10.0,
        seed: int = 42,
    ):
        """Initialize the ESN.
        
        Args:
            n_reservoir: Number of reservoir neurons. Default: 200.
            spectral_radius: Spectral radius for stability. Default: 0.9.
            leak_rate: Leaky integration rate. Default: 0.3.
            alpha: Ridge regularization. Default: 10.0.
            seed: Random seed. Default: 42.
        """
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.alpha = alpha
        self.seed = seed
        
        self.W: Optional[NDArray[np.floating[Any]]] = None
        self.W_in: Optional[NDArray[np.floating[Any]]] = None
        self.readout: Optional[Ridge] = None
        self._input_dim: Optional[int] = None
        
    def _init_weights(self, input_dim: int) -> None:
        """Initialize reservoir weights."""
        rng = np.random.default_rng(self.seed)
        
        # Reservoir weights
        W = rng.standard_normal((self.n_reservoir, self.n_reservoir))
        # Scale to desired spectral radius
        eigenvalues = np.linalg.eigvals(W)
        W = W * (self.spectral_radius / np.max(np.abs(eigenvalues)))
        self.W = W
        
        # Input weights
        self.W_in = rng.uniform(-1, 1, (self.n_reservoir, input_dim))
        self._input_dim = input_dim
        
    def _compute_states(self, X: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """Compute reservoir states for input sequence."""
        assert self.W is not None
        assert self.W_in is not None
        T = len(X)
        states = np.zeros((T, self.n_reservoir))
        state = np.zeros(self.n_reservoir)

        for t in range(T):
            pre_activation = np.tanh(self.W_in @ X[t] + self.W @ state)
            state = (1 - self.leak_rate) * state + self.leak_rate * pre_activation
            states[t] = state

        return states
    
    def fit(self, X: NDArray[np.floating[Any]], y: NDArray[np.floating[Any]]) -> None:
        """Fit the ESN to training data.
        
        Args:
            X: Input features of shape (T, input_dim).
            y: Target values of shape (T,).
        """
        if self.W is None:
            self._init_weights(X.shape[1])
            
        states = self._compute_states(X)

        self.readout = Ridge(alpha=self.alpha)
        self.readout.fit(states, y)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the trained ESN.
        
        Args:
            X: Input features of shape (T, input_dim).
            
        Returns:
            Predictions of shape (T,).
        """
        states = self._compute_states(X)
        assert self.readout is not None
        return self.readout.predict(states)  # type: ignore[no-any-return]
    
    def get_states(self, X: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """Get reservoir states without prediction.

        Args:
            X: Input features of shape (T, input_dim).

        Returns:
            Reservoir states of shape (T, n_reservoir).
        """
        if self.W is None:
            self._init_weights(X.shape[1])
        return self._compute_states(X)
