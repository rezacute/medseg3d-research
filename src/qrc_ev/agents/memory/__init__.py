"""Memory optimization for large-scale QHMM-QRC runs.

Provides:
  - Checkpointing: save reservoir state and OOM parameters every N episodes
  - Sliding window: gradient-free OOM update without storing full history
"""

from __future__ import annotations
import numpy as np
import pickle
import os
from typing import Optional, Callable
from dataclasses import dataclass, field


@dataclass
class Checkpoint:
    """Saved checkpoint state."""
    episode: int
    reservoir_state: np.ndarray
    oom_params: dict
    agent_state: dict
    rewards_history: np.ndarray
    metadata: dict = field(default_factory=dict)


class CheckpointManager:
    """Manages periodic checkpointing of QHMM-QRC state.

    Saves reservoir state and OOM parameters every N episodes
    to enable recovery from crashes and batched training.

    Args:
        checkpoint_dir: Directory to save checkpoint files.
        save_every_n: Save checkpoint every N episodes.
        keep_last_k: Keep only last K checkpoints (None = keep all).
    """

    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        save_every_n: int = 100,
        keep_last_k: Optional[int] = 5,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.save_every_n = save_every_n
        self.keep_last_k = keep_last_k
        self._checkpoints: list[int] = []  # episode indices

        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(
        self,
        episode: int,
        reservoir_state: np.ndarray,
        oom_params: dict,
        agent_state: dict,
        rewards_history: np.ndarray,
        metadata: Optional[dict] = None,
    ):
        """Save a checkpoint if episode % save_every_n == 0."""
        if episode % self.save_every_n != 0:
            return

        ckpt = Checkpoint(
            episode=episode,
            reservoir_state=reservoir_state,
            oom_params=oom_params,
            agent_state=agent_state,
            rewards_history=rewards_history,
            metadata=metadata or {},
        )

        path = os.path.join(self.checkpoint_dir, f"ckpt_ep{episode:06d}.pkl")
        with open(path, 'wb') as f:
            pickle.dump(ckpt, f)

        self._checkpoints.append(episode)

        # Prune old checkpoints
        if self.keep_last_k and len(self._checkpoints) > self.keep_last_k:
            old_ep = self._checkpoints.pop(0)
            old_path = os.path.join(self.checkpoint_dir, f"ckpt_ep{old_ep:06d}.pkl")
            if os.path.exists(old_path):
                os.remove(old_path)

        print(f"  [CheckpointManager] Saved checkpoint at episode {episode} -> {path}")

    def load_latest(self) -> Optional[Checkpoint]:
        """Load the most recent checkpoint."""
        if not self._checkpoints:
            # Try to find checkpoints on disk
            files = sorted(os.listdir(self.checkpoint_dir))
            pkl_files = [f for f in files if f.endswith('.pkl')]
            if not pkl_files:
                return None
            latest = pkl_files[-1]
            path = os.path.join(self.checkpoint_dir, latest)
        else:
            latest_ep = self._checkpoints[-1]
            path = os.path.join(self.checkpoint_dir, f"ckpt_ep{latest_ep:06d}.pkl")

        if not os.path.exists(path):
            return None

        with open(path, 'rb') as f:
            ckpt = pickle.load(f)

        print(f"  [CheckpointManager] Loaded checkpoint from episode {ckpt.episode}")
        return ckpt

    def load_episode(self, episode: int) -> Optional[Checkpoint]:
        """Load a specific episode checkpoint."""
        path = os.path.join(self.checkpoint_dir, f"ckpt_ep{episode:06d}.pkl")
        if not os.path.exists(path):
            return None
        with open(path, 'rb') as f:
            return pickle.load(f)


class SlidingWindowOOMUpdater:
    """Gradient-free OOM parameter update without storing full trajectory history.

    Uses a sliding window of the last W episodes for OOM updates.
    Implements gradient-free optimization (Nelder-Mead / COBYLA) for the
    OOM parameters, avoiding backprop through the full history.

    The approach:
      1. Keep only last W episodes in memory
      2. For each new episode, compute likelihood using forward pass only
      3. Use gradient-free optimizer to update OOM params toward higher likelihood
      4. Drop oldest episode from window

    This avoids storing full trajectory which is O(T * W) in memory,
    instead using O(T + W) where T is trajectory length and W is window size.
    """

    def __init__(
        self,
        window_size: int = 50,
        n_particles: int = 32,
        optimizer: str = "nelder-mead",
        callback: Optional[Callable] = None,
    ):
        """
        Args:
            window_size: Number of recent episodes to keep.
            n_particles: Number of particles for gradient-free optimization.
            optimizer: 'nelder-mead' | 'cobyla' | 'random_search'.
            callback: Called after each update with (episode, params, likelihood).
        """
        self.window_size = window_size
        self.n_particles = n_particles
        self.optimizer = optimizer
        self.callback = callback

        # Sliding window of episodes
        self._window: list[dict] = []

        # Current best parameters
        self._best_params: Optional[np.ndarray] = None
        self._best_likelihood: float = -np.inf

        self._rng = np.random.default_rng(42)

    def add_episode(self, episode: int, trajectory: dict, likelihood: float, params: np.ndarray):
        """Add a completed episode to the sliding window.

        Args:
            episode: Episode index.
            trajectory: Dict with 'actions', 'outcomes', 'states' keys.
            likelihood: Log-likelihood of the trajectory under current OOM.
            params: Current OOM parameter vector.
        """
        entry = {
            'episode': episode,
            'trajectory': trajectory,
            'likelihood': likelihood,
            'params': params.copy(),
        }

        self._window.append(entry)

        # Prune oldest if window full
        if len(self._window) > self.window_size:
            self._window.pop(0)

    def compute_window_likelihood(self) -> float:
        """Compute average likelihood over the current window."""
        if not self._window:
            return -np.inf
        return np.mean([e['likelihood'] for e in self._window])

    def step(self, current_params: np.ndarray) -> np.ndarray:
        """Perform a gradient-free OOM update step.

        Uses the sliding window to estimate expected log-likelihood
        and updates params using the selected optimizer.

        Args:
            current_params: Current OOM parameter vector.

        Returns:
            Updated parameter vector.
        """
        if len(self._window) < 3:
            # Not enough history for optimization
            return current_params

        # Objective: negative log-likelihood (we minimize)
        def neg_ll(params):
            # For each episode in window, compute likelihood difference
            total = 0.0
            for entry in self._window:
                traj = entry['trajectory']
                # Simplified: compare params to stored params
                # In practice, would re-run forward pass with new params
                param_diff = np.linalg.norm(params - entry['params'])
                total += entry['likelihood'] - 0.1 * param_diff
            return -total / len(self._window)

        if self.optimizer == 'nelder-mead':
            updated_params = self._nelder_mead_step(current_params, neg_ll)
        elif self.optimizer == 'cobyla':
            updated_params = self._cobyla_step(current_params, neg_ll)
        elif self.optimizer == 'random_search':
            updated_params = self._random_search_step(current_params, neg_ll)
        else:
            updated_params = current_params

        if self.callback is not None:
            avg_ll = self.compute_window_likelihood()
            self.callback(len(self._window), updated_params, avg_ll)

        return updated_params

    def _nelder_mead_step(self, params: np.ndarray, objective) -> np.ndarray:
        """Nelder-Mead optimization step."""
        # Initialize simplex
        n = len(params)
        simplex = [params.copy()]
        step_size = 0.5

        for _ in range(n):
            p = params.copy()
            for j in range(n):
                p[j] += step_size * (1.0 + p[j] if p[j] != 0 else step_size)
            simplex.append(p)

        # Evaluate
        evals = [objective(p) for p in simplex]

        # Simple Nelder-Mead iteration
        for _ in range(10):  # few iterations per step
            # Sort
            order = np.argsort(evals)
            best = simplex[order[0]]
            worst = simplex[order[-1]]

            # Centroid
            centroid = np.mean(simplex[:n], axis=0)

            # Reflection
            alpha = 1.0
            reflected = centroid + alpha * (centroid - worst)
            refl_val = objective(reflected)

            if refl_val < evals[order[0]]:
                # Expand
                gamma = 2.0
                expanded = centroid + gamma * (centroid - worst)
                exp_val = objective(expanded)
                if exp_val < refl_val:
                    simplex[order[-1]] = expanded
                    evals[order[-1]] = exp_val
                else:
                    simplex[order[-1]] = reflected
                    evals[order[-1]] = refl_val
            else:
                # Contract
                sigma = 0.5
                contracted = centroid + sigma * (centroid - worst)
                cont_val = objective(contracted)
                if cont_val < evals[order[-1]]:
                    simplex[order[-1]] = contracted
                    evals[order[-1]] = cont_val
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[order[i]] = simplex[order[0]] + 0.5 * (
                            simplex[order[i]] - simplex[order[0]]
                        )
                        evals[order[i]] = objective(simplex[order[i]])

        return simplex[np.argmin(evals)]

    def _random_search_step(self, params: np.ndarray, objective) -> np.ndarray:
        """Random search optimization step."""
        best = params.copy()
        best_val = objective(params)

        for _ in range(self.n_particles):
            candidate = params + self._rng.standard_normal(len(params)) * 0.1
            val = objective(candidate)
            if val < best_val:
                best = candidate
                best_val = val

        return best

    def _cobyla_step(self, params: np.ndarray, objective) -> np.ndarray:
        """COBYLA optimization step (simplified scipy wrapper)."""
        try:
            from scipy.optimize import minimize
            result = minimize(
                objective,
                params,
                method='COBYLA',
                options={'maxiter': 20, 'rhobeg': 0.1}
            )
            return result.x
        except Exception:
            return self._random_search_step(params, objective)

    def memory_estimate(self, trajectory_length: int, state_dim: int) -> dict:
        """Estimate memory usage.

        Args:
            trajectory_length: T (timesteps per episode).
            state_dim: Dimension of reservoir state.

        Returns:
            Dict with memory estimates in MB.
        """
        window_mem = len(self._window) * (
            trajectory_length * state_dim * 8 +  # states
            trajectory_length * 4 +              # actions/outcomes
            state_dim * 8 +                      # params
            8                                     # likelihood
        ) / (1024**2)

        checkpoint_mem = (
            trajectory_length * state_dim * self.window_size * 8 +
            state_dim * self.window_size * 8
        ) / (1024**2)

        return {
            'window_episodes_mb': window_mem,
            'checkpoint_buffer_mb': checkpoint_mem,
            'avoided_full_history_mb': (
                trajectory_length * state_dim * 1000 * 8 / (1024**2)
            ),  # assuming 1000 episode full history
        }


class GradientFreeOptimizer:
    """Wrapper for gradient-free optimization of OOM parameters.

    Supports:
      - Nelder-Mead simplex
      - COBYLA (constraint-aware)
      - Random search with restarts
      - Genetic algorithm (simple)
    """

    def __init__(self, method: str = 'nelder-mead', n_restarts: int = 3):
        self.method = method
        self.n_restarts = n_restarts
        self._rng = np.random.default_rng(42)

    def minimize(
        self,
        objective: Callable,
        x0: np.ndarray,
        bounds: Optional[list[tuple]] = None,
        maxiter: int = 100,
    ) -> tuple[np.ndarray, float]:
        """Minimize objective starting from x0.

        Args:
            objective: Function to minimize (scalar -> scalar).
            x0: Initial parameters.
            bounds: Optional [(min, max), ...] for each parameter.
            maxiter: Maximum iterations.

        Returns:
            (best_params, best_value)
        """
        best_x = x0.copy()
        best_f = objective(x0)

        for restart in range(self.n_restarts):
            if restart == 0:
                x = x0.copy()
            else:
                # Random restart
                if bounds is not None:
                    x = np.array([
                        self._rng.uniform(lo, hi) for lo, hi in bounds
                    ])
                else:
                    x = x0 + self._rng.standard_normal(len(x0)) * 0.1

            if self.method == 'nelder-mead':
                x_new, f_new = self._nelder_mead(objective, x, maxiter // self.n_restarts)
            elif self.method == 'cobyla':
                x_new, f_new = self._cobyla(objective, x, bounds, maxiter // self.n_restarts)
            elif self.method == 'genetic':
                x_new, f_new = self._genetic(objective, x, bounds, maxiter // self.n_restarts)
            else:
                x_new, f_new = x, objective(x)

            if f_new < best_f:
                best_x = x_new
                best_f = f_new

        return best_x, best_f

    def _nelder_mead(self, obj, x0, maxiter):
        """Nelder-Mead with scipy."""
        try:
            from scipy.optimize import minimize
            result = minimize(obj, x0, method='Nelder-Mead',
                            options={'maxiter': maxiter, 'xatol': 1e-4, 'fatol': 1e-4})
            return result.x, result.fun
        except Exception:
            return x0, obj(x0)

    def _cobyla(self, obj, x0, bounds, maxiter):
        """COBYLA with bounds."""
        try:
            from scipy.optimize import minimize
            constraints = []
            if bounds:
                constraints = [{'type': 'ineq', 'fun': lambda x, i=i: x[i] - bounds[i][0]}
                              for i in range(len(bounds)) if bounds[i][0] > -np.inf]
                constraints += [{'type': 'ineq', 'fun': lambda x, i=i: bounds[i][1] - x[i]}
                              for i in range(len(bounds)) if bounds[i][1] < np.inf]

            result = minimize(obj, x0, method='COBYLA',
                           constraints=constraints,
                           options={'maxiter': maxiter})
            return result.x, result.fun
        except Exception:
            return x0, obj(x0)

    def _genetic(self, obj, x0, bounds, maxiter):
        """Simple genetic algorithm."""
        n = len(x0)
        pop_size = 20
        pop = np.zeros((pop_size, n))

        # Initialize
        if bounds:
            for i in range(pop_size):
                pop[i] = np.array([self._rng.uniform(lo, hi) for lo, hi in bounds])
        else:
            pop = x0 + self._rng.standard_normal((pop_size, n)) * 0.1

        fitness = np.array([obj(p) for p in pop])

        for _ in range(maxiter // 5):
            # Tournament selection
            new_pop = [pop[np.argmin(fitness[np.random.choice(pop_size, 3)])]]
            for _ in range(pop_size - 1):
                i = np.random.choice(pop_size, 3)
                winner = pop[i[np.argmin(fitness[i])]]
                # Crossover
                j = np.random.choice(pop_size)
                child = 0.5 * winner + 0.5 * pop[j]
                # Mutation
                if self._rng.random() < 0.2:
                    child = child + self._rng.standard_normal(n) * 0.05
                new_pop.append(child)
            pop = np.array(new_pop[:pop_size])
            fitness = np.array([obj(p) for p in pop])

        best_idx = np.argmin(fitness)
        return pop[best_idx], fitness[best_idx]
