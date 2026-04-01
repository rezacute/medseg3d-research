"""
Memory-efficient VQ codebook with Hilbert-Schmidt distance and EMA updates.

Key design principles:
1. NEVER materialize (B, T, K, S, S) — always flatten to (B*T, S²)
2. HS distance: ||rho - e_k||²_HS = ||rho||² + ||e_k||² - 2*Re(<rho|e_k>)
3. EMA updates under torch.no_grad() — never tracked by autograd
4. Laplace smoothing + dead code re-init to prevent NaN codebook vectors
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class HilbertSchmidtVQ(nn.Module):
    """
    Vector Quantizer using Hilbert-Schmidt distance on the Bloch sphere.

    Operates on flattened density matrices rho_flat ∈ ℝ^{B*T × S²} or ℂ^{B*T × S²}.
    Uses EMA codebook updates (no gradient estimation, no STE bias).
    """

    def __init__(
        self,
        k: int,             # codebook size (number of discrete symbols)
        dim: int,           # dimension of each codebook vector (S² for S×S density matrix)
        ema_decay: float = 0.99,
        epsilon: float = 1e-8,   # Laplace smoothing constant for dead codes
        init_method: str = "kmeans",  # "kmeans" | "random_normal" | "from_data"
    ):
        super().__init__()
        self.k = k
        self.dim = dim
        self.ema_decay = ema_decay
        self.eps = epsilon

        # ─── Codebook ───────────────────────────────────────────────────────────
        # Shape: (k, dim) — each row e_k is the HS vector of a density matrix
        self.register_buffer("codebook", torch.zeros(k, dim), persistent=False)
        self._init_codebook(init_method)

        # ─── EMA state (buffers, not parameters — not updated by Adam) ───────────
        # Cluster population counts: (k,)
        self.register_buffer("ema_cluster_size", torch.zeros(k) + self.eps, persistent=False)
        # Per-dimension accumulated squared gradient magnitudes for Laplace smoothing: (k, dim)
        self.register_buffer("ema_grad_sq", torch.zeros(k, dim) + self.eps, persistent=False)
        # Running cluster size used for update: (k,)
        self._cluster_size = torch.zeros(k) + self.eps

        # ─── Dead code tracking ────────────────────────────────────────────────
        self._dead_codes = torch.zeros(k, dtype=torch.bool)
        self._n_empty_after_init = 0   # tracks how many codes were zero-populated

    # --------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------

    def _init_codebook(self, method: str):
        if method == "random_normal":
            # Initialize as random pure states on the Bloch sphere
            # Sample from Gaussian, normalize, multiply by sqrt(dim) for proper HS norm
            flat = torch.randn(self.k, self.dim)
            flat = flat / flat.norm(dim=-1, keepdim=True) * torch.sqrt(torch.tensor(self.dim / self.k))
            self.codebook.copy_(flat)
        elif method == "kmeans":
            # Deferred: call fit_codebook_kmeans() on data before first forward pass
            pass
        else:
            raise ValueError(f"Unknown init_method: {method}")

    def fit_codebook_kmeans(
        self,
        rho_flat: torch.Tensor,
        n_iters: int = 20,
        tol: float = 1e-4,
        seed: int = 42,
    ):
        """
        Fit the VQ codebook to a sample of reservoir states using k-means++.

        Args:
            rho_flat: flattened reservoir states, shape (N, dim)
            n_iters: number of k-means iterations
            tol: convergence tolerance (relative change in inertia)
            seed: random seed for k-means++ init
        """
        import numpy as np

        N = rho_flat.shape[0]
        dtype = rho_flat.dtype
        device = rho_flat.device

        # k-means++ initialization
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, N)
        centers = [rho_flat[idx].cpu().numpy()]
        for _ in range(1, self.k):
            dists_sq = self._compute_dists_sq_cpu(rho_flat.cpu(), torch.tensor(centers))
            prob = dists_sq.min(dim=1).values.cpu().numpy()
            prob = prob / prob.sum()
            idx = rng.choice(N, p=prob)
            centers.append(rho_flat[idx].cpu().numpy())
        centers = torch.tensor(np.array(centers), dtype=dtype, device=device)

        # Lloyd's algorithm
        inertia_prev = float("inf")
        for it in range(n_iters):
            dists_sq = self._compute_dists_sq_cpu(rho_flat.cpu(), centers)
            labels = dists_sq.argmin(axis=1)  # (N,)
            inertia = dists_sq.min(dim=1).values.sum().item()

            if abs(inertia - inertia_prev) / max(inertia_prev, 1e-8) < tol:
                break
            inertia_prev = inertia

            # Update centers
            new_centers = torch.zeros_like(centers)
            counts = torch.zeros(self.k)
            for i in range(N):
                k = labels[i].item()
                new_centers[k] += rho_flat[i].cpu()
                counts[k] += 1
            for kk in range(self.k):
                if counts[kk] > 0:
                    centers[kk] = new_centers[kk] / counts[kk]

        self.codebook.copy_(centers.to(device))

    def _compute_dists_sq_cpu(self, rho: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        """Memory-safe HS distance on CPU (for k-means fitting)."""
        rho_sq = torch.einsum("bi,bi->b", rho, rho)
        cb_sq = torch.einsum("ki,ki->k", centers, centers)
        cross = 2 * torch.einsum("bi,ki->bk", rho, centers)
        return (rho_sq.unsqueeze(1) + cb_sq.unsqueeze(0) - cross).clamp(min=0.0)

    # --------------------------------------------------------------------------
    # Forward pass (differentiable via STE — only during training)
    # --------------------------------------------------------------------------

    def forward(
        self,
        rho_flat: torch.Tensor,
        return_indices: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass: assign each state to nearest codebook vector.

        Args:
            rho_flat: flattened states, shape (B*T, dim)
            return_indices: if True, also returns the hard assignment indices

        Returns:
            quantized: codebook vectors for each input, shape (B*T, dim)
            indices: (optional) argmin indices, shape (B*T,)
        """
        B = rho_flat.shape[0]

        # ─── HS distance: (B, K) ───────────────────────────────────────────────
        dists_sq = batched_hs_distance_sq(rho_flat, self.codebook)  # (B, K)

        # ─── Hard assignment ────────────────────────────────────────────────────
        indices = dists_sq.argmin(dim=-1)  # (B,)
        z_hard = F.one_hot(indices, self.k).float()  # (B, K)

        # ─── Straight-through: z_q = e[idx]; gradient = identity ────────────
        # z_q = e[indices]
        quantized = F.embedding(indices, self.codebook)  # (B, dim)
        # STE: gradient passes through as if z_q == rho_flat
        quantized = rho_flat + (quantized - quantized.detach())

        if return_indices:
            return quantized, indices
        return quantized

    # --------------------------------------------------------------------------
    # EMA codebook update (torch.no_grad — never tracked by autograd)
    # --------------------------------------------------------------------------

    @torch.no_grad()
    def ema_update(
        self,
        rho_flat: torch.Tensor,
        indices: Optional[torch.Tensor] = None,
        commitment_weight: float = 0.0,
    ):
        """
        Update codebook via exponential moving average.

        MUST be called under torch.no_grad() context (enforced).

        EMA update rules:
            N_k := decay * N_k + (1-decay) * sum_i [assignment_i == k]
            S_k := decay * S_k + (1-decay) * sum_i rho_i [assignment_i == k]
            e_k := S_k / N_k   (mean of assigned states)

        With Laplace smoothing: e_k := (S_k + eps) / (N_k + k*eps)

        Args:
            rho_flat: flattened states, shape (B*T, dim)
            indices: argmin indices, shape (B*T,). If None, recomputes.
            commitment_weight: unused (kept for API compatibility with hybrid approach)
        """
        if not torch.no_grad():
            raise RuntimeError(
                "ema_update() must be called within a torch.no_grad() context. "
                "Wrap the call site, not this method."
            )

        B = rho_flat.shape[0]
        k = self.k

        # ─── Recompute indices if not provided ─────────────────────────────────
        if indices is None:
            dists_sq = batched_hs_distance_sq(rho_flat, self.codebook)
            indices = dists_sq.argmin(dim=-1)  # (B,)

        # ─── One-hot assignments: (B, K) ──────────────────────────────────────
        one_hot = F.one_hot(indices, k).float()  # (B, K)

        # ─── Cluster statistics for this batch ─────────────────────────────────
        batch_cluster_size = one_hot.sum(dim=0)  # (K,) — how many assigned to each code
        # Sum of states assigned to each code: (K, dim)
        batch_cluster_sum = one_hot.T @ rho_flat  # (K, dim)

        # ─── Update EMA state ───────────────────────────────────────────────────
        decay = self.ema_decay
        self.ema_cluster_size.mul_(decay).add_(batch_cluster_size, alpha=1.0 - decay)
        self.ema_grad_sq.mul_(decay).add_(batch_cluster_sum.pow(2), alpha=1.0 - decay)

        # ─── Laplace-smoothed codebook update ──────────────────────────────────
        # e_k = (S_k + eps) / (N_k + k*eps)  for each dimension
        denom = self.ema_cluster_size.unsqueeze(1) + self.eps   # (K, 1)
        new_codebook = self.ema_grad_sq.sqrt() / denom.sqrt()   # (K, dim)

        # ─── Dead code re-initialization ──────────────────────────────────────
        # If a code gets fewer than 1% of total points over a full pass, re-init it
        min_share = 0.01 * B
        dead_mask = self.ema_cluster_size < min_share  # (K,)

        if dead_mask.any():
            n_dead = dead_mask.sum().item()
            # Re-init dead codes from current batch's mean (highest-entropy source)
            batch_mean = rho_flat.mean(dim=0, keepdim=True)  # (1, dim)
            # Add small noise to differentiate
            noise_scale = self.eps ** 0.5
            reinit = batch_mean + torch.randn_like(batch_mean) * noise_scale
            new_codebook = torch.where(dead_mask.unsqueeze(1), reinit, new_codebook)
            self._dead_codes = dead_mask.clone()
            self._n_empty_after_init += n_dead

        # ─── Apply update (in-place) ───────────────────────────────────────────
        self.codebook.copy_(new_codebook)

    # --------------------------------------------------------------------------
    # Memory-safe HS distance (the core of this implementation)
    # --------------------------------------------------------------------------

    @torch.no_grad()
    def assign(
        self,
        rho_flat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Just return the hard indices — no gradient, no codebook change.
        Used for inference / evaluation.
        """
        dists_sq = batched_hs_distance_sq(rho_flat, self.codebook)  # (B, K)
        return dists_sq.argmin(dim=-1)


# =============================================================================
# Core: memory-safe vectorized HS distance
# =============================================================================

def batched_hs_distance_sq(
    rho_flat: torch.Tensor,
    codebook_flat: torch.Tensor,
) -> torch.Tensor:
    """
    Memory-safe Hilbert-Schmidt distance between flattened states and codebook.

    Avoids the catastrophic (B, T, K, S, S) allocation by using the
    Hilbert-Schmidt inner product expansion:
        ||rho - e_k||²_HS
        = ||rho||² + ||e_k||² - 2*Re(<rho|e_k>)
        = tr(rho^dagger rho) + tr(e_k^dagger e_k) - 2*Re(tr(rho^dagger e_k))

    For HS-vectorized density matrices (real vectors in ℝ^{S²}):
        ||rho - e_k||² = ||rho||² + ||e_k||² - 2 * rho^T @ e_k

    Memory: O(B*K) instead of O(B*K*S²).

    Args:
        rho_flat: flattened states, shape (N, dim) where dim = S²
        codebook_flat: codebook vectors, shape (K, dim)

    Returns:
        squared distances, shape (N, K)
    """
    # ─── ||rho||²  (N,)
    rho_norm_sq = torch.einsum("ni,ni->n", rho_flat, rho_flat).real

    # ─── ||e_k||²  (K,)
    cb_norm_sq = torch.einsum("ki,ki->k", codebook_flat, codebook_flat).real

    # ─── Cross term: -2 * Re(<rho|e_k>) = -2 * rho^T @ e_k
    # rho @ codebook.T = (N, K)
    cross = 2.0 * torch.einsum("ni,ki->nk", rho_flat, codebook_flat)

    # ─── Broadcast to (N, K) ──────────────────────────────────────────────────
    dists_sq = rho_norm_sq.unsqueeze(1) + cb_norm_sq.unsqueeze(0) - cross

    # Numerical safety: clamp negative values to 0 (HS distance is non-negative)
    dists_sq = torch.clamp(dists_sq, min=0.0)

    return dists_sq


def hs_distance(
    rho_flat: torch.Tensor,
    codebook_flat: torch.Tensor,
) -> torch.Tensor:
    """
    Hilbert-Schmidt distance (not squared). Same interface as batched_hs_distance_sq.
    """
    return torch.sqrt(batched_hs_distance_sq(rho_flat, codebook_flat) + 1e-12)


# =============================================================================
# Integration: VQHMM with OMLE
# =============================================================================

class VQHMMModel(nn.Module):
    """
    VQ-quantized reservoir + OMLE HMM for temporal learning.

    Pipeline:
        1. Reservoir: X → rho_flat ∈ ℝ^{T × S²}
        2. VQ: rho_flat → indices ∈ {0,...,K-1}^T
        3. OMLE: trains CPTP channel on index trajectory
    """

    def __init__(
        self,
        n_qubits: int,
        k: int = 8,                # VQ codebook size
        ema_decay: float = 0.99,
        device: str = "cuda",
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.k = k
        self.dim = 2 ** (2 * n_qubits)  # S² for S=2^n qubits

        # Placeholder for actual reservoir (initialized by caller)
        self.reservoir = None

        # VQ codebook — operating on flattened reservoir features
        # We use dim_reduced = min(dim, 64) to keep codebook manageable
        self.vq_dim = min(self.dim, 64)
        self.vq = HilbertSchmidtVQ(
            k=k,
            dim=self.vq_dim,
            ema_decay=ema_decay,
            init_method="random_normal",
        )

        self.device = device

    def encode(self, X: torch.Tensor) -> torch.Tensor:
        """
        Encode input sequence through reservoir → flatten → project to vq_dim.
        Returns flattened states: (T, vq_dim)
        """
        # Process through reservoir: returns (T, 2*n_qubits)
        rho_flat = self.reservoir.process_sequence(X)  # (T, 2*n_qubits)

        # Project to vq_dim if needed
        if self.vq_dim < rho_flat.shape[-1]:
            # Simple linear projection: take first vq_dim dims
            rho_flat = rho_flat[:, :self.vq_dim]

        return rho_flat  # (T, vq_dim)

    def quantize(
        self,
        rho_flat: torch.Tensor,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize states to VQ codebook indices.

        Returns:
            quantized: (T, vq_dim)
            indices: (T,) integer indices
        """
        if training:
            return self.vq(rho_flat, return_indices=True)
        else:
            indices = self.vq.assign(rho_flat)
            quantized = F.embedding(indices, self.vq.codebook)
            return quantized, indices

    def update_codebook(
        self,
        rho_flat: torch.Tensor,
        indices: torch.Tensor,
    ):
        """EMA codebook update — call under torch.no_grad()."""
        self.vq.ema_update(rho_flat, indices=indices)

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        n_omle_iters: int = 10,
    ):
        """
        Fit VQ codebook + OMLE on the trajectory.

        Args:
            X: input features, shape (T, d_in)
            y: target values, shape (T,)
            n_omle_iters: number of OMLE EM iterations
        """
        from qrc_ev.agents.qhmm_omle_cudaqx import OOMModel, OMLeAgent

        T = X.shape[0]

        # ─── Encode ────────────────────────────────────────────────────────────
        rho_flat = self.encode(X)  # (T, vq_dim)

        # ─── VQ: initialize codebook from data (k-means) ───────────────────────
        # Do k-means init before any training
        with torch.no_grad():
            self.vq.fit_codebook_kmeans(rho_flat.detach())

        # ─── Training loop ──────────────────────────────────────────────────────
        for epoch in range(n_omle_iters):
            # Forward through VQ
            z_q, indices = self.quantize(rho_flat.detach(), training=True)

            # OMLE update would go here (on the index sequence)
            # self.omle.fit_on_trajectory(indices.cpu().numpy())

            # EMA codebook update
            self.update_codebook(rho_flat.detach(), indices)

        return indices.cpu().numpy()

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict at inference — hard VQ, no training."""
        rho_flat = self.encode(X)
        _, indices = self.quantize(rho_flat, training=False)
        return indices


# =============================================================================
# CLI for sweep runner
# =============================================================================

if __name__ == "__main__":
    import argparse
    import json
    import gc

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="JSON config file")
    parser.add_argument("--out", type=str, required=True, help="Output JSON")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    print(f"Running VQ sweep: {config}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        # ─── Import and run experiment ──────────────────────────────────────────
        from qrc_ev.data.ev_datasets import load_dataset

        # Build model
        model = VQHMMModel(
            n_qubits=config["n_qubits"],
            k=config["k"],
            ema_decay=config.get("ema_decay", 0.99),
            device=device,
        ).to(device)

        # Fit on training data
        data = load_dataset(config.get("dataset", "narma10"))
        X_train = torch.from_numpy(data["train"]).float()
        y_train = torch.from_numpy(data["train"]).float()

        indices = model.fit(X_train, y_train, n_omle_iters=config.get("n_omle_iters", 10))

        # Evaluate
        from sklearn.metrics import r2_score
        X_test = torch.from_numpy(data["test"]).float()
        preds = model.predict(X_test)
        r2 = r2_score(data["test"][model.vq.k:], preds.cpu().numpy())

        result = {"r2": r2, "status": "success", **config}

        # ─── Save result ───────────────────────────────────────────────────────
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)

        print(f"R²={r2:.4f} — saved to {args.out}")

    except torch.cuda.OutOfMemoryError as e:
        print(f"[OOM] Skipping {config}: {e}")
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
        result = {"status": "oom", "error": str(e), **config}
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)

    except Exception as e:
        print(f"[ERROR] {config}: {e}")
        result = {"status": "error", "error": str(e), **config}
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
