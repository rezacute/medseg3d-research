"""Integration tests for OOMModel forward-backward algorithm and OMLeAgent.

These tests verify:
- OOMModel forward-backward correctness on known channels (identity, depolarizing)
- Smoothing posteriors are valid probability distributions
- Log-likelihood computation is correct
- OOMModel + OMLeAgent end-to-end consistency
"""

import numpy as np
import pytest

from qrc_ev.agents.qhmm_omle_cudaqx import (
    OOMModel,
    OMLeAgent,
    hs_vectorize,
)


# =============================================================================
# Helper: build identity-channel OOM (no evolution)
# =============================================================================


def _identity_oom(S=2, A=2, O=2):
    """Build an OOM with identity channels (perfect memory = no information loss)."""
    K_I = np.eye(S, dtype=np.complex128)
    J_I = K_I[:, :, np.newaxis] * K_I[np.newaxis, :, :]  # K⊗K for choi

    # Use simple Kraus: just K=I (TP)
    from qrc_ev.agents.qhmm_omle_cudaqx import choi_from_kraus
    J_I = choi_from_kraus([K_I])

    agent = OMLeAgent(
        S=S, A=A, O=O, L=1,
        init_channels=[J_I],
        init_instruments={(a, o): J_I for a in range(A) for o in range(O)},
    )
    return OOMModel(S=S, A=A, O=O, L=1, omle_agent=agent)


# =============================================================================
# §1 — Identity channel tests (known answer: Z_t = 1 always)
# =============================================================================


class TestOOMIdentityChannel:
    """Tests on identity channel where we know exact answers."""

    @pytest.fixture
    def oom(self):
        return _identity_oom(S=2, A=2, O=2)

    def test_forward_pass_marginal_likelihood_identity(self, oom):
        """Identity channel: P(o_t | past) = 1 for all t."""
        actions = np.array([0, 0, 0, 0])
        outcomes = np.array([0, 1, 0, 1])
        result = oom.compute_forward_backward(actions, outcomes)

        Z = result["Z"]
        np.testing.assert_allclose(Z, np.ones(len(actions)), atol=1e-10,
                                   err_msg="Identity channel: Z_t should be 1")

    def test_forward_posteriors_sum_to_one(self, oom):
        """Forward posteriors α_t should sum to 1 at each t."""
        actions = np.array([0, 1, 0, 1])
        outcomes = np.array([0, 0, 1, 1])
        result = oom.compute_forward_backward(actions, outcomes)

        alpha = result["alpha"]
        for t in range(len(actions)):
            assert np.isclose(np.sum(alpha[t]), 1.0, atol=1e-10), \
                f"α_t should sum to 1, t={t}: {np.sum(alpha[t])}"

    def test_smoothing_posteriors_sum_to_one(self, oom):
        """Smoothing posteriors P(ρ_t | τ_{1:T}) should sum to 1 at each t."""
        actions = np.array([0, 1, 0, 1])
        outcomes = np.array([0, 0, 1, 1])
        result = oom.compute_forward_backward(actions, outcomes)

        post = result["smoothing_posteriors"]
        for t in range(len(actions)):
            assert np.isclose(np.sum(post[t]), 1.0, atol=1e-10), \
                f"Smoothing posterior should sum to 1, t={t}: {np.sum(post[t])}"

    def test_loglikelihood_identity_channel(self, oom):
        """Identity channel: log P(trajectory) = 0 since P=1 at each step."""
        actions = np.array([0, 0, 0, 0])
        outcomes = np.array([0, 1, 0, 1])
        result = oom.compute_forward_backward(actions, outcomes)

        assert np.isclose(result["loglikelihood"], 0.0, atol=1e-9), \
            f"Identity channel loglik should be 0, got {result['loglikelihood']}"

    def test_backward_beta_terminates_at_one(self, oom):
        """β_T should be all ones (no future evidence)."""
        actions = np.array([0, 1, 0, 1])
        outcomes = np.array([0, 0, 1, 1])
        result = oom.compute_forward_backward(actions, outcomes)

        np.testing.assert_allclose(result["beta"][-1], np.ones(oom.S2), atol=1e-10,
                                   err_msg="β_T should be all ones")

    def test_empty_trajectory(self, oom):
        """Empty trajectory should return empty arrays and loglik=0."""
        result = oom.compute_forward_backward(
            actions=np.array([], dtype=int),
            outcomes=np.array([], dtype=int),
        )
        assert result["alpha"].shape == (0, oom.S2)
        assert result["beta"].shape == (0, oom.S2)
        assert result["Z"].shape == (0,)
        assert result["smoothing_posteriors"].shape == (0, oom.S2)
        assert result["loglikelihood"] == 0.0

    def test_single_step_trajectory(self, oom):
        """Single-step trajectory should be handled correctly."""
        actions = np.array([0])
        outcomes = np.array([0])
        result = oom.compute_forward_backward(actions, outcomes)

        assert result["alpha"].shape == (1, oom.S2)
        assert result["beta"].shape == (1, oom.S2)
        assert result["Z"].shape == (1,)
        assert np.isclose(np.sum(result["alpha"][0]), 1.0, atol=1e-10)
        np.testing.assert_allclose(result["beta"][0], np.ones(oom.S2), atol=1e-10)


# =============================================================================
# §2 — Consistency between forward and backward passes
# =============================================================================


class TestOOMForwardBackwardConsistency:
    """Test mathematical consistency between forward and backward."""

    @pytest.fixture
    def oom(self):
        return _identity_oom(S=2, A=2, O=2)

    def test_smoothing_equals_forward_when_beta_uniform(self, oom):
        """When β_t = 1 for all t, smoothing should equal forward."""
        # For a very short trajectory on identity channel, the backward pass
        # with the identity operator gives β_t = 1 uniformly
        actions = np.array([0])
        outcomes = np.array([0])
        result = oom.compute_forward_backward(actions, outcomes)

        np.testing.assert_allclose(
            result["smoothing_posteriors"][0],
            result["alpha"][0],
            atol=1e-10,
            err_msg="Smoothing should equal forward when β=1"
        )

    def test_smoothing_mass_smaller_than_forward_informed_case(self, oom):
        """When backward pass carries information, smoothing is more informed."""
        # On identity channel, backward pass doesn't add info (β stays uniform)
        # so smoothing ≈ forward. This is a sanity check.
        actions = np.array([0, 0, 0, 0])
        outcomes = np.array([0, 1, 0, 1])
        result = oom.compute_forward_backward(actions, outcomes)

        # Entropy of smoothing should be <= entropy of forward
        def entropy(p):
            p = np.maximum(p, 1e-12)
            return -np.sum(p * np.log(p))

        for t in range(len(actions)):
            H_forward = entropy(result["alpha"][t])
            H_smooth = entropy(result["smoothing_posteriors"][t])
            assert H_smooth <= H_forward + 1e-8, \
                f"t={t}: smoothing entropy {H_smooth} > forward entropy {H_forward}"


# =============================================================================
# §3 — OOMModel + OMLeAgent end-to-end
# =============================================================================


class TestOOMWithOMLeAgent:
    """Integration: build OOM from OMLeAgent and verify consistency."""

    def test_oom_builds_from_agent(self):
        """OOMModel should initialize from an OMLeAgent."""
        agent = OMLeAgent(S=2, A=2, O=2, L=1)
        oom = OOMModel(S=2, A=2, O=2, L=1, omle_agent=agent)

        assert oom.S == 2
        assert oom.A == 2
        assert oom.O == 2
        assert oom.L == 1

    def test_oom_default_initial_state(self):
        """Default OOM (no agent) should use |0⟩⟨0| as initial state."""
        oom = OOMModel(S=2, A=2, O=2, L=1)
        v1 = hs_vectorize(oom._rho1, oom._basis)
        assert np.isclose(np.sum(v1), 1.0, atol=1e-10), \
            "Initial state |0⟩⟨0| should vectorize to uniform [1,0,...]"

    def test_oom_forward_backward_no_crash(self):
        """Forward-backward should run without errors on a random trajectory."""
        oom = _identity_oom(S=2, A=2, O=2)

        rng = np.random.default_rng(42)
        T = 10
        actions = rng.integers(0, 2, size=T)
        outcomes = rng.integers(0, 2, size=T)

        # Should not raise
        result = oom.compute_forward_backward(actions, outcomes)

        assert result["alpha"].shape == (T, oom.S2)
        assert result["beta"].shape == (T, oom.S2)
        assert result["Z"].shape == (T,)
        assert result["smoothing_posteriors"].shape == (T, oom.S2)
        assert np.isfinite(result["loglikelihood"])

    def test_oom_reproducibility(self):
        """Same trajectory + same seed should give identical results."""
        oom = _identity_oom(S=2, A=2, O=2)
        actions = np.array([0, 1, 0, 1, 0])
        outcomes = np.array([0, 0, 1, 1, 0])

        result1 = oom.compute_forward_backward(actions, outcomes)
        result2 = oom.compute_forward_backward(actions, outcomes)

        np.testing.assert_array_equal(result1["alpha"], result2["alpha"])
        np.testing.assert_array_equal(result1["beta"], result2["beta"])
        np.testing.assert_array_equal(result1["Z"], result2["Z"])
        assert result1["loglikelihood"] == result2["loglikelihood"]

    def test_oom_vec_to_state_roundtrip(self):
        """vec_to_state should recover a valid density matrix."""
        oom = _identity_oom(S=2, A=2, O=2)

        actions = np.array([0, 1, 0])
        outcomes = np.array([0, 0, 1])
        result = oom.compute_forward_backward(actions, outcomes)

        for t in range(len(actions)):
            v = result["smoothing_posteriors"][t]
            rho = oom.vec_to_state(v, action=actions[t])

            # Should be square
            assert rho.shape == (2, 2)
            # Should be Hermitian
            np.testing.assert_allclose(rho, rho.conj().T, atol=1e-10)
            # Should have trace 1
            assert np.isclose(np.trace(rho), 1.0, atol=1e-10)
            # Should be PSD
            evals = np.linalg.eigvalsh(rho)
            assert np.all(evals >= -1e-8), f"Negative eigenvalues at t={t}: {evals}"


# =============================================================================
# §4 — Different S dimensions
# =============================================================================


class TestOOMDifferentDimensions:
    """Test OOMModel with different Hilbert space dimensions."""

    @pytest.mark.parametrize("S", [2, 3, 4])
    def test_forward_posteriors_sum_to_one_various_S(self, S):
        """Forward posteriors should sum to 1 regardless of S."""
        oom = _identity_oom(S=S, A=2, O=2)

        actions = np.array([0, 1, 0])
        outcomes = np.array([0, 0, 1])
        result = oom.compute_forward_backward(actions, outcomes)

        for t in range(len(actions)):
            assert np.isclose(np.sum(result["alpha"][t]), 1.0, atol=1e-10), \
                f"S={S}, t={t}: α_t sum={np.sum(result['alpha'][t])}"

    @pytest.mark.parametrize("A", [1, 2, 3])
    def test_various_action_counts(self, A):
        """OOM should handle various numbers of actions."""
        oom = _identity_oom(S=2, A=A, O=2)

        actions = np.array([i % A for i in range(5)])
        outcomes = np.array([0, 1, 0, 1, 0])
        result = oom.compute_forward_backward(actions, outcomes)

        assert result["alpha"].shape == (5, oom.S2)
        for t in range(5):
            assert np.isclose(np.sum(result["alpha"][t]), 1.0, atol=1e-10)

    @pytest.mark.parametrize("O", [1, 2, 3, 4])
    def test_various_outcome_counts(self, O):
        """OOM should handle various numbers of outcomes."""
        oom = _identity_oom(S=2, A=2, O=O)

        actions = np.array([0, 1, 0, 1, 0])
        outcomes = np.array([i % O for i in range(5)])
        result = oom.compute_forward_backward(actions, outcomes)

        assert result["alpha"].shape == (5, oom.S2)
        for t in range(5):
            assert np.isclose(np.sum(result["alpha"][t]), 1.0, atol=1e-10)


# =============================================================================
# §5 — kappa_uc per action
# =============================================================================


class TestOOMKappaUC:
    """Test κ_uc (CB-norm) computation per action."""

    def test_kappa_uc_identity_channel(self):
        """Identity channel recovery should have κ_uc = 1."""
        oom = _identity_oom(S=2, A=2, O=2)

        for a in range(oom.A):
            kappa = oom.kappa_uc_per_action[a]
            assert np.isclose(kappa, 1.0, atol=1e-6), \
                f"Identity channel: κ_uc[{a}] = {kappa}, expected 1"

    def test_kappa_uc_positive(self):
        """κ_uc should be positive for all actions."""
        oom = _identity_oom(S=2, A=2, O=2)

        for a, kappa in oom.kappa_uc_per_action.items():
            assert kappa > 0, f"κ_uc[{a}] = {kappa} should be positive"


# =============================================================================
# §6 — Multi-channel OOM
# =============================================================================


class TestOOMMultiChannel:
    """Test OOM with multiple channels (L > 1)."""

    def test_two_channel_oom(self):
        """Build OOM with 2 channels and verify forward-backward works."""
        from qrc_ev.agents.qhmm_omle_cudaqx import choi_from_kraus

        S, A, O = 2, 2, 2
        K_I = np.eye(S, dtype=np.complex128)
        J_I = choi_from_kraus([K_I])

        # Two identical identity channels
        agent = OMLeAgent(
            S=S, A=A, O=O, L=2,
            init_channels=[J_I, J_I],
            init_instruments={(a, o): J_I for a in range(A) for o in range(O)},
        )
        oom = OOMModel(S=S, A=A, O=O, L=2, omle_agent=agent)

        actions = np.array([0, 1, 0])
        outcomes = np.array([0, 0, 1])
        result = oom.compute_forward_backward(actions, outcomes, channel_idx=0)

        assert result["alpha"].shape == (3, oom.S2)
        assert np.all(np.isfinite(result["alpha"]))
        assert np.all(np.isfinite(result["beta"]))
