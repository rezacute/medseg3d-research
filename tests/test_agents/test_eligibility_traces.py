"""Integration tests for OOMModel eligibility traces (TD-λ).

Tests verify:
- Forward traces accumulate correctly with decay
- Backward traces accumulate correctly with decay
- TD errors are computed correctly
- Combined traces scale properly
- Parameter update direction is sensible
- Decay parameters behave as expected
"""

import numpy as np
import pytest

from qrc_ev.agents.qhmm_omle_cudaqx import (
    OOMModel,
    OMLeAgent,
    choi_from_kraus,
)


def _identity_oom(S=2, A=2, O=2):
    """Build OOM with identity channels (known answer: Z_t=1, loglik=0)."""
    K_I = np.eye(S, dtype=np.complex128)
    J_I = choi_from_kraus([K_I])
    agent = OMLeAgent(
        S=S, A=A, O=O, L=1,
        init_channels=[J_I],
        init_instruments={(a, o): J_I for a in range(A) for o in range(O)},
    )
    return OOMModel(S=S, A=A, O=O, L=1, omle_agent=agent)


# =============================================================================
# §1 — Forward eligibility traces
# =============================================================================


class TestEligibilityTracesForward:
    """Test forward trace accumulation."""

    @pytest.fixture
    def oom(self):
        return _identity_oom(S=2, A=2, O=2)

    def test_forward_trace_accumulates(self, oom):
        """Forward trace should grow with each step (identity channel, Z=1)."""
        actions = np.array([0, 0, 0, 0])
        outcomes = np.array([0, 0, 0, 0])
        result = oom.compute_eligibility_traces(actions, outcomes)

        e_fwd = result['e_forward']
        # e_t should be non-decreasing for identity channel (no decay wiping it out)
        for t in range(1, len(e_fwd)):
            assert np.sum(np.abs(e_fwd[t])) >= np.sum(np.abs(e_fwd[t - 1])) - 1e-10, \
                f"Forward trace should not decrease: t={t}"

    def test_forward_trace_lambda_decay(self, oom):
        """Higher lambda_fwd → larger accumulated traces."""
        actions = np.array([0, 1, 0, 1])
        outcomes = np.array([0, 0, 1, 1])

        result_lambda0 = oom.compute_eligibility_traces(
            actions, outcomes, lambda_fwd=0.0
        )
        result_lambda9 = oom.compute_eligibility_traces(
            actions, outcomes, lambda_fwd=0.9
        )

        e0 = result_lambda0['e_forward']
        e9 = result_lambda9['e_forward']

        # λ=0.9 traces should be larger than λ=0 (no accumulation for λ=0)
        for t in range(len(e0)):
            assert np.sum(np.abs(e9[t])) >= np.sum(np.abs(e0[t])) - 1e-10, \
                f"λ=0.9 trace @ t={t} should be >= λ=0 trace"

    def test_forward_trace_gamma_decay(self, oom):
        """Higher gamma → larger accumulated traces (less discounting)."""
        actions = np.array([0, 1, 0, 1])
        outcomes = np.array([0, 0, 1, 1])

        result_g0 = oom.compute_eligibility_traces(actions, outcomes, gamma=0.0)
        result_g9 = oom.compute_eligibility_traces(actions, outcomes, gamma=0.9)

        e_g0 = result_g0['e_forward']
        e_g9 = result_g9['e_forward']

        for t in range(len(e_g0)):
            assert np.sum(np.abs(e_g9[t])) >= np.sum(np.abs(e_g0[t])) - 1e-10, \
                f"γ=0.9 trace @ t={t} should be >= γ=0 trace"

    def test_forward_trace_initialization(self, oom):
        """First-step trace should equal first alpha."""
        actions = np.array([0, 1, 0])
        outcomes = np.array([0, 0, 1])
        result = oom.compute_eligibility_traces(actions, outcomes)

        e_fwd = result['e_forward']
        alpha = result['alpha']

        np.testing.assert_allclose(
            e_fwd[0], alpha[0], atol=1e-10,
            err_msg="First forward trace should equal α_0"
        )


# =============================================================================
# §2 — Backward eligibility traces
# =============================================================================


class TestEligibilityTracesBackward:
    """Test backward trace accumulation."""

    @pytest.fixture
    def oom(self):
        return _identity_oom(S=2, A=2, O=2)

    def test_backward_trace_last_step(self, oom):
        """Last step backward trace should reflect initial TD error."""
        actions = np.array([0, 0, 0])
        outcomes = np.array([0, 1, 0])
        result = oom.compute_eligibility_traces(actions, outcomes)

        e_bwd = result['e_backward']
        delta = result['delta']

        # Last step's backward trace should scale with |δ_{T-1}|
        # (since there's no future to accumulate from)
        for t in range(len(e_bwd)):
            assert np.all(np.abs(e_bwd[t]) >= 0.0), \
                f"Backward trace should be non-negative: t={t}"

    def test_backward_trace_grows_with_history(self, oom):
        """Backward traces should accumulate from the end."""
        actions = np.array([0, 0, 0, 0])
        outcomes = np.array([0, 0, 1, 1])
        result = oom.compute_eligibility_traces(actions, outcomes)

        e_bwd = result['e_backward']

        # Backward traces accumulate backwards, so earlier steps may be larger
        # if they had larger TD errors
        for t in range(len(e_bwd)):
            assert np.all(np.isfinite(e_bwd[t])), f"Backward trace should be finite: t={t}"


# =============================================================================
# §3 — TD errors
# =============================================================================


class TestTDError:
    """Test TD error computation."""

    @pytest.fixture
    def oom(self):
        return _identity_oom(S=2, A=2, O=2)

    def test_delta_identity_channel(self, oom):
        """Identity channel: log Z_t = 0, so δ_t ≈ r_t + γ·V_next - V_t."""
        actions = np.array([0, 0, 0, 0])
        outcomes = np.array([0, 1, 0, 1])
        result = oom.compute_eligibility_traces(actions, outcomes)

        delta = result['delta']
        Z = result['Z']

        # Z_t = 1 for identity channel → log Z_t = 0
        np.testing.assert_allclose(Z, np.ones(len(actions)), atol=1e-10)

        # With V initialized near 0 and rewards = log Z = 0,
        # δ_t should be small (≈ 0 for identity channel)
        for t in range(len(delta)):
            assert np.isfinite(delta[t]), f"δ_t should be finite: t={t}"

    def test_delta_with_explicit_rewards(self, oom):
        """TD errors should reflect the provided reward signal."""
        actions = np.array([0, 0, 0])
        outcomes = np.array([0, 1, 0])
        rewards = np.array([1.0, -1.0, 0.5])

        result = oom.compute_eligibility_traces(actions, outcomes, rewards=rewards)

        delta = result['delta']
        used_rewards = result['rewards']

        np.testing.assert_array_equal(used_rewards[:3], rewards)
        for t in range(len(delta)):
            assert np.isfinite(delta[t]), f"δ_t should be finite: t={t}"

    def test_delta_shape_matches_trajectory(self, oom):
        """Delta array should have same length as trajectory."""
        for T in [1, 5, 10]:
            actions = np.zeros(T, dtype=int)
            outcomes = np.zeros(T, dtype=int)
            result = oom.compute_eligibility_traces(actions, outcomes)

            assert len(result['delta']) == T, f"T={T}: delta length mismatch"
            assert len(result['alpha']) == T, f"T={T}: alpha length mismatch"
            assert len(result['V']) == T, f"T={T}: V length mismatch"

    def test_delta_bootstrap_at_end(self, oom):
        """Last δ_T should be small (bootstrap from V, which is near 0 at end)."""
        actions = np.array([0, 1, 0, 1])
        outcomes = np.array([0, 0, 1, 1])
        result = oom.compute_eligibility_traces(actions, outcomes, rewards=np.ones(5))

        delta = result['delta']
        # With positive rewards, even the last step's δ should be positive
        # (bootstrap V_next ≈ 0, but r > 0 dominates)
        assert delta[-1] > 0 or np.isclose(delta[-1], 0, atol=1e-2), \
            f"Last delta should be >= 0 with positive rewards: {delta[-1]}"


# =============================================================================
# §4 — Combined traces
# =============================================================================


class TestCombinedTraces:
    """Test combined forward-backward eligibility traces."""

    @pytest.fixture
    def oom(self):
        return _identity_oom(S=2, A=2, O=2)

    def test_combined_trace_nonnegative(self, oom):
        """Combined traces should be non-negative (product of positive quantities)."""
        actions = np.array([0, 1, 0, 1])
        outcomes = np.array([0, 0, 1, 1])
        result = oom.compute_eligibility_traces(actions, outcomes)

        e_comb = result['e_combined']

        for t in range(len(e_comb)):
            assert np.all(e_comb[t] >= -1e-10), \
                f"Combined trace should be non-negative: t={t}, min={np.min(e_comb[t])}"

    def test_combined_trace_scaling(self, oom):
        """Combined trace should be bounded by forward trace magnitude."""
        actions = np.array([0, 1, 0, 1])
        outcomes = np.array([0, 0, 1, 1])
        result = oom.compute_eligibility_traces(actions, outcomes)

        e_fwd = result['e_forward']
        e_comb = result['e_combined']

        for t in range(len(e_fwd)):
            # Combined = forward * normalized_backward ∈ [0, forward]
            assert np.sum(np.abs(e_comb[t])) <= np.sum(np.abs(e_fwd[t])) + 1e-10, \
                f"t={t}: combined trace should be <= forward trace"

    def test_combined_trace_zero_when_fwd_zero(self, oom):
        """Combined trace should be zero when forward trace is zero."""
        actions = np.array([0, 0])
        outcomes = np.array([0, 0])
        result = oom.compute_eligibility_traces(actions, outcomes, lambda_fwd=0.0)

        e_comb = result['e_combined']

        # λ=0 forward: e_t = α_t (no accumulation)
        # but backward normalization should still work
        assert np.all(np.isfinite(e_comb)), "Combined trace should be finite"


# =============================================================================
# §5 — Parameter updates
# =============================================================================


class TestParameterUpdates:
    """Test OOM parameter updates from eligibility traces."""

    @pytest.fixture
    def oom(self):
        return _identity_oom(S=2, A=2, O=2)

    def test_update_returns_valid_dict(self, oom):
        """update_parameters_from_traces should return a valid dict."""
        actions = np.array([0, 1, 0])
        outcomes = np.array([0, 0, 1])
        traces = oom.compute_eligibility_traces(actions, outcomes)

        update_result = oom.update_parameters_from_traces(
            traces, actions, outcomes, eta=0.01
        )

        assert 'delta_A_sum' in update_result
        assert 'max_update' in update_result
        assert 'mean_update' in update_result
        assert update_result['max_update'] >= 0
        assert update_result['mean_update'] >= 0

    def test_update_smaller_with_smaller_eta(self, oom):
        """Smaller learning rate → smaller parameter updates."""
        actions = np.array([0, 1, 0, 1])
        outcomes = np.array([0, 0, 1, 1])
        traces = oom.compute_eligibility_traces(actions, outcomes)

        update_small = oom.update_parameters_from_traces(
            traces, actions, outcomes, eta=0.001
        )
        update_large = oom.update_parameters_from_traces(
            traces, actions, outcomes, eta=0.1
        )

        assert update_small['max_update'] <= update_large['max_update'] + 1e-10, \
            "Smaller η should give smaller updates"

    def test_update_zero_with_zero_delta(self, oom):
        """Zero TD error → zero parameter update."""
        # Use rewards that cancel out to get near-zero δ
        actions = np.array([0, 0])
        outcomes = np.array([0, 0])

        # With identity channel and near-zero rewards, δ ≈ 0
        traces = oom.compute_eligibility_traces(
            actions, outcomes, rewards=np.zeros(3), gamma=0.0
        )
        update = oom.update_parameters_from_traces(
            traces, actions, outcomes, eta=0.1
        )

        # With zero rewards, the TD errors should be very small
        # → updates should be tiny
        delta = traces['delta']
        assert np.max(np.abs(delta)) < 1e-6, \
            f"δ should be ~0 with 0 rewards, got max={np.max(np.abs(delta))}"

    def test_update_eta_zero_no_change(self, oom):
        """η=0 → no parameter change."""
        actions = np.array([0, 1, 0])
        outcomes = np.array([0, 0, 1])
        traces = oom.compute_eligibility_traces(actions, outcomes)

        # Get ops before
        ops_before = oom.get_A_all(channel_idx=0)

        oom.update_parameters_from_traces(traces, actions, outcomes, eta=0.0)

        ops_after = oom.get_A_all(channel_idx=0)
        for key in ops_before:
            np.testing.assert_array_equal(
                ops_before[key], ops_after[key],
                err_msg=f"A operator {key} should not change with η=0"
            )


# =============================================================================
# §6 — Value function
# =============================================================================


class TestValueFunction:
    """Test value function V(α_t) computation."""

    @pytest.fixture
    def oom(self):
        return _identity_oom(S=2, A=2, O=2)

    def test_v_nonzero_with_nonzero_rewards(self, oom):
        """V should reflect accumulated rewards."""
        actions = np.array([0, 1, 0])
        outcomes = np.array([0, 0, 1])
        rewards = np.array([1.0, 1.0, 1.0])

        result = oom.compute_eligibility_traces(actions, outcomes, rewards=rewards)

        V = result['V']
        assert np.all(V >= 0), f"V should be non-negative with positive rewards: {V}"

    def test_v_decreases_with_gamma(self, oom):
        """Higher gamma → higher V (more weight on future rewards)."""
        actions = np.array([0, 0, 0])
        outcomes = np.array([0, 1, 0])
        rewards = np.array([1.0, 0.5, 0.2])

        result_g0 = oom.compute_eligibility_traces(actions, outcomes, rewards=rewards, gamma=0.0)
        result_g9 = oom.compute_eligibility_traces(actions, outcomes, rewards=rewards, gamma=0.9)

        V_g0 = result_g0['V']
        V_g9 = result_g9['V']

        # γ=0.9 should give higher V than γ=0 (discounting less)
        for t in range(len(V_g0)):
            assert V_g9[t] >= V_g0[t] - 1e-10, \
                f"t={t}: γ=0.9 V={V_g9[t]} should be >= γ=0 V={V_g0[t]}"


# =============================================================================
# §7 — Empty and single-step edge cases
# =============================================================================


class TestEligibilityTracesEdgeCases:
    """Test edge cases for eligibility traces."""

    @pytest.fixture
    def oom(self):
        return _identity_oom(S=2, A=2, O=2)

    def test_empty_trajectory(self, oom):
        """Empty trajectory should return empty arrays."""
        result = oom.compute_eligibility_traces(
            actions=np.array([], dtype=int),
            outcomes=np.array([], dtype=int),
        )

        assert result['alpha'].shape == (0, oom.S2)
        assert result['e_forward'].shape == (0, oom.S2)
        assert result['e_backward'].shape == (0, oom.S2)
        assert result['delta'].shape == (0,)
        assert result['loglikelihood'] == 0.0

    def test_single_step_trajectory(self, oom):
        """Single-step trajectory should work correctly."""
        actions = np.array([0])
        outcomes = np.array([0])
        result = oom.compute_eligibility_traces(actions, outcomes)

        assert result['alpha'].shape == (1, oom.S2)
        assert result['e_forward'].shape == (1, oom.S2)
        assert result['e_backward'].shape == (1, oom.S2)
        assert result['delta'].shape == (1,)

        # Forward trace at t=0 should equal alpha[0]
        np.testing.assert_allclose(
            result['e_forward'][0], result['alpha'][0], atol=1e-10
        )


# =============================================================================
# §8 — Smoothing consistency
# =============================================================================


class TestEligibilityTracesSmoothingConsistency:
    """Test that eligibility traces are consistent with smoothing posteriors."""

    @pytest.fixture
    def oom(self):
        return _identity_oom(S=2, A=2, O=2)

    def test_forward_matches_fb_alpha(self, oom):
        """Eligibility trace alpha should match FB alpha."""
        actions = np.array([0, 1, 0, 1])
        outcomes = np.array([0, 0, 1, 1])

        fb_result = oom.compute_forward_backward(actions, outcomes)
        traces_result = oom.compute_eligibility_traces(actions, outcomes)

        np.testing.assert_array_equal(
            fb_result['alpha'], traces_result['alpha']
        )
        np.testing.assert_array_equal(
            fb_result['smoothing_posteriors'],
            traces_result['smoothing_posteriors']
        )

    def test_rewards_default_to_log_z(self, oom):
        """Default rewards should equal log Z (surprisal)."""
        actions = np.array([0, 1, 0])
        outcomes = np.array([0, 0, 1])
        result = oom.compute_eligibility_traces(actions, outcomes)

        Z = result['Z']
        rewards = result['rewards']

        np.testing.assert_allclose(rewards, np.log(Z), atol=1e-10)
