# Roadmap: QRC-EV Research Improvement Plan

**Date:** 2026-03-31
**Current status:** Working paper — QHMM-20q achieves R²=0.86 on NARMA-10, R²=0.58 on real EV data (vs Ridge R²=0.91). All GPU-simulated. Single EV network.

---

## Executive Summary

The QRC-EV framework demonstrates that stateful quantum reservoir computing with
QHMM-OOM can outperform classical baselines on memory-intensive temporal tasks
(NARMA-10). However, the current work has three critical limitations that
must be addressed before claiming practical quantum advantage: (1) evaluation
on a single real-world dataset, (2) GPU simulation only, and (3) absence of
modern classical competitors. This roadmap addresses these gaps in three
time-horizon phases.

---

## Phase 1 — Near-Term Empirical Strengthening (Weeks 1–6)

These are high-confidence improvements that directly address the paper's
weakest claims.

### 1.1 — Modern Classical Baselines

**Problem:** The paper compares against Ridge, ESN, LSTM, and TFT.
NARMA-10 SOTA is significantly beyond R²=0.86 using classical methods.

\index{baselines!modern}

**Add:**
- **N-BEATS** \citep{oreskanin2020nbeats}: Deep neural basis expansion,
  SOTA on many forecasting benchmarks. Open-source implementation available.
- **Informer** \citep{zhou2021informer}: Transformer variant designed for
  long-sequence time-series forecasting (LSTF). Relevant for $h=10$ results.
- **Neural Hierarchy Encoder (NHiC)**: Alternative deep forecasting method.
- **D-linear** \citep{zerveas2021}: Simple linear attention mechanism that
  achieves competitive results with fewer parameters.

\index{N-BEATS}
\index{Informer}
\index{D-linear}

**Expected impact:** Provides a more honest comparison. N-BEATS/Informer
may achieve NARMA-10 R² > 0.9 classically, which would frame the QHMM
result differently (competitive rather than superior).

\index{NARMA-10!classical SOTA}

### 1.2 — Multiple EV Charging Datasets

**Problem:** Single network (Palo Alto) limits generalizability.

\index{datasets!EV charging!multiple networks}

**Add:**
- **ACN-Data SFO** \citep{acaroglu2022acn}: San Francisco airport network.
- **ChargePoint Data** \citep{chargepoint}: Freely available EV charging
  dataset covering multiple US cities.
- **UK National Grid EV charging data**: Open dataset with national coverage.
- **NEDRA/NEVER**: European EV charging datasets.

\index{ChargePoint}
\index{UK National Grid}

**Expected impact:** Demonstrates whether QHMM advantage on memory-intensive
tasks transfers across geographical contexts and charging behaviors
(residential vs workplace vs fast-charging).

\index{generalization!multiple networks}

### 1.3 — Proper Hyperparameter Optimization

**Problem:** QHMM uses $S=2$, $\lambda_f=\lambda_b=0.8$, $\gamma=0.9$ throughout.
No systematic tuning.

\index{hyperparameter tuning}
\index{QHMM!hyperparameters}

**Add:**
- **State dimension sweep:** $S \in \{2, 3, 4, 6\}$ — larger state space
  may improve EV data performance (currently collapses on sinusoidal
  but may help on real data).
- **Trace decay sweep:** $\lambda \in \{0.0, 0.4, 0.6, 0.8, 0.95\}$
  — the optimal $\lambda$ is task-dependent and currently fixed.
- **Learning rate $\eta$ sweep:** $\eta \in \{0.001, 0.01, 0.1\}$.
- **OMLE convergence criteria:** tighten tolerance from $10^{-6}$ to
  $10^{-8}$, increase max iterations to 200.

\index{state dimension!S sweep}
\index{trace decay!sweep}
\index{OMLE!convergence}

**Use:** Optuna or Bayesian optimization (10-fold cross-validation on
training set) to select hyperparameters. Report tuning curves.

\index{Optuna}
\index{Bayesian optimization}

**Expected impact:** Likely to improve EV data R² from 0.58 to a more
competitive range, and may reveal whether larger $S$ can prevent
sinusoidal collapse.

---

## Phase 2 — Bridging the Simulation-Reality Gap (Weeks 4–10)

These address the most significant concern: results are entirely on GPU
simulation, not physical quantum hardware.

\index{physical quantum hardware}

### 2.1 — Physical QPU Experiments

**Problem:** CUDA Quantum simulation is a proxy for actual quantum hardware.
Gate errors, decoherence, and connectivity constraints are absent.

\index{CUDA Quantum!simulation gap}
\index{decoherence}
\index{gate errors}

**Execute on:**
- **Amazon Braket IonQ Forte**: Trapped-ion QPU with AllToAll connectivity,
  ~30 qubits. Suitable for circuit-based QHMM encoding.
- **Amazon Braket Rigetti Aspen-M2**: Superconducting qubits, gate-based
  operations. Better for nearest-neighbor circuit structures.
- **IBM Quantum** via Qiskit: Largest available QPU network (127 qubits on
  Eagle r3), accessible via the IBM Quantum Platform.

\index{IonQ Forte}
\index{Rigetti Aspen}
\index{IBM Quantum}

**Protocol:**
1. Reproduce NARMA-10 benchmark on QPU (small $nQ \in \{4, 6\}$ due to
   current gate error rates).
2. Measure fidelity degradation vs. GPU simulation at each qubit count.
3. Model the relationship: gate error rate $\epsilon_g \to$ prediction
   accuracy $R^2$.
4. Extrapolate: at what error rate does QHMM advantage disappear?

\index{gate error rate}
\index{quantum advantage!error threshold}

**Expected impact:** This is the single most important experiment for the
paper's credibility. If QHMM advantage persists at IonQ error rates
($\epsilon_g \approx 10^{-3}$ per gate), the claim of quantum advantage
is substantially strengthened. If it degrades significantly, the paper
must frame results as "demonstrating potential quantum advantage under
ideal conditions."

\index{error threshold analysis}

### 2.2 — Circuit Depth and Qubit Connectivity Study

\index{circuit depth}
\index{qubit connectivity}

**On GPU simulation, add:**
- **Depth sweep:** For fixed $nQ=8$, vary circuit depth $d \in \{1, 2, 4, 8\}$.
  Studies whether deeper circuits (more entanglement per step) improve or
  degrade performance.
- **Connectivity constraint:** Compare fully-connected topology vs. linear
  nearest-neighbor (superconducting qubit architecture). Quantifies the
  cost of realistic connectivity constraints.

\index{entanglement!depth scaling}
\index{nearest-neighbor connectivity}

---

## Phase 3 — Theoretical Contributions (Months 2–4)

Strengthening the theoretical foundation increases the paper's impact
beyond an engineering demonstration.

\index{theory}

### 3.1 — Convergence Analysis for TD($\lambda$) on QHMM

\index{TD($\lambda$)!convergence proof}

**Target:** Prove that the combined forward-backward TD($\lambda$) update
converges to a fixed point under standard stochastic approximation
conditions \citep{sutton2000review}:

1. Bounded rewards: $\|r_t\| \leq R_{\max}$.
2. Contraction: $\exists \gamma \in [0,1)$ such that
   $\|V_w - V_{w^*}\| \leq \gamma^t \|V_0 - V_{w^*}\|$ for the
   learned value function.
3. Step-size schedule: $\sum \eta_t = \infty$, $\sum \eta_t^2 < \infty$.

\index{stochastic approximation conditions}

**Current gap:** The combined trace update is an ad-hoc combination of
forward and backward views. While empirically stable, there is no
theoretical guarantee of convergence to the optimal value function.

\index{convergence guarantee!TD(λ)}

**Deliverable:** Theorem with proof in Appendix B, with empirical
verification on synthetic MDPs.

### 3.2 — Memory Capacity Bound for Stateful QHMM

\index{memory capacity!theoretical bound}
\index{QHMM!memory bound}

**Target:** Prove a bound on the effective memory capacity (MC) of the
stateful QHMM in terms of:
- Number of qubits $nQ$
- Depolarizing channel parameter $p_{\mathrm{depol}}$
- State dimension $S$

**Reference:** Classical ESN memory capacity is bounded by the
logarithmic largest singular value of the reservoir matrix
\citep{jaeger2004harnessing}. The quantum analog should relate to the
spectral gap of the dissipative channel $\mathcal{E}$.

\index{spectral gap}
\index{dissipative channel!spectral gap}

**Conjecture:** For a depolarizing channel with parameter $p$:
\index{depolarizing channel!memory lifetime}
\begin{equation}
\mathrm{MC}_{\mathrm{QHMM}} \approx \frac{nQ}{-\log(p)} \quad \text{bits},
\end{equation}
i.e., the memory lifetime scales inversely with the depolarizing rate,
and linearly with qubit count. This is consistent with the observed
$nQ \approx 8$ cross-over point for NARMA-10 (memory-10).

\index{memory lifetime!depolarizing rate}

**Expected impact:** Provides a principled basis for qubit count selection
and reframes the paper from "empirical observation" to "theoretically
grounded design guideline."

\index{design guideline!qubit selection}

### 3.3 — Sample Complexity Bound for OMLE

\index{sample complexity!OMLE}
\index{OMLE!sample complexity}

**Target:** Bound the number of trajectories $K$ needed for OMLE to achieve
$\epsilon$-accurate channel estimation with high probability.

**Approach:** Use the fact that the OOM log-likelihood is concave in the
Choi matrices under CPTP constraints. Apply standard statistical learning
theory for concave MLE: $\ell(\hat\theta) - \ell(\theta^*) \leq$
$\mathcal{O}\bigl(\sqrt{\frac{d}{K}}\bigr)$ where $d = O(S^4)$ is the
parameter space dimension.

\index{MLE!concave}
\index{statistical learning theory}

---

## Phase 4 — Methodological Extensions (Months 3–6)

\index{methodological improvements}

### 4.1 — Learned State Discretization

\index{disambiguation!learned}
\index{vector quantization}
\index{codebook}

**Problem:** Simple quantile binning discards information.

\index{bucket quantization}

**Approach:**
1. **Vector quantization (VQ):** Learn a codebook $C = \{c_1, \ldots, c_O\}$
   of $O$ representative reservoir states via $k$-means++ on the training
   state trajectory. Map each state $\rho_t$ to its nearest codebook entry.
   This preserves the geometric structure of the state space.

   \index{k-means++}

2. **Gumbel-softmax relaxation:** During training, use the Gumbel-Softmax
   trick to make the discretization differentiable, allowing gradient
   backpropagation through the binning step. This enables joint
   optimization of reservoir parameters and discretization.

   \index{Gumbel-softmax}
   \index{differentiable discretization}

3. **Attention-based discretization:** Replace the fixed binning with a
   small attention network that maps reservoir features to outcome
   probabilities: $p(o_t \mid \rho_t) = \mathrm{softmax}(W_{\mathrm{disc}}\, \mathrm{vec}(\rho_t))$.

   \index{attention!discretization}

**Expected impact:** May significantly improve EV data performance (currently
R²=0.58) by preserving fine-grained state information that is discarded
by quantile binning.

\index{EV charging!improved discretization}

### 4.2 — Hierarchical QHMM for Multi-Resolution Forecasting

\index{hierarchical QHMM}
\index{multi-resolution}

**Motivation:** EV charging data has simultaneous daily, weekly, and
anomaly-level periodicities. A single QHMM with $S=2$ cannot capture
all three simultaneously.

\index{multi-scale!QHMM}

**Approach:** Stack two QHMMs:
- **Fast QHMM:** High-resolution (hourly) states for daily dynamics.
- **Slow QHMM:** Low-resolution (daily) states for weekly patterns.
- **Coupling:** Slow QHMM modifies fast QHMM's transition operators as a
  function of day-of-week, enabling context-dependent dynamics.

\index{coupled oscillators!QHMM}
\index{multi-timescale}

### 4.3 — Uncertainty Quantification

\index{uncertainty quantification}
\index{prediction intervals}

**Motivation:** Point predictions are insufficient for grid operations.
Operators need confidence intervals.

\index{grid operations!confidence intervals}

**Approach:**
1. **Ensemble QHMM:** Train $M=5$ QHMMs with different random seeds.
   Ensemble prediction $\pm$ std across seeds gives a crude uncertainty estimate.

   \index{ensemble}

2. **Dropout QHMM:** Apply dropout (random instrument channel masking) during
   inference. Monte Carlo over dropout masks approximates Bayesian posterior
   over QHMM parameters.

   \index{dropout!Bayesian}
   \index{Monte Carlo dropout}

3. **Conformal prediction:** Model-agnostic uncertainty using exchangeability
   and hold-out calibration sets. Produces finite-sample valid prediction
   intervals without distributional assumptions.

   \index{conformal prediction}

**Deliverable:** Prediction intervals at 80\% and 95\% coverage levels.
Evaluate using coverage probability and interval width on test sets.

\index{coverage probability}
\index{interval width}

---

## Phase 5 — Practical Deployment (Months 4–8)

\index{practical deployment}

### 5.1 — Integration with RL-Based EV Charging Scheduling

\index{reinforcement learning!EV charging}
\index{peak shaving}
\index{grid load balancing}

**Motivation:** Forecasting is only useful if it drives decisions.

**Pipeline:**
\index{forecasting-scheduling pipeline}
1. QHMM forecasts 24-hour EV load profile $\hat{y}_{t+1:t+H}$.
2. RL agent (Proximal Policy Optimization, PPO) uses $\hat{y}$ as
   input features to decide charging schedule:
   - Which stations to prioritize during peak hours.
   - How much load to shift to off-peak periods.
   - Battery degradation cost model.
3. Evaluate using: peak-to-average ratio (PAR), customer waiting time,
   and grid stability metrics.

\index{PPO}
\index{peak-to-average ratio}

**Baselines:** Compare against rule-based scheduling (e.g., dumb charging)
and linear model-based MPC.

\index{rule-based scheduling}
\index{MPC}

**Expected impact:** Demonstrates practical utility beyond benchmark R².
This is the most compelling demonstration for the energy infrastructure
community.

\index{energy infrastructure}

### 5.2 — Cost-Benefit Analysis

\index{cost-benefit analysis}
\index{quantum computing!cost}

**Current gap:** The paper does not address the economic viability of QRC.

**Compute:**
\index{compute cost}
- GPU simulation: ~\$0.50/hr on AWS p4d (NVIDIA A100).
  QHMM-20q NARMA-10 training: ~20 min = ~\$0.17.
- Physical QPU on Braket: ~\$1.50/sec for IonQ Forte.
  Equivalent experiment: ~\$1,800.
- Classical ESN: ~\$0.01 on CPU.

\index{AWS pricing}
\index{IonQ Forte pricing}

**Framework:**
\index{quantum advantage!economic threshold}
\begin{align}
\text{Break-even accuracy gain} &=
  \frac{\text{QPU cost} - \text{Classical cost}}
       {\text{Value of 1\% R² improvement for grid operator}} .
\end{align}

\index{R² value}

If a 1\% R² improvement is worth \$10K/year per grid node, and QPU cost
is \$1,800 vs classical \$0.01, the break-even gain is 0.18\% — achievable
for some applications but not all.

\index{economic analysis}

### 5.3 — Streaming/Online Learning Mode

\index{online learning}
\index{streaming}

**Motivation:** EV charging infrastructure is non-stationary:
charging patterns change with seasons, EV fleet composition, and user behavior.

\index{non-stationary}

**Add:**
- **Incremental OMLE:** Replace batch SDP with online SDP updates using
  the stochastic approximation method (dual averaging).
- **Concept drift detection:** Monitor $\bar{Z}_t$ (local model quality)
  from the forward-backward algorithm. If $\bar{Z}_t$ drops below a
  threshold, trigger OMLE re-estimation.
- **TD($\lambda$) with eligibility decay:** As the environment changes,
  older traces should decay faster. Adapt $\lambda$ based on drift magnitude.

\index{concept drift}
\index{dual averaging}

\index{adaptive learning rate}

---

## Phase 6 — Paper and Submission (Months 5–8)

\index{paper submission}

### 6.1 — Target Venues

\index{conference submission}
\index{journal submission}

**Quantum ML / Physics:**
- \textbf{PRX Quantum} (Physical Review X, Q1): Strong quantum computing
  track, high impact. Best fit for theory-heavy revision.
- \textbf{Quantum} (MDPI, open access): Faster review, broader scope.
- \textbf{NeurIPS Quantum Computing Track}: Top ML venue, quantum track is
  competitive but lower theory expectations.

\index{PRX Quantum}
\index{NeurIPS}

**Time-Series / ML:**
- \textbf{UAI} (Uncertainty in AI): Strong statistics and evaluation norms.
- \textbf{AISTATS}: Good for applied ML with uncertainty quantification.
- \textbf{ICML} Time-Series Workshop: Good for forecasting-specific
  exposure.

\index{UAI}
\index{ICML Time-Series}

**Energy / Systems:**
- \textbf{IEEE Transactions on Smart Grid}: High practical impact,
  review may be slower. Best for application-community credibility.

\index{IEEE Smart Grid}

**Recommended:** Submit to PRX Quantum (highest impact) with a draft
targeting the quantum ML audience, supplemented by a preprint on arXiv
for the ML and energy communities.

\index{arXiv}

### 6.2 — Preprint

\index{arXiv!preprint}

Upload to arXiv immediately after internal review. Include:
- Full paper PDF
- Code repository link
- All experiment outputs and hyperparameter configs

\index{reproducibility}

### 6.3 — Response to Referees (Mock)

Prepare pre-emptive responses to likely reviewer concerns:

\index{reviewer concerns}

| Concern | Pre-emptive response |
|---|---|
| "Only GPU simulation" | Phase 2 experiments on physical QPU. |
| "Ridge outperforms on real data" | Phase 1 HPO + learned discretization |
| "No modern baselines" | Phase 1.1 N-BEATS, Informer, D-linear |
| "QHMM is just an HMM" | Ch2 theory: CPTP constraints + quantum channel structure |
| "Small dataset" | Phase 1.2: 3+ EV networks, N=50K+ per dataset |
| "No uncertainty quantification" | Phase 4.3: conformal prediction intervals |
| "Expensive for marginal gains" | Phase 5.2: economic cost-benefit analysis |

\index{reviewer responses}

---

## Summary: Prioritized Action Items

\index{priority}

### Immediately (This Week)
1. [ ] Add N-BEATS and D-linear baselines to experiments
2. [ ] Run systematic HPO sweep for QHMM (S, λ, η)
3. [ ] Upload current draft to arXiv as preprint

### Weeks 1–4
4. [ ] Download and process 2 additional EV datasets
5. [ ] Complete Phase 1 HPO, report tuning curves
6. [ ] Draft Phase 2 QPU experiment protocol

### Months 1–3
7. [ ] Physical QPU runs on IonQ Forte or Rigetti Aspen
8. [ ] Implement learned state discretization (VQ or Gumbel-softmax)
9. [ ] Derive and verify memory capacity bound (Phase 3.2)
10. [ ] Add uncertainty quantification (conformal prediction)

### Months 3–6
11. [ ] RL charging scheduler integration (Phase 5.1)
12. [ ] Economic cost-benefit analysis (Phase 5.2)
13. [ ] Submit to PRX Quantum

---

## Budget Estimate

\index{budget}

| Item | Estimated Cost |
|---|---|
| AWS GPU compute (experiments) | \$200–500 |
| Amazon Braket QPU time (IonQ, 10 hours) | \$1,500 |
| IBM Quantum (free tier available) | \$0 |
| Graduate RA time (6 months) | \$15,000 |
| Conference travel | \$3,000 |
| **Total** | **\$20,000–25,000** |

\index{budget!total}

---

## Risks and Mitigations

\index{risks}

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| QHMM advantage disappears on real QPU | High | Severe | Frame as "potential quantum advantage, hardware validation needed" |
| Learned discretization doesn't help | Medium | Moderate | Fall back to quantile binning + larger $S$ sweep |
| EV data too linear for quantum advantage | High | Moderate | Reframe: QHMM is best for memory tasks, Ridge dominates on linear tasks |
| Submission rejected from top venue | Medium | Moderate | Revise and submit to IEEE Smart Grid or Quantum (MDPI) |
| Hyperparameter sweep too expensive | Low | Low | Use Bayesian optimization, reduce sweep grid |

\index{mitigations}
