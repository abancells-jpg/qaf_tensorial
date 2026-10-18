#!/usr/bin/env python3
"""
Preliminary benchmark for QAF validation.

IMPORTANT SCIENTIFIC SCOPE
This code does NOT implement the full MPS/TDVP QAF stack described in the QAF paper.
It implements a QAF-inspired contextual QUBO proxy designed to test one narrow
hypothesis: whether contextual pair relationships (including mutually exclusive /
negative correlations suggestive of decoys) improve allocation quality under
uncertainty when the downstream optimizer is held fixed.

Baselines:
1. Greedy rules
2. MILP (SciPy/HiGHS)
3. Standard QUBO + local simulated annealing
4. Context-aware MILP ablation (same contextual marginal beliefs as QAF proxy)
5. QAF-inspired contextual QUBO + same simulated annealing
6. Hybrid-BQM local surrogate (greedy seed + multi-start SA local repair)

A D-Wave Leap Hybrid adapter is included but is only executed when Ocean is
installed and a configured Leap credential is available. It is not required for
this local benchmark.

The benchmark uses synthetic scenarios; results are evidence about this synthetic
model, not operational defense performance.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import csr_matrix
from numba import njit

#
# Scenario model
#

@dataclass
class Scenario:
    N: int
    M: int
    uncertainty: float
    seed: int
    truth: np.ndarray # (N,) 1 real, 0 decoy
    damage: np.ndarray # (N,)
    effectiveness: np.ndarray # (N,M)
    resource_cost: np.ndarray # (M,)
    p_base: np.ndarray # (N,) conventional fused confidence
    corr: np.ndarray # (N,N), contextual pair relation in [-1,1]
    spoofed: np.ndarray # (N,) bool

    @property
    def K(self) -> int:
        return self.N * self.M


def _sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def _logit(p):
    p = np.clip(p, 1e-5, 1-1e-5)
    return np.log(p/(1-p))

def generate_scenario(N: int, M: int, uncertainty: float, seed: int) -> Scenario:
    """Generate a synthetic threat/resource allocation scenario.
    The generator intentionally contains adversarial/spoofed decoys. Some spoofed
    decoys are strongly negatively related to a higher-confidence real threat.
    This creates a controlled test for contextual reasoning.
    """
    rng = np.random.default_rng(seed)
    uncertainty = float(np.clip(uncertainty, 0.0, 0.9))

    sensor_weights = np.array([0.42, 0.34, 0.24])
    decoy_rate = 0.20 + 0.25 * uncertainty
    truth = (rng.random(N) > decoy_rate).astype(np.int8)

    # Ensure both classes and enough real threats to make allocation meaningful.
    if truth.sum() < max(1, min(M, N // 2)):
        truth[np.argsort(rng.random(N))[:max(1, min(M, N // 2))]] = 1
    if truth.sum() == N and N >= 3:
        truth[rng.integers(0, N)] = 0

    # Latent damage if the signal were a real threat. Decoys can look dangerous.
    damage = rng.uniform(0.30, 1.00, N)
    effectiveness = rng.uniform(0.55, 0.97, (N, M))
    resource_cost = rng.uniform(0.025, 0.080, M)

    # Three synthetic sensors. Uncertainty reduces class separation and increases missingness.
    scores = np.zeros((N, 3), dtype=float)
    missing = rng.random((N, 3)) < (0.03 + 0.22 * uncertainty)

    # Base means compress toward 0.5 as uncertainty increases.
    real_mean = 0.88 - 0.30 * uncertainty
    decoy_mean = 0.18 + 0.28 * uncertainty
    concentration = max(4.0, 18.0 * (1.0 - 0.65 * uncertainty))

    for i in range(N):
        mean = real_mean if truth[i] else decoy_mean
        a = max(0.5, mean * concentration)
        b = max(0.5, (1 - mean) * concentration)
        scores[i] = rng.beta(a, b, 3)

    decoys = np.where(truth == 0)[0]
    # Spoof a subset of decoys so their raw sensor confidence resembles a real threat.
    spoofed = np.zeros(N, dtype=bool)

    if len(decoys):
        n_spoof = max(1, int(round(len(decoys) * (0.35 + 0.35 * uncertainty))))
        spoof_idx = rng.choice(decoys, size=min(n_spoof, len(decoys)), replace=False)
        spoofed[spoof_idx] = True
        spoof_mean = 0.72 - 0.08 * uncertainty
        a = spoof_mean * concentration
        b = (1 - spoof_mean) * concentration
        scores[spoof_idx] = rng.beta(max(a, 5), max(b, 5), (len(spoof_idx), 3))

    scores[missing] = np.nan
    weighted_num = np.nansum(scores * sensor_weights[None, :], axis=1)
    weighted_den = np.nansum((~np.isnan(scores)) * sensor_weights[None, :], axis=1)

    p_base = np.divide(weighted_num, weighted_den, out=np.full(N, 0.5), where=weighted_den > 0)
    p_base = np.clip(p_base, 0.02, 0.98)

    # Context graph. Negative links connect spoofed decoys to stronger real candidates.
    corr = np.zeros((N, N), dtype=float)
    real_idx = np.where(truth == 1)[0]

    for i in np.where(spoofed)[0]:
        if len(real_idx):
            # Pair with a high-confidence real threat if available.
            j = real_idx[np.argmax(p_base[real_idx] * damage[real_idx])]
            if i != j:
                c = rng.uniform(-0.95, -0.65)
                corr[i, j] = corr[j, i] = c

    # Add a sparse set of positive contextual relationships between real threats.
    if len(real_idx) >= 2:
        n_pos = max(1, int(0.05 * N))
        for _ in range(n_pos):
            i, j = rng.choice(real_idx, size=2, replace=False)
            c = rng.uniform(0.20, 0.55)
            corr[i, j] = corr[j, i] = max(corr[i, j], c)

    return Scenario(N, M, uncertainty, seed, truth, damage, effectiveness, resource_cost, p_base, corr, spoofed)

def qaf_context_belief(s: Scenario, gamma_neg: float = 4.0, gamma_pos: float = 0.8) -> np.ndarray:
    """QAF-inspired contextual belief proxy.
    Negative pair relations act competitively: of two mutually inconsistent signals,
    the stronger one is reinforced and the weaker one is suppressed. Positive links
    provide modest contextual support.
    """
    p = s.p_base
    ctx = np.zeros(s.N)

    for i in range(s.N):
        for j in range(s.N):
            c = s.corr[i, j]
            if c < 0:
                # Competitive contextual update. If p_i > p_j, context i decreases.
                ctx[i] += (-c) * (p[i] - p[j])
            elif c > 0:
                ctx[i] += c * (p[j] - 0.5)

    # Normalize by degree to avoid graph-size effects.
    degree = np.maximum(1.0, np.sum(np.abs(s.corr), axis=1))
    ctx /= degree

    # Use separate gain implicitly via sign.
    gain = np.where(ctx < 0, gamma_neg, gamma_pos)
    return np.clip(_sigmoid(_logit(p) + gain * ctx), 0.005, 0.995)

def expected_scores(s: Scenario, p: np.ndarray) -> np.ndarray:
    return p[:, None] * s.damage[:, None] * s.effectiveness - s.resource_cost[None, :]

def oracle_scores(s: Scenario) -> np.ndarray:
    return s.truth[:, None] * s.damage[:, None] * s.effectiveness - s.resource_cost[None, :]

#
# Evaluation utilities
#

def x_to_matrix(x: np.ndarray, N: int, M: int) -> np.ndarray:
    return np.asarray(x, dtype=np.int8).reshape(N, M)

def repair_assignment(x: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Repair to at most one resource per threat and one threat per resource."""
    N, M = scores.shape
    mat = x_to_matrix(x, N, M).copy()

    # Iteratively keep highest-score conflict until feasible.
    changed = True
    while changed:
        changed = False
        for i in range(N):
            js = np.where(mat[i] > 0)[0]
            if len(js) > 1:
                keep = js[np.argmax(scores[i, js])]
                mat[i, js] = 0
                mat[i, keep] = 1
                changed = True
        for r in range(M):
            is_ = np.where(mat[:, r] > 0)[0]
            if len(is_) > 1:
                keep = is_[np.argmax(scores[is_, r])]
                mat[is_, r] = 0
                mat[keep, r] = 1
                changed = True
    return mat.ravel()

def actual_reward(s: Scenario, x: np.ndarray) -> float:
    mat = x_to_matrix(x, s.N, s.M)
    prevented = np.sum(mat * (s.truth[:, None] * s.damage[:, None] * s.effectiveness))
    cost = np.sum(mat * s.resource_cost[None, :])
    return float(prevented - cost)

def damage_prevented_raw(s: Scenario, x: np.ndarray) -> float:
    mat = x_to_matrix(x, s.N, s.M)
    return float(np.sum(mat * (s.truth[:, None] * s.damage[:, None] * s.effectiveness)))

def decoy_waste_rate(s: Scenario, x: np.ndarray) -> float:
    mat = x_to_matrix(x, s.N, s.M)
    selected = mat.sum()
    if selected == 0:
        return 0.0
    decoy_selected = np.sum(mat * (1 - s.truth)[:, None])
    return float(decoy_selected / selected)

def selected_count(s: Scenario, x: np.ndarray) -> int:
    return int(x_to_matrix(x, s.N, s.M).sum())

def solve_oracle(s: Scenario) -> np.ndarray:
    return solve_milp_from_scores(s, oracle_scores(s))[0]

#
# Solvers
#

def solve_greedy(s: Scenario, scores: np.ndarray) -> np.ndarray:
    N, M = scores.shape
    order = np.argsort(scores.ravel())[::-1]
    used_i, used_r = set(), set()
    x = np.zeros(N * M, dtype=np.int8)

    for idx in order:
        i, r = divmod(int(idx), M)
        if scores[i, r] <= 0:
            break
        if i not in used_i and r not in used_r:
            x[idx] = 1
            used_i.add(i)
            used_r.add(r)
            if len(used_r) == M:
                break
    return x

def _assignment_constraints(N: int, M: int):
    K = N * M
    rows, cols, data = [], [], []
    row = 0

    # threat <= 1
    for i in range(N):
        for r in range(M):
            rows.append(row); cols.append(i * M + r); data.append(1.0)
        row += 1

    # resource <= 1
    for r in range(M):
        for i in range(N):
            rows.append(row); cols.append(i * M + r); data.append(1.0)
        row += 1

    A = csr_matrix((data, (rows, cols)), shape=(N + M, K))
    lb = np.zeros(N + M)
    ub = np.ones(N + M)
    return LinearConstraint(A, lb, ub)

def solve_milp_from_scores(s: Scenario, scores: np.ndarray, time_limit: float = 2.0) -> Tuple[np.ndarray, Dict]:
    c = -scores.ravel()
    bounds = Bounds(np.zeros(s.K), np.ones(s.K))
    integrality = np.ones(s.K, dtype=np.int8)
    constraint = _assignment_constraints(s.N, s.M)

    res = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraint,
               options={"time_limit": time_limit, "mip_rel_gap": 1e-8, "presolve": True})

    if res.x is None:
        x = solve_greedy(s, scores)
    else:
        x = (res.x > 0.5).astype(np.int8)

    meta = {
        "milp_success": bool(getattr(res, "success", False)),
        "mip_gap": float(getattr(res, "mip_gap", np.nan)) if getattr(res, "mip_gap", None) is not None else np.nan,
        "mip_node_count": int(getattr(res, "mip_node_count", 0) or 0),
    }
    return x, meta

def build_qubo(s: Scenario, scores: np.ndarray, contextual: bool = False, penalty: float = 2.5, corr_penalty: float = 1.2):
    K, N, M = s.K, s.N, s.M
    h = -scores.ravel().astype(np.float64).copy()
    edges: Dict[Tuple[int, int], float] = {}

    def add_edge(a, b, w):
        if a == b: return
        if a > b: a, b = b, a
        edges[(a, b)] = edges.get((a, b), 0.0) + float(w)

    # At-most-one constraints encoded quadratically.
    for i in range(N):
        vars_i = [i * M + r for r in range(M)]
        for a in range(len(vars_i)):
            for b in range(a + 1, len(vars_i)):
                add_edge(vars_i[a], vars_i[b], penalty)

    for r in range(M):
        vars_r = [i * M + r for i in range(N)]
        for a in range(len(vars_r)):
            for b in range(a + 1, len(vars_r)):
                add_edge(vars_r[a], vars_r[b], penalty)

    # QAF proxy adds contextual incompatibility between negatively related threats.
    if contextual:
        neg_pairs = np.argwhere(np.triu(s.corr < -1e-12, 1))
        for i, j in neg_pairs:
            w = corr_penalty * (-s.corr[i, j])
            for r1 in range(M):
                for r2 in range(M):
                    add_edge(int(i) * M + r1, int(j) * M + r2, w)

    if edges:
        uvw = np.array([(a, b, w) for (a, b), w in edges.items()], dtype=np.float64)
        u = uvw[:, 0].astype(np.int64)
        v = uvw[:, 1].astype(np.int64)
        w = uvw[:, 2].astype(np.float64)
    else:
        u = np.empty(0, np.int64); v = np.empty(0, np.int64); w = np.empty(0, np.float64)

    return h, u, v, w

def edges_to_csr(K: int, u: np.ndarray, v: np.ndarray, w: np.ndarray):
    deg = np.zeros(K, dtype=np.int64)
    for a, b in zip(u, v):
        deg[a] += 1; deg[b] += 1
    indptr = np.zeros(K + 1, dtype=np.int64)
    np.cumsum(deg, out=indptr[1:])
    indices = np.empty(indptr[-1], dtype=np.int64)
    weights = np.empty(indptr[-1], dtype=np.float64)
    cursor = indptr[:-1].copy()

    for a, b, ww in zip(u, v, w):
        pa = cursor[a]; indices[pa] = b; weights[pa] = ww; cursor[a] += 1
        pb = cursor[b]; indices[pb] = a; weights[pb] = ww; cursor[b] += 1

    return indptr, indices, weights

@njit(cache=True)
def _sa_numba(h, indptr, indices, weights, seed, reads, sweeps, beta0, beta1, init_x):
    np.random.seed(seed)
    K = h.shape[0]
    best_x = np.zeros(K, dtype=np.int8)
    best_E = 1e30

    for read in range(reads):
        if read == 0 and init_x.shape[0] == K:
            x = init_x.copy()
        else:
            x = (np.random.random(K) < 0.08).astype(np.int8)

        local = h.copy()
        # local_i = h_i + sum_j w_ij * x_j
        for i in range(K):
            sm = 0.0
            for p in range(indptr[i], indptr[i+1]):
                sm += weights[p] * x[indices[p]]
            local[i] += sm

        # energy, pair terms counted twice in local - h correction; calculate explicitly
        E = 0.0
        for i in range(K):
            E += h[i] * x[i]
        for i in range(K):
            if x[i] == 1:
                for p in range(indptr[i], indptr[i+1]):
                    j = indices[p]
                    if j > i and x[j] == 1:
                        E += weights[p]

        if E < best_E:
            best_E = E; best_x = x.copy()

        total_steps = max(1, sweeps * K)
        for step in range(total_steps):
            i = np.random.randint(0, K)
            frac = step / total_steps
            beta = beta0 + (beta1 - beta0) * frac
            delta = (1 - 2 * x[i]) * local[i]

            if delta <= 0 or np.random.random() < math.exp(-beta * delta):
                old = x[i]
                new = 1 - old
                dx = new - old
                x[i] = new

                for p in range(indptr[i], indptr[i+1]):
                    j = indices[p]
                    local[j] += weights[p] * dx
                E += delta

                if E < best_E:
                    best_E = E; best_x = x.copy()

    return best_x, best_E

def solve_sa(s: Scenario, scores: np.ndarray, contextual: bool, seed: int, reads: int = 6, sweeps: int = 35, hybrid: bool = False):
    h, u, v, w = build_qubo(s, scores, contextual=contextual)
    indptr, indices, weights = edges_to_csr(s.K, u, v, w)

    # Use the same deterministic greedy seed for the first read in all SA variants.
    # This improves fairness: differences then come from the energy model, not initialization.
    init = solve_greedy(s, scores)

    if hybrid:
        reads = max(reads, 10); sweeps = max(sweeps, 55)

    x, E = _sa_numba(h, indptr, indices, weights, int(seed), int(reads), int(sweeps), 0.2, 7.0, init)
    x = repair_assignment(x, scores)
    meta = {"qubo_energy": float(E), "n_edges": int(len(w))}
    return x, meta

#
# Simplified particle-belief planner
#

def solve_particle_belief(s: Scenario, n_particles: int = 250, seed: int = 0) -> Tuple[np.ndarray, Dict]:
    """Simplified POMDP particle-belief baseline for a one-step allocation horizon.
    This is NOT full POMCP. It uses root sampling / particles to estimate contextual
    threat marginals under negative-pair exclusivity and then solves the expected
    allocation with MILP. It is included as a bridge to a future dynamic POMCP test.
    """
    rng = np.random.default_rng(seed)
    p = s.p_base
    samples = rng.random((n_particles, s.N)) < p[None, :]

    # Enforce negative-pair exclusivity probabilistically: if both sampled real, retain
    # the higher-confidence one with probability increasing with |corr|.
    neg_pairs = np.argwhere(np.triu(s.corr < -1e-12, 1))
    for i, j in neg_pairs:
        both = np.where(samples[:, i] & samples[:, j])[0]
        if len(both):
            c = -s.corr[i, j]
            for k in both:
                if rng.random() < c:
                    if p[i] >= p[j]: samples[k, j] = False
                    else: samples[k, i] = False

    p_part = samples.mean(axis=0)
    scores = expected_scores(s, p_part)
    x, meta = solve_milp_from_scores(s, scores)
    meta["particle_count"] = n_particles
    return x, meta

#
# Optional D-Wave Leap adapter
#

def try_dwave_leap(s: Scenario, scores: np.ndarray, contextual: bool = False):
    """Try true D-Wave Leap Hybrid BQM if dependency and credentials exist."""
    try:
        import dimod # type: ignore
        from dwave.system import LeapHybridSampler # type: ignore
    except Exception as e:
        return None, {"executed": False, "reason": f"Ocean unavailable: {e}"}

    h, u, v, w = build_qubo(s, scores, contextual=contextual)
    linear = {i: float(h[i]) for i in range(s.K)}
    quadratic = {(int(a), int(b)): float(ww) for a, b, ww in zip(u, v, w)}
    bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, dimod.BINARY)

    try:
        with LeapHybridSampler() as sampler:
            sampleset = sampler.sample(bqm)
            sample = sampleset.first.sample
            x = np.array([sample[i] for i in range(s.K)], dtype=np.int8)
            x = repair_assignment(x, scores)
            return x, {"executed": True}
    except Exception as e:
        return None, {"executed": False, "reason": str(e)}

#
# Benchmark runner
#

def model_bytes_for(method: str, s: Scenario, scores: np.ndarray, contextual=False) -> int:
    if "QUBO" in method or "Hybrid" in method:
        h, u, v, w = build_qubo(s, scores, contextual=contextual)
        return int(h.nbytes + u.nbytes + v.nbytes + w.nbytes)
    if "MILP" in method or "Particle" in method:
        # Objective + sparse assignment matrix estimate.
        return int(scores.nbytes + (s.K * 2) * 8 + (s.K * 2) * 8)
    return int(scores.nbytes)

def timed_call(fn, *args, **kwargs):
    tracemalloc.start()
    t0 = time.perf_counter_ns()
    out = fn(*args, **kwargs)
    dt_ms = (time.perf_counter_ns() - t0) / 1e6
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return out, dt_ms, peak / (1024**2)

def evaluate_method(method: str, s: Scenario, seed: int):
    base_scores = expected_scores(s, s.p_base)
    p_qaf = qaf_context_belief(s)
    qaf_scores = expected_scores(s, p_qaf)

    if method == "Greedy":
        (ret), rt, py_mem = timed_call(solve_greedy, s, base_scores)
        x = ret; meta = {}; model_bytes = model_bytes_for(method, s, base_scores)
    elif method == "MILP":
        (ret), rt, py_mem = timed_call(solve_milp_from_scores, s, base_scores)
        x, meta = ret; model_bytes = model_bytes_for(method, s, base_scores)
    elif method == "QUBO-SA":
        (ret), rt, py_mem = timed_call(solve_sa, s, base_scores, False, seed, 6, 35, False)
        x, meta = ret; model_bytes = model_bytes_for(method, s, base_scores, False)
    elif method == "Context-MILP":
        (ret), rt, py_mem = timed_call(solve_milp_from_scores, s, qaf_scores)
        x, meta = ret; model_bytes = model_bytes_for(method, s, qaf_scores)
    elif method == "QAF-QUBO-SA":
        (ret), rt, py_mem = timed_call(solve_sa, s, qaf_scores, True, seed, 6, 35, False)
        x, meta = ret; model_bytes = model_bytes_for(method, s, qaf_scores, True)
    elif method == "Hybrid-BQM-local":
        (ret), rt, py_mem = timed_call(solve_sa, s, base_scores, False, seed, 10, 55, True)
        x, meta = ret; model_bytes = model_bytes_for(method, s, base_scores, False)
    elif method == "Particle-POMDP":
        (ret), rt, py_mem = timed_call(solve_particle_belief, s, 250, seed)
        x, meta = ret; model_bytes = model_bytes_for(method, s, base_scores)
    else:
        raise ValueError(method)

    oracle_x = solve_oracle(s)
    oracle_damage = damage_prevented_raw(s, oracle_x)
    oracle_reward = actual_reward(s, oracle_x)

    raw_damage = damage_prevented_raw(s, x)
    reward = actual_reward(s, x)

    quality = 100.0 * raw_damage / oracle_damage if oracle_damage > 1e-12 else 100.0
    utility_quality = 100.0 * reward / oracle_reward if oracle_reward > 1e-12 else 100.0

    return {
        "method": method,
        "N": s.N, "M": s.M, "K": s.K,
        "uncertainty": s.uncertainty, "seed": s.seed,
        "quality_pct_oracle_damage": quality,
        "utility_pct_oracle": utility_quality,
        "damage_prevented": raw_damage,
        "actual_reward": reward,
        "decoy_waste_rate": decoy_waste_rate(s, x),
        "selected": selected_count(s, x),
        "runtime_ms": rt,
        "python_peak_mb": py_mem,
        "model_kb": model_bytes / 1024,
        "deadline_500ms": float(rt <= 500.0),
        "base_brier": float(np.mean((s.p_base - s.truth)**2)),
        "qaf_brier": float(np.mean((p_qaf - s.truth)**2)),
        "spoofed_decoys": int(np.sum(s.spoofed)),
        **meta,
    }

def run_benchmark(out_csv: Path, reps: int = 6):
    # Warm up numba compilation outside measurements.
    warm = generate_scenario(5, 3, 0.2, 123)
    solve_sa(warm, expected_scores(warm, warm.p_base), False, 123, 1, 1, False)

    scales = [(5, 3), (10, 5), (20, 5), (25, 10), (50, 10)] # K = 15, 50, 100, 250, 500
    uncertainties = [0.10, 0.35, 0.60]
    methods = ["Greedy", "MILP", "QUBO-SA", "Context-MILP", "QAF-QUBO-SA", "Hybrid-BQM-local", "Particle-POMDP"]

    rows = []
    total = len(scales) * len(uncertainties) * reps
    done = 0

    for N, M in scales:
        for u in uncertainties:
            for rep in range(reps):
                seed = 10000 + (N * 1000 + M * 100) + int(u * 100) * 10 + rep
                s = generate_scenario(N, M, u, seed)
                for mi, method in enumerate(methods):
                    row = evaluate_method(method, s, seed + mi * 97)
                    rows.append(row)
                done += 1
                if done % 10 == 0:
                    print(f"Completed scenarios: {done}/{total}")

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df

#
# Plots
#

def make_plots(df: pd.DataFrame, figdir: Path):
    import matplotlib.pyplot as plt
    figdir.mkdir(parents=True, exist_ok=True)
    preferred = ["Greedy", "MILP", "QUBO-SA", "Context-MILP", "QAF-QUBO-SA", "Hybrid-BQM-local", "Particle-POMDP"]

    # 1 overall quality
    g = df.groupby('method')['quality_pct_oracle_damage'].agg(['mean', 'std']).reindex(preferred)
    fig, ax = plt.subplots(figsize=(9, 5)); ax.bar(g.index, g['mean'], yerr=g['std'], capsize=3); ax.set_ylabel('% del dano optimo evitado'); ax.set_title('1. Calidad global de decision'); ax.tick_params(axis='x', rotation=35); ax.set_ylim(0, 105); fig.tight_layout(); fig.savefig(figdir / '01_calidad_global.png', dpi=180); plt.close(fig)

    # 2 waste rate
    g = df.groupby('method')['decoy_waste_rate'].mean().reindex(preferred) * 100
    fig, ax = plt.subplots(figsize=(9, 5)); ax.bar(g.index, g.values); ax.set_ylabel('Recursos asignados a senuelos (%)'); ax.set_title('2. Desperdicio de recursos en senuelos'); ax.tick_params(axis='x', rotation=35); fig.tight_layout(); fig.savefig(figdir / '02_senuelos.png', dpi=180); plt.close(fig)

    # 3 runtime vs K (median, log y)
    g = df.groupby(['method', 'K'])['runtime_ms'].median().reset_index()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for m in preferred:
        z = g[g.method == m]; ax.plot(z.K, z.runtime_ms, marker='o', label=m)
    ax.axhline(500, ls='--', lw=1, label='500 ms'); ax.set_yscale('log'); ax.set_xlabel('Variables K'); ax.set_ylabel('Latencia mediana (ms, escala log)'); ax.set_title('3. Escalabilidad temporal'); ax.legend(fontsize=7, ncol=2); fig.tight_layout(); fig.savefig(figdir / '03_runtime_vs_k.png', dpi=180); plt.close(fig)

    # 4 model memory vs K
    g = df.groupby(['method', 'K'])['model_kb'].median().reset_index()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for m in preferred:
        z = g[g.method == m]; ax.plot(z.K, z.model_kb, marker='o', label=m)
    ax.set_yscale('log'); ax.set_xlabel('Variables K'); ax.set_ylabel('Tamano estimado del modelo (KB, log)'); ax.set_title('4. Escalabilidad de representacion'); ax.legend(fontsize=7, ncol=2); fig.tight_layout(); fig.savefig(figdir / '04_model_memory.png', dpi=180); plt.close(fig)

    # 5 deadline compliance by K
    g = df.groupby(['method', 'K'])['deadline_500ms'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for m in preferred:
        z = g[g.method == m]; ax.plot(z.K, z.deadline_500ms * 100, marker='o', label=m)
    ax.set_xlabel('Variables K'); ax.set_ylabel('Corridas <= 500 ms (%)'); ax.set_ylim(-2, 102); ax.set_title('5. Cumplimiento de ventana de 500 ms'); ax.legend(fontsize=7, ncol=2); fig.tight_layout(); fig.savefig(figdir / '05_deadline.png', dpi=180); plt.close(fig)

    # 6 quality vs uncertainty
    g = df.groupby(['method', 'uncertainty'])['quality_pct_oracle_damage'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for m in preferred:
        z = g[g.method == m]; ax.plot(z.uncertainty * 100, z.quality_pct_oracle_damage, marker='o', label=m)
    ax.set_xlabel('Incertidumbre sintetica (%)'); ax.set_ylabel('% del dano optimo evitado'); ax.set_title('6. Robustez frente a incertidumbre'); ax.legend(fontsize=7, ncol=2); fig.tight_layout(); fig.savefig(figdir / '06_quality_uncertainty.png', dpi=180); plt.close(fig)

    # 7 QAF relative improvement vs key baselines across uncertainty
    agg = df.groupby(['method', 'uncertainty'])['quality_pct_oracle_damage'].mean().unstack('method')
    fig, ax = plt.subplots(figsize=(9, 5))
    for b in ['Greedy', 'MILP', 'QUBO-SA', 'Particle-POMDP']:
        imp = (agg['QAF-QUBO-SA'] - agg[b]) / agg[b] * 100
        ax.plot(agg.index * 100, imp.values, marker='o', label=f'vs {b}')
    ax.axhline(0, lw=1); ax.set_xlabel('Incertidumbre sintetica (%)'); ax.set_ylabel('Mejora relativa QAF (%)'); ax.set_title('7. Ventaja relativa del proxy QAF'); ax.legend(); fig.tight_layout(); fig.savefig(figdir / '07_qaf_improvement.png', dpi=180); plt.close(fig)

    # 8 belief calibration brier
    b = df[['seed', 'N', 'M', 'uncertainty', 'base_brier', 'qaf_brier']].drop_duplicates()
    g = b.groupby('uncertainty')[['base_brier', 'qaf_brier']].mean()
    fig, ax = plt.subplots(figsize=(8, 5)); ax.plot(g.index * 100, g.base_brier, marker='o', label='Fusion convencional'); ax.plot(g.index * 100, g.qaf_brier, marker='o', label='Contexto QAF proxy'); ax.set_xlabel('Incertidumbre sintetica (%)'); ax.set_ylabel('Brier score (menor es mejor)'); ax.set_title('8. Calibracion de confianza'); ax.legend(); fig.tight_layout(); fig.savefig(figdir / '08_brier.png', dpi=180); plt.close(fig)

    # 9 runtime-quality frontier at K=500
    z = df[df.K == 500].groupby('method').agg(runtime_ms=('runtime_ms', 'median'), quality=('quality_pct_oracle_damage', 'mean')).reindex(preferred)
    fig, ax = plt.subplots(figsize=(8, 5.5)); ax.scatter(z.runtime_ms, z.quality, s=65)
    for m, row in z.iterrows(): ax.annotate(m, (row.runtime_ms, row.quality), xytext=(4, 4), textcoords='offset points', fontsize=7)
    ax.set_xscale('log'); ax.axvline(500, ls='--', lw=1); ax.set_xlabel('Latencia mediana K=500 (ms, log)'); ax.set_ylabel('% del dano optimo evitado'); ax.set_title('9. Frontera calidad-latencia en K=500'); fig.tight_layout(); fig.savefig(figdir / '09_frontier_k500.png', dpi=180); plt.close(fig)

    # 10 benchmark architecture diagram
    fig, ax = plt.subplots(figsize=(10, 6)); ax.axis('off')
    boxes = [(0.06, 0.68, 0.18, 0.16, 'Escenario\nsintetico'), (0.31, 0.68, 0.18, 0.16, 'Fusion de\nsensores'), (0.56, 0.68, 0.18, 0.16, 'Metodos\ncomparados'), (0.81, 0.68, 0.14, 0.16, 'Decision\nx')]
    for x, y, w, h, t in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, lw=1.5)); ax.text(x + w / 2, y + h / 2, t, ha='center', va='center', fontsize=11)
    for x1, x2 in [(0.24, 0.31), (0.49, 0.56), (0.74, 0.81)]: ax.annotate('', xy=(x2, 0.76), xytext=(x1, 0.76), arrowprops=dict(arrowstyle='->'))
    methods = 'Greedy\nMILP / HiGHS\nQUBO + SA\nContext-MILP\nQAF-QUBO + SA\nHybrid-BQM local\nParticle-POMDP'
    ax.text(0.65, 0.47, methods, ha='center', va='center', fontsize=10, bbox=dict(boxstyle='round', fill=False))
    ax.annotate('', xy=(0.65, 0.60), xytext=(0.65, 0.66), arrowprops=dict(arrowstyle='->'))
    ax.text(0.28, 0.30, 'Ground truth reservado\npara evaluacion', ha='center', va='center', fontsize=10, bbox=dict(boxstyle='round', fill=False))
    ax.text(0.72, 0.25, 'Metricas: calidad, senuelos, \nlatencia, memoria, deadline, \ncalibracion y escalabilidad', ha='center', va='center', fontsize=10, bbox=dict(boxstyle='round', fill=False))
    ax.annotate('', xy=(0.62, 0.31), xytext=(0.38, 0.31), arrowprops=dict(arrowstyle='->'))
    ax.annotate('', xy=(0.76, 0.39), xytext=(0.86, 0.68), arrowprops=dict(arrowstyle='->'))
    ax.set_title('10. Diseno del banco de pruebas', fontsize=14); fig.tight_layout(); fig.savefig(figdir / '10_benchmark_diagram.png', dpi=180); plt.close(fig)

def summary_tables(df: pd.DataFrame, outdir: Path):
    methods = ["Greedy", "MILP", "QUBO-SA", "Context-MILP", "QAF-QUBO-SA", "Hybrid-BQM-local", "Particle-POMDP"]
    overall = df.groupby('method').agg(
        quality_mean=('quality_pct_oracle_damage', 'mean'),
        quality_sd=('quality_pct_oracle_damage', 'std'),
        waste_pct=('decoy_waste_rate', lambda x: 100 * x.mean()),
        runtime_median_ms=('runtime_ms', 'median'),
        runtime_p95_ms=('runtime_ms', lambda x: np.quantile(x, 0.95)),
        deadline_pct=('deadline_500ms', lambda x: 100 * x.mean()),
        model_kb_median=('model_kb', 'median'),
    ).reindex(methods)
    overall.to_csv(outdir / 'summary_overall.csv')

    byk = df.groupby(['method', 'K']).agg(quality=('quality_pct_oracle_damage', 'mean'), runtime_ms=('runtime_ms', 'median'), deadline=('deadline_500ms', 'mean'), waste=('decoy_waste_rate', 'mean')).reset_index()
    byk.to_csv(outdir / 'summary_by_k.csv', index=False)

    byu = df.groupby(['method', 'uncertainty']).agg(quality=('quality_pct_oracle_damage', 'mean'), runtime_ms=('runtime_ms', 'median'), waste=('decoy_waste_rate', 'mean')).reset_index()
    byu.to_csv(outdir / 'summary_by_uncertainty.csv', index=False)

    return overall, byk, byu

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default='/mnt/data/qaf_benchmark')
    ap.add_argument('--reps', type=int, default=6)
    ap.add_argument('--reuse', action='store_true')
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'results').mkdir(exist_ok=True)
    (out / 'figures').mkdir(exist_ok=True)

    csv = out / 'results' / 'benchmark_results.csv'

    if args.reuse and csv.exists():
        df = pd.read_csv(csv)
    else:
        df = run_benchmark(csv, args.reps)

    overall, _, _ = summary_tables(df, out / 'results')
    make_plots(df, out / 'figures')

    print('\nOVERALL SUMMARY')
    print(overall.to_string(float_format=lambda x: f'{x:,.3f}'))
    print(f'\nSaved: {csv}')

if __name__ == '__main__':
    main()
