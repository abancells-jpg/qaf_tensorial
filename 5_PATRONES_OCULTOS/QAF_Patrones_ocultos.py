#!/usr/bin/env python3
"""
QAF Hidden Pattern Discovery Benchmark
Author: Juan Manuel Goenaga
Date: 2026-09-19

This is a QAF-inspired proxy benchmark, not the full MPS/TDVP QAF engine.
It tests whether contextual selection plus a phase/coherence representation
reveals relational patterns whose first-order marginals are deliberately
uninformative.

Ground-truth labels are used only after fitting, for evaluation.
"""

from pathlib import Path
import time
import platform
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.stats import wilcoxon
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

warnings.filterwarnings("ignore")

SEED_BASE = 20260919
K = 4
M = 6
NOISE = [0.10, 0.25, 0.40, 0.60]
HIDDEN = ["phase4", "phase_gain", "context_decoy"]
CONTROL = "visible_control"
N_CLUSTER = 500
REPS = 4
IRR = 16
N_ANOM = 1600
ANOM_REPS = 4
CONTAM = 0.08

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
FIGURES = BASE / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)


def R(th):
    return np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])


def cluster_acc(y, p):
    cm = confusion_matrix(y, p)
    r, c = linear_sum_assignment(-cm)
    return cm[r, c].sum() / cm.sum()


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def gen_latent(n=N_CLUSTER, noise=0.25, irr=IRR, seed=0, family="phase4"):
    rng = np.random.default_rng(seed)
    ctx = rng.integers(0, M, n)
    post = rng.normal(size=(n, 2 * M))
    y = rng.integers(0, K, n)
    pre = rng.normal(size=(n, 2 * M))
    angles = np.array([0, np.pi / 2, np.pi, -np.pi / 2])
    gains = np.array([0.82, 1.00, 1.18, 1.36])

    if family == CONTROL:
        centers = np.array([[-3, -3], [-3, 3], [3, -3], [3, 3]])
        pre[:, :2] = centers[y] + rng.normal(scale=0.45 + noise, size=(n, 2))
        post[:, :2] = pre[:, :2] + rng.normal(scale=0.25 + noise, size=(n, 2))
    else:
        for i in range(n):
            j = int(ctx[i])
            sl = slice(2 * j, 2 * j + 2)
            v = pre[i, sl]
            if family == "phase4":
                post[i, sl] = R(angles[y[i]]) @ v + rng.normal(scale=noise, size=2)
            elif family == "phase_gain":
                post[i, sl] = gains[y[i]] * (R(angles[y[i]]) @ v) + rng.normal(scale=noise, size=2)
            elif family == "context_decoy":
                post[i, sl] = R(angles[y[i]]) @ v + rng.normal(scale=noise, size=2)
                for jj in range(M):
                    if jj == j:
                        continue
                    ssl = slice(2 * jj, 2 * jj + 2)
                    d = rng.integers(0, K)
                    post[i, ssl] = R(angles[d]) @ pre[i, ssl] + rng.normal(scale=noise * 1.05, size=2)

    if irr:
        pre = np.hstack([pre, rng.normal(size=(n, irr))])
        post = np.hstack([post, rng.normal(size=(n, irr))])

    X = np.hstack([pre, post, np.eye(M)[ctx]])
    return X, y, ctx, pre, post


def gen_anom(n=N_ANOM, noise=0.25, irr=IRR, contam=CONTAM, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < contam).astype(int)
    ctx = rng.integers(0, M, n)
    pre = rng.normal(size=(n, 2 * M))
    post = rng.normal(size=(n, 2 * M))
    normal = np.array([0, np.pi / 2, np.pi, -np.pi / 2])
    abnormal = np.array([np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4])
    reg = rng.integers(0, K, n)

    for i in range(n):
        j = int(ctx[i])
        sl = slice(2 * j, 2 * j + 2)
        v = pre[i, sl]
        th = abnormal[rng.integers(0, 4)] if y[i] else normal[reg[i]]
        post[i, sl] = R(th) @ v + rng.normal(scale=noise, size=2)

    if irr:
        pre = np.hstack([pre, rng.normal(size=(n, irr))])
        post = np.hstack([post, rng.normal(size=(n, irr))])

    X = np.hstack([pre, post, np.eye(M)[ctx]])
    return X, y, ctx, pre, post


def qaf(pre, post, ctx):
    f = np.zeros((len(ctx), 2))
    for i, j in enumerate(ctx):
        a = pre[i, 2 * j : 2 * j + 2]
        b = post[i, 2 * j : 2 * j + 2]
        na = np.linalg.norm(a) + 1e-12
        nb = np.linalg.norm(b) + 1e-12
        f[i, 0] = np.dot(a, b) / (na * nb)
        f[i, 1] = (a[0] * b[1] - a[1] * b[0]) / (na * nb)
    return f


def qaf_nophase(pre, post, ctx):
    f = np.zeros((len(ctx), 2))
    for i, j in enumerate(ctx):
        a = pre[i, 2 * j : 2 * j + 2]
        b = post[i, 2 * j : 2 * j + 2]
        na = np.linalg.norm(a) + 1e-12
        nb = np.linalg.norm(b) + 1e-12
        f[i] = [np.dot(a, b) / (na * nb), np.log(nb / na)]
    return f


def qaf_nocontext(pre, post):
    allf = []
    for j in range(M):
        a = pre[:, 2 * j : 2 * j + 2]
        b = post[:, 2 * j : 2 * j + 2]
        na = np.linalg.norm(a, axis=1) + 1e-12
        nb = np.linalg.norm(b, axis=1) + 1e-12
        dot = np.sum(a * b, axis=1) / (na * nb)
        wedge = (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) / (na * nb)
        allf.append(np.c_[dot, wedge])
    return np.mean(np.stack(allf, axis=2), axis=2)


def relational(pre, post, ctx):
    f = np.zeros((len(ctx), 5))
    for i, j in enumerate(ctx):
        a = pre[i, 2 * j : 2 * j + 2]
        b = post[i, 2 * j : 2 * j + 2]
        na = np.linalg.norm(a) + 1e-12
        nb = np.linalg.norm(b) + 1e-12
        dot = np.dot(a, b) / (na * nb)
        f[i] = [dot, np.log(nb / na), abs(a[0] - b[0]), abs(a[1] - b[1]), np.linalg.norm(a - b)]
    return f


def clustering_methods(X, pre, post, ctx, seed):
    out = {}
    raw = StandardScaler().fit_transform(X)

    t = time.perf_counter()
    out["Raw-KMeans"] = (KMeans(K, n_init=12, random_state=seed).fit_predict(raw), (time.perf_counter() - t) * 1000)

    t = time.perf_counter()
    pca = PCA(n_components=min(12, raw.shape[1]), svd_solver="randomized", random_state=seed).fit_transform(raw)
    out["PCA-KMeans"] = (KMeans(K, n_init=12, random_state=seed).fit_predict(pca), (time.perf_counter() - t) * 1000)

    t = time.perf_counter()
    out["Spectral-Raw"] = (
        SpectralClustering(K, affinity="nearest_neighbors", n_neighbors=12, assign_labels="kmeans", random_state=seed).fit_predict(raw),
        (time.perf_counter() - t) * 1000,
    )

    t = time.perf_counter()
    rr = StandardScaler().fit_transform(relational(pre, post, ctx))
    out["Context-Relational"] = (KMeans(K, n_init=15, random_state=seed).fit_predict(rr), (time.perf_counter() - t) * 1000)

    t = time.perf_counter()
    q0 = StandardScaler().fit_transform(qaf_nophase(pre, post, ctx))
    out["QAF-NoPhase"] = (KMeans(K, n_init=15, random_state=seed).fit_predict(q0), (time.perf_counter() - t) * 1000)

    t = time.perf_counter()
    q1 = StandardScaler().fit_transform(qaf_nocontext(pre, post))
    out["QAF-NoContext"] = (KMeans(K, n_init=15, random_state=seed).fit_predict(q1), (time.perf_counter() - t) * 1000)

    t = time.perf_counter()
    qq = qaf(pre, post, ctx)
    out["QAF-Coherence"] = (KMeans(K, n_init=20, random_state=seed).fit_predict(qq), (time.perf_counter() - t) * 1000)

    return out


def run_clustering():
    rows = []
    families = HIDDEN + [CONTROL]
    for family in families:
        for noise in NOISE:
            for rep in range(REPS):
                seed = SEED_BASE + families.index(family) * 100000 + int(noise * 1000) * 100 + rep
                X, y, ctx, pre, post = gen_latent(noise=noise, seed=seed, family=family)
                for method, (pred, rt) in clustering_methods(X, pre, post, ctx, seed).items():
                    rows.append(
                        dict(
                            family=family,
                            noise=noise,
                            rep=rep,
                            seed=seed,
                            method=method,
                            ari=adjusted_rand_score(y, pred),
                            nmi=normalized_mutual_info_score(y, pred),
                            matched_accuracy=cluster_acc(y, pred),
                            runtime_ms=rt,
                        )
                    )
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "clustering_raw.csv", index=False)
    return df


def anomaly_methods(X, pre, post, ctx, seed):
    out = {}
    raw = StandardScaler().fit_transform(X)

    t = time.perf_counter()
    m = IsolationForest(n_estimators=150, contamination=CONTAM, random_state=seed).fit(raw)
    out["IsolationForest-Raw"] = (-m.score_samples(raw), (time.perf_counter() - t) * 1000)

    t = time.perf_counter()
    m = LocalOutlierFactor(n_neighbors=30)
    m.fit_predict(raw)
    out["LOF-Raw"] = (-m.negative_outlier_factor_, (time.perf_counter() - t) * 1000)

    t = time.perf_counter()
    m = OneClassSVM(kernel="rbf", nu=CONTAM, gamma="scale").fit(raw)
    out["OneClassSVM-Raw"] = (-m.score_samples(raw), (time.perf_counter() - t) * 1000)

    t = time.perf_counter()
    rr = StandardScaler().fit_transform(relational(pre, post, ctx))
    m = IsolationForest(n_estimators=150, contamination=CONTAM, random_state=seed).fit(rr)
    out["IsolationForest-Relational"] = (-m.score_samples(rr), (time.perf_counter() - t) * 1000)

    t = time.perf_counter()
    qq = qaf(pre, post, ctx)
    m = GaussianMixture(4, n_init=4, random_state=seed).fit(qq)
    out["QAF-Coherence-GMM"] = (-m.score_samples(qq), (time.perf_counter() - t) * 1000)

    return out


def run_anomaly():
    rows = []
    for noise in NOISE:
        for rep in range(ANOM_REPS):
            seed = SEED_BASE + 500000 + int(noise * 1000) * 100 + rep
            X, y, ctx, pre, post = gen_anom(noise=noise, seed=seed)
            for method, (score, rt) in anomaly_methods(X, pre, post, ctx, seed).items():
                rows.append(
                    dict(
                        noise=noise,
                        rep=rep,
                        seed=seed,
                        method=method,
                        roc_auc=roc_auc_score(y, score),
                        average_precision=average_precision_score(y, score),
                        runtime_ms=rt,
                    )
                )
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "anomaly_raw.csv", index=False)
    return df


def run_irrelevant():
    rows = []
    for irr in [0, 16, 64, 128]:
        for rep in range(4):
            seed = SEED_BASE + 700000 + irr * 10 + rep
            X, y, ctx, pre, post = gen_latent(n=700, noise=0.30, irr=irr, seed=seed, family="phase4")
            raw = StandardScaler().fit_transform(X)
            methods = {
                "Raw-KMeans": KMeans(K, n_init=10, random_state=seed).fit_predict(raw),
                "Context-Relational": KMeans(
                    K, n_init=12, random_state=seed
                ).fit_predict(StandardScaler().fit_transform(relational(pre, post, ctx))),
                "QAF-Coherence": KMeans(K, n_init=15, random_state=seed).fit_predict(qaf(pre, post, ctx)),
            }
            for method, p in methods.items():
                rows.append(dict(irrelevant_dims=irr, rep=rep, method=method, ari=adjusted_rand_score(y, p)))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "irrelevant_stress.csv", index=False)
    return df


def run_scaling():
    rows = []
    for n in [1000, 2500, 5000, 10000]:
        for rep in range(2):
            seed = SEED_BASE + 900000 + n + rep
            X, y, ctx, pre, post = gen_latent(n=n, noise=0.25, seed=seed, family="phase4")

            t = time.perf_counter()
            raw = StandardScaler().fit_transform(X)
            KMeans(K, n_init=8, random_state=seed).fit_predict(raw)
            rows.append(dict(n_samples=n, rep=rep, method="Raw-KMeans", runtime_ms=(time.perf_counter() - t) * 1000))

            t = time.perf_counter()
            rr = StandardScaler().fit_transform(relational(pre, post, ctx))
            KMeans(K, n_init=8, random_state=seed).fit_predict(rr)
            rows.append(dict(n_samples=n, rep=rep, method="Context-Relational", runtime_ms=(time.perf_counter() - t) * 1000))

            t = time.perf_counter()
            qq = qaf(pre, post, ctx)
            KMeans(K, n_init=8, random_state=seed).fit_predict(qq)
            rows.append(dict(n_samples=n, rep=rep, method="QAF-Coherence", runtime_ms=(time.perf_counter() - t) * 1000))

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "scaling_runtime.csv", index=False)
    return df


def stats(cluster, anom):
    hidden = cluster[cluster.family.isin(HIDDEN)].copy()
    hsum = hidden.groupby("method")[["ari", "nmi", "matched_accuracy", "runtime_ms"]].mean().sort_values("ari", ascending=False)
    hsum.to_csv(RESULTS / "hidden_pattern_summary.csv")

    csum = cluster[cluster.family.eq(CONTROL)].groupby("method")[["ari", "nmi", "matched_accuracy", "runtime_ms"]].mean().sort_values("ari", ascending=False)
    csum.to_csv(RESULTS / "visible_control_summary.csv")

    asum = anom.groupby("method")[["roc_auc", "average_precision", "runtime_ms"]].mean().sort_values("roc_auc", ascending=False)
    asum.to_csv(RESULTS / "anomaly_summary.csv")

    p = hidden.pivot_table(index=["family", "noise", "rep"], columns="method", values="ari")
    d = p["QAF-Coherence"] - p["Context-Relational"]
    w = wilcoxon(p["QAF-Coherence"], p["Context-Relational"], alternative="greater")
    se = d.std(ddof=1) / np.sqrt(len(d))

    ap = anom.pivot_table(index=["noise", "rep"], columns="method", values="roc_auc")
    da = ap["QAF-Coherence-GMM"] - ap["IsolationForest-Relational"]
    wa = wilcoxon(ap["QAF-Coherence-GMM"], ap["IsolationForest-Relational"], alternative="greater")

    result = pd.DataFrame(
        [
            {
                "hidden_pairs": len(d),
                "qaf_mean_ari": p["QAF-Coherence"].mean(),
                "relational_mean_ari": p["Context-Relational"].mean(),
                "absolute_ari_advantage": d.mean(),
                "relative_ari_improvement_pct": (p["QAF-Coherence"].mean() - p["Context-Relational"].mean())
                / max(abs(p["Context-Relational"].mean()), 1e-12)
                * 100,
                "qaf_hidden_win_rate_pct": (d > 0).mean() * 100,
                "ari_ci95_low": d.mean() - 1.96 * se,
                "ari_ci95_high": d.mean() + 1.96 * se,
                "ari_wilcoxon_p": float(w.pvalue),
                "qaf_anomaly_auc": ap["QAF-Coherence-GMM"].mean(),
                "relational_anomaly_auc": ap["IsolationForest-Relational"].mean(),
                "anomaly_auc_advantage": da.mean(),
                "qaf_anomaly_win_rate_pct": (da > 0).mean() * 100,
                "anomaly_wilcoxon_p": float(wa.pvalue),
            }
        ]
    )
    result.to_csv(RESULTS / "primary_statistics.csv", index=False)
    return hsum, csum, asum, result


def figures(cluster, anom, irr, scaling):
    X, y, ctx, pre, post = gen_latent(n=1000, noise=0.25, seed=SEED_BASE + 123, family="phase4")
    raw = StandardScaler().fit_transform(X)
    qq = qaf(pre, post, ctx)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axis("off")
    boxes = [
        (0.10, 0.70, "Pre/pos estado\n+ contexto"),
        (0.33, 0.70, "Selección del\nsubespacio relevante"),
        (0.56, 0.70, "Coherencia QAF\nRe: alineación\nIm: orientación"),
        (0.80, 0.70, "Clustering/densidad\nsin etiquetas"),
        (0.33, 0.28, "Regimenes latentes"),
        (0.70, 0.28, "Anomalias de coherencia"),
    ]
    for x0, y0, l in boxes:
        ax.text(x0, y0, l, ha="center", va="center", transform=ax.transAxes, bbox=dict(boxstyle="round, pad=.5", fill=False))
    for p1, p2 in [
        ((0.16, 0.70), (0.26, 0.70)),
        ((0.40, 0.70), (0.49, 0.70)),
        ((0.63, 0.70), (0.73, 0.70)),
        ((0.56, 0.62), (0.37, 0.36)),
        ((0.68, 0.62), (0.69, 0.36)),
    ]:
        ax.annotate("", xy=p2, xytext=p1, xycoords=ax.transAxes, arrowprops=dict(arrowstyle="->"))
    ax.set_title("Figura 1. Arquitectura experimental de patrones ocultos")
    savefig(fig, "fig01_architecture.png")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(raw[:, 0], raw[:, 1], c=y, s=11)
    ax.set_xlabel("Observable 1")
    ax.set_ylabel("Observable 2")
    ax.set_title("Figura 2. Espacio observable: regímenes superpuestos")
    savefig(fig, "fig02_raw_overlap.png")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(qq[:, 0], qq[:, 1], c=y, s=11)
    ax.set_xlabel("Coherencia real")
    ax.set_ylabel("Coherencia antisimétrica")
    ax.set_title("Figura 3. Espacio QAF: regimenes ocultos separados")
    savefig(fig, "fig03_qaf_space.png")

    hidden = cluster[cluster.family.isin(HIDDEN)]
    g = hidden.groupby("method").ari.mean().sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(g.index, g.values)
    ax.set_xlabel("ARI")
    ax.set_title("Figura 4. Recuperación media de patrones ocultos")
    savefig(fig, "fig04_hidden_ari.png")

    g = hidden.groupby("method").nmi.mean().sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(g.index, g.values)
    ax.set_xlabel("NMI")
    ax.set_title("Figura 5. Información recuperada del régimen latente")
    savefig(fig, "fig05_hidden_nmi.png")

    focus = ["Raw-KMeans", "Context-Relational", "QAF-NoPhase", "QAF-Coherence"]
    g = hidden[hidden.method.isin(focus)].groupby(["noise", "method"]).ari.mean().reset_index()
    fig, ax = plt.subplots(figsize=(9, 6))
    for m in focus:
        s = g[g.method.eq(m)]
        ax.plot(s.noise, s.ari, marker="o", label=m)
    ax.set_xlabel("Ruido")
    ax.set_ylabel("ARI")
    ax.set_title("Figura 6. Robustez frente a ruido")
    ax.legend()
    savefig(fig, "fig06_noise.png")

    g = irr.groupby(["irrelevant_dims", "method"]).ari.mean().reset_index()
    fig, ax = plt.subplots(figsize=(9, 6))
    for m in ["Raw-KMeans", "Context-Relational", "QAF-Coherence"]:
        s = g[g.method.eq(m)]
        ax.plot(s.irrelevant_dims, s.ari, marker="o", label=m)
    ax.set_xlabel("Dimensiones irrelevantes")
    ax.set_ylabel("ARI")
    ax.set_title("Figura 7. Robustez a variables irrelevantes")
    ax.legend()
    savefig(fig, "fig07_irrelevant.png")

    g = anom.groupby(["noise", "method"]).roc_auc.mean().reset_index()
    fig, ax = plt.subplots(figsize=(9, 6))
    for m in ["IsolationForest-Raw", "OneClassSVM-Raw", "IsolationForest-Relational", "QAF-Coherence-GMM"]:
        s = g[g.method.eq(m)]
        ax.plot(s.noise, s.roc_auc, marker="o", label=m)
    ax.set_xlabel("Ruido")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Figura 8. Detección de anomalias de coherencia")
    ax.legend()
    savefig(fig, "fig08_anomaly_auc.png")

    g = scaling.groupby(["n_samples", "method"]).runtime_ms.mean().reset_index()
    fig, ax = plt.subplots(figsize=(9, 6))
    for m in ["Raw-KMeans", "Context-Relational", "QAF-Coherence"]:
        s = g[g.method.eq(m)]
        ax.plot(s.n_samples, s.runtime_ms, marker="o", label=m)
    ax.set_xlabel("Muestras")
    ax.set_ylabel("Tiempo (ms)")
    ax.set_title("Figura 9. Escalabilidad temporal")
    ax.legend()
    savefig(fig, "fig09_scaling.png")

    p = hidden.pivot_table(index=["family", "noise", "rep"], columns="method", values="ari").reset_index()
    p["adv"] = p["QAF-Coherence"] - p["Context-Relational"]
    hm = p.groupby(["family", "noise"]).adv.mean().unstack("noise")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(hm.values, aspect="auto")
    ax.set_xticks(range(len(hm.columns)))
    ax.set_xticklabels([f"{v:.2f}" for v in hm.columns])
    ax.set_yticks(range(len(hm.index)))
    ax.set_yticklabels(hm.index)
    ax.set_xlabel("Ruido")
    ax.set_ylabel("Familia")
    ax.set_title("Figura 10. Ventaja QAF sobre baseline relacional (ΔARI)")
    for i in range(hm.shape[0]):
        for j in range(hm.shape[1]):
            ax.text(j, i, f"{hm.iloc[i, j]:.2f}", ha="center", va="center")
    fig.colorbar(im, ax=ax, label="ΔARI")
    savefig(fig, "fig10_heatmap.png")


def main():
    t = time.perf_counter()
    cluster = run_clustering()
    anom = run_anomaly()
    irr = run_irrelevant()
    scaling = run_scaling()
    h, c, a, s = stats(cluster, anom)
    figures(cluster, anom, irr, scaling)
    print("Benchmark completed")
    print(f"Clustering executions: {len(cluster)}")
    print(f"Anomaly executions: {len(anom)}")
    print(f"Wall time: {time.perf_counter() - t:.2f} s")
    print("\nHidden pattern summary:")
    print(h.round(4).to_string())
    print("\nVisible control:")
    print(c.round(4).to_string())
    print("\nHidden anomaly summary:")
    print(a.round(4).to_string())
    print("\nPrimary statistics:")
    print(s.T.to_string(header=False))
    print("\nEnvironment:", platform.platform(), "Python", sys.version.split()[0])


if __name__ == "__main__":
    main()
