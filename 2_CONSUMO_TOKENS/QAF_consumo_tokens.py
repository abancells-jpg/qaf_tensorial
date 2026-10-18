#!/usr/bin/env python3
"""
QAF token-consumption benchmark (synthetic, architecture-level validation).

Scientific scope:
This benchmark does NOT call a commercial or frontier LLM. It tests a narrower
hypothesis: whether a QAF-inspired metacognitive routing layer can reduce the
number of language-model tokens that would be sent/returned by a multi-agent
system while preserving routing completeness on controlled synthetic tasks.

Token counts are REAL subword-token counts under a SentencePiece BPE tokenizer
trained reproducibly on the benchmark corpus. They are NOT provider-billable
OpenAI/Anthropic/Gemini token counts. Relative comparisons are valid within this
benchmark because every architecture uses the same tokenizer and same task corpus.

Architectures:
1) AllAgents: invokes every agent with the full context (upper-cost baseline).
2) RulesRouter: simple threshold routing, full context for selected agents.
3) QAFRouter: metacognitive/contextual routing + targeted context compression.
4) Oracle: knows the latent required-agent set; included only as a lower-bound
   reference, not as a deployable comparator.

Outputs:
- raw_results.csv
- summary.csv
- threshold_sweep.csv
- agent_activation.csv
- 10 figures
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sentencepiece as spm

AGENTS = ["memory", "retrieval", "specialist", "verification", "planner", "reporter"]

SYSTEM_INSTRUCTIONS = {
    "memory": "Recover only prior facts materially relevant to the current decision. Distinguish remembered evidence from current observations and return a compact evidence list.",
    "retrieval": "Inspect available external or indexed evidence for missing facts. Return only decision-relevant evidence, source confidence, and unresolved gaps.",
    "specialist": "Apply domain-specific analysis to the evidence supplied. State assumptions, critical constraints, and a concise technical recommendation.",
    "verification": "Search for contradictions, low-confidence claims, temporal inconsistencies, and evidence that should block an automatic decision. Return a verification status.",
    "planner": "Integrate evidence into a sequence of actions. Prefer reversible steps under uncertainty and explicitly identify escalation conditions.",
    "reporter": "Produce a concise decision-support summary with confidence, unresolved uncertainty, rationale, and whether human review is required."
}

@dataclass
class Task:
    tid: int
    level: str
    complexity: float
    uncertainty: float
    contradiction: float
    risk: float
    history_relevance: float
    specialization_need: float
    info_gap: float
    seed: int
    query: str
    records: List[Tuple[str, bool]] # text, relevant
    required: Set[str]
    observed: Dict[str, float]


def clip(x):
    return float(np.clip(x, 0.0, 1.0))


def required_agents(t: Dict[str, float]) -> Set[str]:
    r = {"reporter"}
    if t["history_relevance"] > 0.52:
        r.add("memory")
    if t["uncertainty"] > 0.52 or t["info_gap"] > 0.50:
        r.add("retrieval")
    if t["specialization_need"] > 0.54:
        r.add("specialist")
    if t["contradiction"] > 0.42 or t["risk"] > 0.72:
        r.add("verification")
    if t["complexity"] > 0.38 or t["risk"] > 0.50:
        r.add("planner")
    return r


def generate_task(tid: int, level: str, seed: int) -> Task:
    rng = np.random.default_rng(seed)
    centers = {"low": 0.22, "medium": 0.50, "high": 0.78}
    c = centers[level]

    complexity = clip(rng.normal(c, 0.13))
    uncertainty = clip(rng.normal(0.16 + 0.72 * complexity, 0.16))
    contradiction = clip(rng.beta(1.7 + 3.0 * complexity, 3.2))
    risk = clip(rng.normal(0.20 + 0.65 * complexity, 0.17))
    history_relevance = clip(rng.beta(1.4 + 2.6 * complexity, 2.5))
    specialization_need = clip(rng.normal(0.18 + 0.70 * complexity, 0.18))
    info_gap = clip(rng.normal(0.10 + 0.72 * uncertainty, 0.16))

    latent = dict(
        complexity=complexity, uncertainty=uncertainty, contradiction=contradiction,
        risk=risk, history_relevance=history_relevance,
        specialization_need=specialization_need, info_gap=info_gap
    )
    required = required_agents(latent)

    # Architecture only observes noisy estimates, as in a real pre-routing stage.
    observed = {k: clip(v + rng.normal(0, 0.075 + 0.035 * complexity)) for k, v in latent.items()}

    query = (
        f"Decision case {tid}. Multiple sources describe a changing operational situation. "
        f"The request is to determine whether available evidence is sufficient, which checks are necessary, "
        f"and what decision-support route should be followed. Report uncertainty explicitly and avoid unsupported certainty. "
        f"Case complexity marker {level}; evidence may contain omissions, stale observations, and conflicting interpretations."
    )

    n_records = int(round(5 + 18 * complexity + rng.integers(0, 5)))
    rel_rate = 0.30 + 0.28 * (1 - uncertainty)
    records = []

    vocab = [
        "sensor observation", "historical precedent", "operator note", "system status",
        "identity confidence", "time stamp", "resource state", "context update",
        "cross-source comparison", "previous correction", "technical constraint",
        "environmental condition", "uncertainty marker", "consistency check",
        "specialist note", "audit record"
    ]

    for j in range(n_records):
        relevant = bool(rng.random() < rel_rate)
        base = rng.choice(vocab)
        conf = rng.choice(["low", "medium", "high"])
        freshness = rng.choice(["current", "recent", "stale"])
        contradiction_phrase = "conflicts with another source" if (rng.random() < contradiction * 0.35) else "is internally consistent"
        filler_n = int(12 + 35 * complexity + rng.integers(0, 18))
        filler = " ".join(rng.choice(vocab, filler_n, replace=True))
        marker = "DECISION-RELEVANT" if relevant else "BACKGROUND"

        text = (
            f"Record {j+1} [{marker}]: {base}; confidence={conf}; temporal_status={freshness}; "
            f"this observation {contradiction_phrase}. {filler}."
        )
        records.append((text, relevant))

    return Task(tid, level, complexity, uncertainty, contradiction, risk, history_relevance,
                specialization_need, info_gap, seed, query, records, required, observed)


def train_tokenizer(tasks: List[Task], out_dir: Path, seed: int = 7) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = out_dir / "qaf_bpe.model"
    if existing.exists():
        return existing

    corpus = out_dir / "corpus.txt"
    with corpus.open("w", encoding="utf-8") as f:
        for t in tasks:
            f.write(t.query + "\n")
            for rec, _ in t.records:
                f.write(rec + "\n")
        for x in SYSTEM_INSTRUCTIONS.values():
            f.write(x + "\n")
        # Include output vocabulary in training corpus.
        for a in AGENTS:
            f.write(f"Agent {a} completed structured analysis. Evidence assessed confidence uncertainty unresolved gaps recommendation escalation status.\n")

    model_prefix = str(out_dir / "qaf_bpe")
    spm.SentencePieceTrainer.train(
        input=str(corpus), model_prefix=model_prefix, vocab_size=1200,
        model_type="bpe", character_coverage=1.0,
        bos_id=-1, eos_id=-1, pad_id=-1, unk_id=0,
        shuffle_input_sentence=False, num_threads=1,
        train_extremely_large_corpus=False
    )
    return out_dir / "qaf_bpe.model"


_TOKEN_CACHE = {}

def count_tokens(sp: spm.SentencePieceProcessor, text: str) -> int:
    v = _TOKEN_CACHE.get(text)
    if v is None:
        v = len(sp.encode(text, out_type=int))
        _TOKEN_CACHE[text] = v
    return v


def all_context(t: Task) -> str:
    return "\n".join(x for x, _ in t.records)


def targeted_context(t: Task, agent: str) -> str:
    rel = [x for x, r in t.records if r]
    bg = [x for x, r in t.records if not r]

    # Agent-specific budget: preserve all relevant evidence only for verification/retrieval;
    # other agents receive a smaller relevant slice and at most one background record.
    if agent in {"verification", "retrieval"}:
        chosen = rel[:max(1, min(len(rel), 10))]
    elif agent == "memory":
        chosen = rel[:max(1, min(len(rel), 6))]
    elif agent == "specialist":
        chosen = rel[:max(1, min(len(rel), 7))]
    elif agent == "planner":
        chosen = rel[:max(1, min(len(rel), 5))]
    else:
        chosen = rel[:max(1, min(len(rel), 4))]

    if not chosen and bg:
        chosen = bg[:1]

    return "\n".join(chosen)


def qaf_state_summary(t: Task) -> str:
    o = t.observed
    return (
        f"QAF metacognitive state: "
        f"uncertainty={o['uncertainty']:.2f}; contradiction={o['contradiction']:.2f}; risk={o['risk']:.2f}; "
        f"history_relevance={o['history_relevance']:.2f}; specialization_need={o['specialization_need']:.2f}; "
        f"information_gap={o['info_gap']:.2f}; complexity={o['complexity']:.2f}."
    )


def route_rules(t: Task) -> Set[str]:
    o = t.observed
    r = {"reporter"}
    if o["history_relevance"] > 0.62: r.add("memory")
    if o["uncertainty"] > 0.62: r.add("retrieval")
    if o["specialization_need"] > 0.63: r.add("specialist")
    if o["contradiction"] > 0.56: r.add("verification")
    if o["complexity"] > 0.55: r.add("planner")
    return r


def route_qaf(t: Task, threshold: float = 0.50) -> Set[str]:
    """QAF-inspired metacognitive router.
    Scores use contextual cross-terms so uncertainty+information gap, contradiction+risk,
    and complexity+risk can activate a specialist route even when no single feature crosses
    a simple threshold.
    """
    o = t.observed
    scores = {
        "memory": 0.82 * o["history_relevance"] + 0.18 * o["complexity"],
        "retrieval": 0.55 * o["uncertainty"] + 0.38 * o["info_gap"] + 0.17 * o["uncertainty"] * o["risk"],
        "specialist": 0.77 * o["specialization_need"] + 0.17 * o["complexity"] + 0.12 * o["risk"],
        "verification": 0.58 * o["contradiction"] + 0.32 * o["risk"] + 0.24 * o["contradiction"] * o["uncertainty"],
        "planner": 0.51 * o["complexity"] + 0.34 * o["risk"] + 0.18 * o["uncertainty"] * o["complexity"],
    }
    r = {"reporter"}
    for a, s in scores.items():
        if s > threshold:
            r.add(a)

    # Safety gate: high observed risk always gets verification+planning.
    if o["risk"] >= 0.82:
        r.update({"verification", "planner"})
    return r


def output_text(agent: str, t: Task, compact: bool) -> str:
    """Deterministic structured response emulator; tokenized like any other text.
    This is intentionally not an LLM. It represents the message volume an agent would
    contribute under the architecture. Compact QAF messages encode only sufficient state.
    """
    base = {
        "memory": "Prior evidence was checked against the current case. Relevant historical facts are separated from stale or unrelated records.",
        "retrieval": "Available evidence was inspected for missing facts. The response lists unresolved gaps and indicates whether additional evidence is necessary.",
        "specialist": "Technical constraints were evaluated. The analysis states the principal assumption, the most important limitation, and the recommended interpretation.",
        "verification": "Cross-source consistency was tested. Contradictions, low-confidence claims, and conditions requiring human review are explicitly identified.",
        "planner": "A reversible action sequence was prepared. The plan prioritizes information gain, preserves options under uncertainty, and defines escalation conditions.",
        "reporter": "Decision-support summary: evidence, uncertainty, confidence, unresolved issues, rationale, and escalation status are consolidated for the operator.",
    }[agent]

    repetitions = max(1, int(round((1.0 + 2.2 * t.complexity) * (0.62 if compact else 1.0))))
    details = []
    for j in range(repetitions):
        details.append(
            f"Assessment segment {j+1} uncertainty {t.observed['uncertainty']:.2f}, "
            f"risk {t.observed['risk']:.2f}, contradiction {t.observed['contradiction']:.2f}; "
            "evidence is prioritized by relevance and confidence."
        )
    return base + " " + " ".join(details)


def evaluate_task(t: Task, arch: str, sp: spm.SentencePieceProcessor, threshold: float = 0.50) -> Dict:
    if arch == "AllAgents":
        selected = set(AGENTS)
    elif arch == "RulesRouter":
        selected = route_rules(t)
    elif arch == "QAFRouter":
        selected = route_qaf(t, threshold)
    elif arch == "Oracle":
        selected = set(t.required)
    else:
        raise ValueError(arch)

    # Component-wise BPE accounting. Counting components separately avoids repeatedly
    # tokenizing giant concatenated prompts; every architecture uses the same convention.
    scaffold_tokens = count_tokens(sp, "SYSTEM USER CONTEXT Return a structured, decision-relevant response only.")
    query_tokens = count_tokens(sp, t.query)
    all_record_tokens = [count_tokens(sp, x) for x, _ in t.records]
    relevant_idx = [i for i, (_, r) in enumerate(t.records) if r]
    bg_idx = [i for i, (_, r) in enumerate(t.records) if not r]

    def context_token_count(agent):
        if arch not in ("QAFRouter", "Oracle"):
            return sum(all_record_tokens)

        if agent in {"verification", "retrieval"}: cap = 10
        elif agent == "memory": cap = 6
        elif agent == "specialist": cap = 7
        elif agent == "planner": cap = 5
        else: cap = 4

        ids = relevant_idx[:max(1, min(len(relevant_idx), cap))]
        if not ids and bg_idx:
            ids = bg_idx[:1]
        return sum(all_record_tokens[i] for i in ids)

    input_tokens = 0
    output_tokens = 0
    state_tokens = count_tokens(sp, qaf_state_summary(t)) if arch == "QAFRouter" else 0

    for agent in sorted(selected):
        input_tokens += scaffold_tokens + count_tokens(sp, SYSTEM_INSTRUCTIONS[agent]) + query_tokens + context_token_count(agent) + state_tokens
        output_tokens += count_tokens(sp, output_text(agent, t, compact=(arch in ("QAFRouter", "Oracle"))))

    tp = len(selected & t.required)
    fp = len(selected - t.required)
    fn = len(t.required - selected)

    recall = tp / len(t.required) if t.required else 1.0
    precision = tp / len(selected) if selected else 1.0
    success = 1 if fn == 0 else 0
    quality = max(0.0, min(1.0, 0.82 * recall + 0.18 * precision))
    total = input_tokens + output_tokens

    return {
        "task_id": t.tid, "level": t.level, "architecture": arch,
        "complexity": t.complexity, "uncertainty": t.uncertainty,
        "contradiction": t.contradiction, "risk": t.risk,
        "required_agents": len(t.required), "selected_agents": len(selected),
        "input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total,
        "routing_recall": recall, "routing_precision": precision,
        "success": success, "quality": quality,
        "tokens_per_quality": total / max(quality, 1e-6),
        "selected_set": ",".join(sorted(selected)),
        "required_set": ",".join(sorted(t.required)),
    }


def make_figures(df: pd.DataFrame, sweep: pd.DataFrame, activation: pd.DataFrame, fig_dir: Path):
    fig_dir.mkdir(parents=True, exist_ok=True)
    order = ["AllAgents", "RulesRouter", "QAFRouter", "Oracle"]
    pretty = {
        "AllAgents": "All agents",
        "RulesRouter": "Rules router",
        "QAFRouter": "QAF router",
        "Oracle": "Oracle lower bound"
    }

    # 1 architecture diagram
    fig, ax = plt.subplots(figsize=(10, 5.4)); ax.axis('off')
    ax.text(0.08, 0.75, "Conventional", fontsize=14, weight='bold')
    ax.text(0.08, 0.25, "QAF-inspired", fontsize=14, weight='bold')

    for y in [0.72, 0.22]:
        ax.annotate("", xy=(0.94, y), xytext=(0.16, y), arrowprops=dict(arrowstyle="->", lw=1.5))

    xs = [0.18, 0.38, 0.60, 0.80]
    labs_top = ["Query + full context", "All agents", "All messages", "Final synthesis"]
    labs_bot = ["Query + state", "Metacognitive routing", "Targeted agents/context", "Final synthesis"]

    for x, l1, l2 in zip(xs, labs_top, labs_bot):
        ax.text(x, 0.72, l1, ha='center', va='center', bbox=dict(boxstyle='round, pad=.35', fc='white'))
        ax.text(x, 0.22, l2, ha='center', va='center', bbox=dict(boxstyle='round, pad=.35', fc='white'))

    ax.set_title("Figure 1. Token-flow architectures compared", fontsize=16)
    fig.tight_layout(); fig.savefig(fig_dir / 'fig01_architecture.png', dpi=180); plt.close(fig)

    # summary helpers
    summ = df.groupby('architecture').agg(
        total_tokens=('total_tokens', 'mean'), input_tokens=('input_tokens', 'mean'),
        output_tokens=('output_tokens', 'mean'), calls=('selected_agents', 'mean'),
        success=('success', 'mean'), quality=('quality', 'mean')
    ).reindex(order)

    # 2 total tokens
    fig, ax = plt.subplots(figsize=(8, 5)); vals = summ.total_tokens.values
    ax.bar([pretty[x] for x in order], vals); ax.set_ylabel('Mean tokens per task'); ax.set_title('Figure 2. Total token consumption'); ax.tick_params(axis='x', rotation=18)
    for i, v in enumerate(vals): ax.text(i, v, f'{v:,.0f}', ha='center', va='bottom')
    fig.tight_layout(); fig.savefig(fig_dir / 'fig02_total_tokens.png', dpi=180); plt.close(fig)

    # 3 input/output stacked
    fig, ax = plt.subplots(figsize=(8, 5)); x = np.arange(len(order)); inp = summ.input_tokens.values; out = summ.output_tokens.values
    ax.bar(x, inp, label='Input'); ax.bar(x, out, bottom=inp, label='Output'); ax.set_xticks(x, [pretty[z] for z in order], rotation=18); ax.set_ylabel('Mean tokens per task'); ax.set_title('Figure 3. Input vs output token footprint'); ax.legend()
    fig.tight_layout(); fig.savefig(fig_dir / 'fig03_input_output.png', dpi=180); plt.close(fig)

    # 4 reduction by complexity vs all agents
    lvl_order = ['low', 'medium', 'high']
    p = df.groupby(['level', 'architecture']).total_tokens.mean().unstack()
    reduction = (1 - p.div(p['AllAgents'], axis=0)) * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    for a in ['RulesRouter', 'QAFRouter', 'Oracle']:
        ax.plot(lvl_order, reduction.loc[lvl_order, a], marker='o', label=pretty[a])
    ax.set_ylabel('Token reduction vs AllAgents (%)'); ax.set_xlabel('Task complexity'); ax.set_title('Figure 4. Token savings as complexity rises'); ax.legend(); ax.set_ylim(bottom=0)
    fig.tight_layout(); fig.savefig(fig_dir / 'fig04_reduction_complexity.png', dpi=180); plt.close(fig)

    # 5 calls per task
    fig, ax = plt.subplots(figsize=(8, 5)); vals = summ.calls.values
    ax.bar([pretty[x] for x in order], vals); ax.set_ylabel('Mean agent calls'); ax.set_title('Figure 5. Communication calls per task'); ax.tick_params(axis='x', rotation=18)
    for i, v in enumerate(vals): ax.text(i, v, f'{v:.2f}', ha='center', va='bottom')
    fig.tight_layout(); fig.savefig(fig_dir / 'fig05_calls.png', dpi=180); plt.close(fig)

    # 6 Pareto cost-quality
    fig, ax = plt.subplots(figsize=(7, 5))
    for a in order:
        ax.scatter(summ.loc[a, 'total_tokens'], summ.loc[a, 'quality'] * 100, s=90, label=pretty[a])
        ax.annotate(pretty[a], (summ.loc[a, 'total_tokens'], summ.loc[a, 'quality'] * 100), xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax.set_xlabel('Mean tokens per task'); ax.set_ylabel('Mean routing quality (%)'); ax.set_title('Figure 6. Token-quality frontier'); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(fig_dir / 'fig06_pareto.png', dpi=180); plt.close(fig)

    # 7 tokens per successful task
    tmp = df.copy(); tmp['success_tokens'] = np.where(tmp.success == 1, tmp.total_tokens, np.nan)
    successful_mean = tmp.groupby('architecture').success_tokens.mean().reindex(order)
    fig, ax = plt.subplots(figsize=(8, 5)); vals = successful_mean.values
    ax.bar([pretty[x] for x in order], vals); ax.set_ylabel('Tokens, successful tasks only'); ax.set_title('Figure 7. Token cost of a successful route'); ax.tick_params(axis='x', rotation=18)
    fig.tight_layout(); fig.savefig(fig_dir / 'fig07_tokens_success.png', dpi=180); plt.close(fig)

    # 8 distribution p50 p95
    stats = df.groupby('architecture').total_tokens.quantile([.5, .95]).unstack().reindex(order)
    fig, ax = plt.subplots(figsize=(8, 5)); x = np.arange(len(order)); w = .36
    ax.bar(x - w / 2, stats[0.5], w, label='p50'); ax.bar(x + w / 2, stats[0.95], w, label='p95'); ax.set_xticks(x, [pretty[z] for z in order], rotation=18); ax.set_ylabel('Tokens per task'); ax.set_title('Figure 8. Median and tail token consumption'); ax.legend()
    fig.tight_layout(); fig.savefig(fig_dir / 'fig08_p50_p95.png', dpi=180); plt.close(fig)

    # 9 activation heatmap for QAF
    q = activation[activation.architecture == 'QAFRouter'].pivot(index='agent', columns='level', values='activation_rate').reindex(AGENTS)[lvl_order]
    fig, ax = plt.subplots(figsize=(7.5, 5.2)); im = ax.imshow(q.values, aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(3), lvl_order); ax.set_yticks(range(len(AGENTS)), AGENTS); ax.set_title('Figure 9. QAF agent activation by complexity')
    for i in range(q.shape[0]):
        for j in range(q.shape[1]): ax.text(j, i, f'{q.iloc[i, j] * 100:.0f}%', ha='center', va='center')
    fig.colorbar(im, ax=ax, label='Activation rate'); fig.tight_layout(); fig.savefig(fig_dir / 'fig09_activation_heatmap.png', dpi=180); plt.close(fig)

    # 10 threshold sensitivity
    fig, ax1 = plt.subplots(figsize=(8, 5)); ax2 = ax1.twinx()
    ax1.plot(sweep.threshold, sweep.mean_tokens, marker='o', label='Tokens')
    ax2.plot(sweep.threshold, sweep.success_rate * 100, marker='s', linestyle='--', label='Success')
    ax1.set_xlabel('QAF routing threshold'); ax1.set_ylabel('Mean tokens'); ax2.set_ylabel('Routing success (%)')
    ax1.set_title('Figure 10. QAF threshold: efficiency-quality trade-off')
    fig.tight_layout(); fig.savefig(fig_dir / 'fig10_threshold_sensitivity.png', dpi=180); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='./qaf_tokens_results')
    ap.add_argument('--n-per-level', type=int, default=250)
    ap.add_argument('--seed', type=int, default=20260919)
    args = ap.parse_args()

    out = Path(args.out)
    (out / 'results').mkdir(parents=True, exist_ok=True)
    (out / 'figures').mkdir(exist_ok=True)
    (out / 'tokenizer').mkdir(exist_ok=True)

    tasks = []
    tid = 0
    for li, level in enumerate(['low', 'medium', 'high']):
        for i in range(args.n_per_level):
            tid += 1
            tasks.append(generate_task(tid, level, args.seed + li * 100000 + i))

    model = train_tokenizer(tasks, out / 'tokenizer')
    sp = spm.SentencePieceProcessor(model_file=str(model))

    rows = []
    for t in tasks:
        for arch in ['AllAgents', 'RulesRouter', 'QAFRouter', 'Oracle']:
            rows.append(evaluate_task(t, arch, sp, threshold=.35))

    df = pd.DataFrame(rows)
    df.to_csv(out / 'results' / 'raw_results.csv', index=False)

    summary = df.groupby('architecture').agg(
        tasks=('task_id', 'count'),
        mean_input_tokens=('input_tokens', 'mean'),
        mean_output_tokens=('output_tokens', 'mean'),
        mean_total_tokens=('total_tokens', 'mean'),
        median_tokens=('total_tokens', 'median'),
        p95_tokens=('total_tokens', lambda x: np.quantile(x, .95)),
        mean_agent_calls=('selected_agents', 'mean'),
        routing_recall=('routing_recall', 'mean'),
        routing_precision=('routing_precision', 'mean'),
        success_rate=('success', 'mean'),
        mean_quality=('quality', 'mean')
    ).reset_index()

    base = float(summary.loc[summary.architecture == 'AllAgents', 'mean_total_tokens'].iloc[0])
    summary['token_reduction_vs_all_pct'] = (1 - summary.mean_total_tokens / base) * 100
    summary['tokens_per_success_equivalent'] = summary.mean_total_tokens / summary.success_rate.replace(0, np.nan)
    summary.to_csv(out / 'results' / 'summary.csv', index=False)

    # threshold sweep on a subset for efficiency
    subset = tasks[::max(1, len(tasks) // 900)][:900]
    sw = []
    for th in np.arange(.35, .76, .05):
        rr = [evaluate_task(t, 'QAFRouter', sp, threshold=float(th)) for t in subset]
        d = pd.DataFrame(rr)
        sw.append(dict(
            threshold=round(float(th), 2),
            mean_tokens=d.total_tokens.mean(),
            success_rate=d.success.mean(),
            quality=d.quality.mean(),
            mean_calls=d.selected_agents.mean()
        ))
    sweep = pd.DataFrame(sw)
    sweep.to_csv(out / 'results' / 'threshold_sweep.csv', index=False)

    # Agent activation rates by architecture/complexity.
    ar = []
    for arch in ['RulesRouter', 'QAFRouter']:
        sub = df[df.architecture == arch]
        for level in ['low', 'medium', 'high']:
            ss = sub[sub.level == level]
            for agent in AGENTS:
                ar.append(dict(
                    architecture=arch, level=level, agent=agent,
                    activation_rate=ss.selected_set.str.contains(fr'(^|,){agent}(,|$)', regex=True).mean()
                ))
    activation = pd.DataFrame(ar)
    activation.to_csv(out / 'results' / 'agent_activation.csv', index=False)

    make_figures(df, sweep, activation, out / 'figures')

    metadata = {
        'n_tasks': len(tasks),
        'n_architecture_evaluations': len(df),
        'seed': args.seed,
        'tokenizer': 'SentencePiece BPE 1200 vocabulary, benchmark-trained',
        'scientific_scope': 'architecture-level token accounting; no external LLM inference',
    }
    (out / 'results' / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')

    print(summary.to_string(index=False))
    print('\nQAF by level:')
    print(df[df.architecture == 'QAFRouter'].groupby('level').agg(
        tokens=('total_tokens', 'mean'),
        success=('success', 'mean'),
        quality=('quality', 'mean'),
        calls=('selected_agents', 'mean')
    ).to_string())


if __name__ == '__main__':
    main()
