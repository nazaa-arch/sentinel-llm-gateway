"""Run comprehensive PII, Guardrails, and Semantic Cache evaluation benchmarks."""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sentinel_gateway.cache import SemanticCache  # noqa: E402
from sentinel_gateway.guardrails import GuardrailsEngine  # noqa: E402
from sentinel_gateway.pii import PIIRedactor  # noqa: E402
from sentinel_gateway.providers import LocalDemoProvider  # noqa: E402
from sentinel_gateway.schemas import ChatMessage  # noqa: E402

DATASETS_DIR = Path(__file__).resolve().parent / "datasets"
REPORT_PATH = Path(__file__).resolve().parent / "latest_report.json"


def _safe_div(num: float, den: float) -> float:
    return round(num / den, 4) if den > 0 else 0.0


def evaluate_pii() -> dict[str, Any]:
    """Evaluate entity-level Precision, Recall, F1, and negative FPR on ID & EN PII."""
    cases = json.loads((DATASETS_DIR / "pii_benchmark.json").read_text(encoding="utf-8"))
    redactor = PIIRedactor()

    tp = 0
    fp = 0
    fn = 0
    neg_cases = 0
    neg_fp_cases = 0
    per_type: dict[str, dict[str, int]] = {}
    errors: list[dict[str, Any]] = []

    started = time.perf_counter()
    for case in cases:
        res = redactor.redact(case["text"])
        expected = {(e["type"], e["value"]) for e in case["expected_entities"]}
        predicted = {(e.entity_type, e.value) for e in res.entities}

        if not expected:
            neg_cases += 1
            if predicted:
                neg_fp_cases += 1

        matched = expected & predicted
        false_pos = predicted - expected
        false_neg = expected - predicted

        tp += len(matched)
        fp += len(false_pos)
        fn += len(false_neg)

        for etype, _ in matched:
            per_type.setdefault(etype, {"tp": 0, "fp": 0, "fn": 0})["tp"] += 1
        for etype, _ in false_pos:
            per_type.setdefault(etype, {"tp": 0, "fp": 0, "fn": 0})["fp"] += 1
        for etype, _ in false_neg:
            per_type.setdefault(etype, {"tp": 0, "fp": 0, "fn": 0})["fn"] += 1

        if false_pos or false_neg:
            errors.append(
                {
                    "id": case["id"],
                    "false_positives": sorted(false_pos),
                    "false_negatives": sorted(false_neg),
                }
            )

    total_ms = (time.perf_counter() - started) * 1000.0
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    neg_fpr = _safe_div(neg_fp_cases, neg_cases)

    return {
        "total_cases": len(cases),
        "negative_trap_cases": neg_cases,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "negative_case_fpr": neg_fpr,
        "avg_latency_ms": round(total_ms / max(1, len(cases)), 3),
        "per_entity_type": per_type,
        "errors": errors,
    }


def evaluate_injection_guardrails() -> dict[str, Any]:
    """Evaluate bilingual prompt injection, obfuscation, and toxicity detection."""
    cases = json.loads((DATASETS_DIR / "injection_benchmark.json").read_text(encoding="utf-8"))
    engine = GuardrailsEngine()

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    errors: list[dict[str, Any]] = []

    started = time.perf_counter()
    for case in cases:
        ev = engine.evaluate(case["text"], policy_mode="block")
        predicted_blocked = not ev.allowed
        expected_blocked = bool(case["expected_blocked"])

        if expected_blocked and predicted_blocked:
            tp += 1
        elif not expected_blocked and not predicted_blocked:
            tn += 1
        elif not expected_blocked and predicted_blocked:
            fp += 1
            errors.append(
                {
                    "id": case["id"],
                    "error": "false_positive",
                    "injection_score": ev.injection_score,
                    "toxicity_score": ev.toxicity_score,
                }
            )
        else:
            fn += 1
            errors.append(
                {
                    "id": case["id"],
                    "error": "false_negative",
                    "injection_score": ev.injection_score,
                    "toxicity_score": ev.toxicity_score,
                }
            )

    total_ms = (time.perf_counter() - started) * 1000.0
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    fpr = _safe_div(fp, fp + tn)
    accuracy = _safe_div(tp + tn, len(cases))

    return {
        "total_cases": len(cases),
        "adversarial_cases": tp + fn,
        "benign_cases": tn + fp,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
        "avg_latency_ms": round(total_ms / max(1, len(cases)), 3),
        "errors": errors,
    }


async def evaluate_semantic_cache(threshold: float = 0.86) -> dict[str, Any]:
    """Evaluate semantic cache paraphrase hit rate, distinct-query FPR, and latency speedup."""
    cases = json.loads((DATASETS_DIR / "cache_benchmark.json").read_text(encoding="utf-8"))
    provider = LocalDemoProvider(simulated_latency_ms=15.0)

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    miss_latencies_ms: list[float] = []
    hit_latencies_ms: list[float] = []
    paraphrase_sims: list[float] = []
    distinct_sims: list[float] = []
    errors: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = SemanticCache(
            db_path=Path(tmpdir) / "eval_cache.db",
            default_threshold=threshold,
            ttl_seconds=3600,
        )
        try:
            # Seed unique baseline prompts and measure uncached LLM generation latency
            seeded: set[str] = set()
            for case in cases:
                seed_prompt = case["seed_prompt"]
                if seed_prompt in seeded:
                    continue
                seeded.add(seed_prompt)
                t0 = time.perf_counter()
                res = await provider.complete(
                    [ChatMessage(role="user", content=seed_prompt)],
                    model="sentinel-demo-v1",
                )
                miss_latencies_ms.append((time.perf_counter() - t0) * 1000.0)
                cache.store(
                    prompt=seed_prompt,
                    response_text=res.content,
                    model="sentinel-demo-v1",
                    provider="local-demo",
                    prompt_tokens_est=res.prompt_tokens,
                    completion_tokens_est=res.completion_tokens,
                )

            # Query paraphrases and distinct prompts
            for case in cases:
                t0 = time.perf_counter()
                hit = cache.lookup(
                    prompt=case["query_prompt"],
                    model="sentinel-demo-v1",
                    threshold=threshold,
                )
                lookup_ms = (time.perf_counter() - t0) * 1000.0

                # Also compute direct pair cosine similarity for diagnostic reporting
                v_seed = cache.vectorizer.encode(case["seed_prompt"])
                v_query = cache.vectorizer.encode(case["query_prompt"])
                pair_sim = round(cache.vectorizer.cosine_similarity(v_seed, v_query), 4)

                predicted_hit = hit is not None
                expected_hit = bool(case["expected_hit"])

                if expected_hit:
                    paraphrase_sims.append(pair_sim)
                else:
                    distinct_sims.append(pair_sim)

                if predicted_hit:
                    hit_latencies_ms.append(lookup_ms)

                if expected_hit and predicted_hit:
                    tp += 1
                elif not expected_hit and not predicted_hit:
                    tn += 1
                elif not expected_hit and predicted_hit:
                    fp += 1
                    errors.append(
                        {"id": case["id"], "error": "false_positive_hit", "sim": hit.similarity}
                    )
                else:
                    fn += 1
                    errors.append(
                        {"id": case["id"], "error": "missed_paraphrase", "pair_sim": pair_sim}
                    )
        finally:
            cache.close()

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    fpr = _safe_div(fp, fp + tn)

    avg_miss_ms = round(sum(miss_latencies_ms) / max(1, len(miss_latencies_ms)), 3)
    avg_hit_ms = round(sum(hit_latencies_ms) / max(1, len(hit_latencies_ms)), 3)
    speedup = round(avg_miss_ms / max(0.01, avg_hit_ms), 2)

    return {
        "threshold": threshold,
        "total_pairs": len(cases),
        "paraphrase_pairs": tp + fn,
        "distinct_pairs": tn + fp,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "paraphrase_hit_rate": recall,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
        "avg_paraphrase_similarity": round(
            sum(paraphrase_sims) / max(1, len(paraphrase_sims)), 4
        ),
        "avg_distinct_similarity": round(sum(distinct_sims) / max(1, len(distinct_sims)), 4),
        "avg_uncached_latency_ms": avg_miss_ms,
        "avg_cache_hit_latency_ms": avg_hit_ms,
        "latency_speedup_x": speedup,
        "errors": errors,
    }


def format_markdown_summary(report: dict[str, Any]) -> str:
    """Render a clean Markdown benchmark table for CLI output and README documentation."""
    pii = report["pii_benchmark"]
    inj = report["injection_benchmark"]
    cache = report["cache_benchmark"]

    lines = [
        "# Sentinel LLM Gateway — Evaluation Benchmark Report",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "| Benchmark Suite | Cases | Precision | Recall | F1 | FPR | Key Performance Metric |",
        "|---|---:|---:|---:|---:|---:|---|",
        (
            f"| Indonesian & Global PII Redaction | {pii['total_cases']} | "
            f"{pii['precision']:.4f} | {pii['recall']:.4f} | {pii['f1']:.4f} | "
            f"{pii['negative_case_fpr']:.4f} | {pii['avg_latency_ms']} ms/req |"
        ),
        (
            f"| Bilingual Prompt Injection & Toxicity | {inj['total_cases']} | "
            f"{inj['precision']:.4f} | {inj['recall']:.4f} | {inj['f1']:.4f} | "
            f"{inj['false_positive_rate']:.4f} | {inj['avg_latency_ms']} ms/req |"
        ),
        (
            f"| Semantic Cache (threshold={cache['threshold']}) | {cache['total_pairs']} | "
            f"{cache['precision']:.4f} | {cache['recall']:.4f} | {cache['f1']:.4f} | "
            f"{cache['false_positive_rate']:.4f} | {cache['latency_speedup_x']}x speedup "
            f"({cache['avg_uncached_latency_ms']} ms -> {cache['avg_cache_hit_latency_ms']} ms) |"
        ),
    ]
    return "\n".join(lines)


def main() -> None:
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "pii_benchmark": evaluate_pii(),
        "injection_benchmark": evaluate_injection_guardrails(),
        "cache_benchmark": asyncio.run(evaluate_semantic_cache(threshold=0.86)),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(format_markdown_summary(report))
    print(f"\nSaved detailed JSON report to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
