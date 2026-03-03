"""
evaluate.py
-----------
Runs DeepEval evaluation over a RAG Q&A dataset.

Judge LLM:  Ollama (local) — default model: mistral
No API keys required. Everything runs locally.

Configuration:
  Edit config.yaml to adjust model, thresholds, output directory,
  data path, and max cases. Falls back to sensible defaults if
  config.yaml is absent.

Metrics evaluated:
  Standard:
    - faithfulness            Does the answer stick to the retrieved context?
    - answer_relevancy        Does the answer address the question?
    - contextual_precision    Are the retrieved chunks relevant?
    - contextual_recall       Did the chunks cover everything needed?
    - contextual_relevancy    Are the chunks relevant at all?
  Extended:
    - hallucination           Rate of invented facts not in context (lower=better)
    - supply_chain_specificity  G-Eval: actionable identifiers present?
    - answer_completeness     G-Eval: all context details covered?

Usage:
  python evaluate.py
"""

# Fix Windows console encoding before rich/deepeval initializes.
# Without this, DeepEval's rich output crashes on emoji (cp1252 can't encode ✨).
import sys
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import json
import logging
import os
import statistics
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    GEval,
    HallucinationMetric,
)
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from data_loader import load_dataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG: dict = {
    "model": "mistral",
    "thresholds": {
        "faithfulness":             0.65,
        "answer_relevancy":         0.60,
        "contextual_precision":     0.55,
        "contextual_recall":        0.55,
        "contextual_relevancy":     0.50,
        "hallucination":            0.0,
        "supply_chain_specificity": 0.55,
        "answer_completeness":      0.55,
        "answer_correctness":       0.60,
    },
    # Weights must sum to 1.0. Used to compute the 0-100 Health Score.
    "weights": {
        "faithfulness":             0.18,
        "answer_correctness":       0.18,
        "answer_relevancy":         0.13,
        "hallucination":            0.13,
        "contextual_precision":     0.09,
        "contextual_recall":        0.09,
        "contextual_relevancy":     0.07,
        "supply_chain_specificity": 0.07,
        "answer_completeness":      0.06,
    },
    "output_dir": "output",
    "max_cases":  None,
    "data_path":  None,
}

# Metrics where a LOWER score is better (e.g. hallucination rate)
_LOWER_IS_BETTER = {"hallucination"}


def _load_config(path: str = "config.yaml") -> dict:
    if not Path(path).exists():
        return _DEFAULT_CONFIG.copy()
    with open(path, "r") as f:
        user_cfg = yaml.safe_load(f) or {}
    cfg = _DEFAULT_CONFIG.copy()
    # Shallow-merge top-level keys (excluding thresholds/weights, handled below)
    cfg.update({k: v for k, v in user_cfg.items() if k not in ("thresholds", "weights")})
    # Deep-merge thresholds so the user only needs to override specific ones
    if "thresholds" in user_cfg and isinstance(user_cfg["thresholds"], dict):
        cfg["thresholds"] = {**_DEFAULT_CONFIG["thresholds"], **user_cfg["thresholds"]}
    # Deep-merge weights
    if "weights" in user_cfg and isinstance(user_cfg["weights"], dict):
        cfg["weights"] = {**_DEFAULT_CONFIG["weights"], **user_cfg["weights"]}
    return cfg


CONFIG = _load_config()
THRESHOLDS: dict = CONFIG["thresholds"]
WEIGHTS: dict = CONFIG["weights"]

METRIC_COLS = [
    "faithfulness",
    "answer_relevancy",
    "contextual_precision",
    "contextual_recall",
    "contextual_relevancy",
    "hallucination",
    "supply_chain_specificity",
    "answer_completeness",
    "answer_correctness",
]

METRIC_SHORT = {
    "faithfulness":             "Faith",
    "answer_relevancy":         "AnsRel",
    "contextual_precision":     "CtxPre",
    "contextual_recall":        "CtxRec",
    "contextual_relevancy":     "CtxRel",
    "hallucination":            "Halluc",
    "supply_chain_specificity": "SCSpec",
    "answer_completeness":      "Compl",
    "answer_correctness":       "AnsCorr",
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging(run_dir: Path) -> None:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    fmt_console = logging.Formatter("%(levelname)-8s %(message)s")
    fmt_file    = logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt_console)
    root.addHandler(console_handler)

    file_handler = logging.FileHandler(run_dir / "eval.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt_file)
    root.addHandler(file_handler)


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Core evaluation functions
# ---------------------------------------------------------------------------

_REQUIRED_KEYS = {"question", "answer", "contexts", "ground_truth"}


def build_test_cases(data: list[dict]) -> list[LLMTestCase]:
    """Build LLMTestCase objects, skipping any records with invalid schema."""
    cases = []
    for i, d in enumerate(data):
        missing = _REQUIRED_KEYS - set(d.keys())
        if missing:
            logger.warning("Record %d missing keys %s — skipped", i, missing)
            continue
        if not isinstance(d.get("contexts"), list):
            logger.warning("Record %d: 'contexts' must be a list — skipped", i)
            continue
        cases.append(
            LLMTestCase(
                input=d["question"],
                actual_output=d["answer"],
                retrieval_context=d["contexts"],   # used by retrieval metrics
                context=d["contexts"],             # used by HallucinationMetric
                expected_output=d["ground_truth"],
            )
        )
    return cases


def _build_metrics(llm) -> list:
    t = THRESHOLDS
    return [
        FaithfulnessMetric(
            threshold=t["faithfulness"],
            model=llm,
            include_reason=True,
        ),
        AnswerRelevancyMetric(
            threshold=t["answer_relevancy"],
            model=llm,
            include_reason=True,
        ),
        ContextualPrecisionMetric(
            threshold=t["contextual_precision"],
            model=llm,
            include_reason=True,
        ),
        ContextualRecallMetric(
            threshold=t["contextual_recall"],
            model=llm,
            include_reason=True,
        ),
        ContextualRelevancyMetric(
            threshold=t["contextual_relevancy"],
            model=llm,
            include_reason=True,
        ),
        HallucinationMetric(
            threshold=t["hallucination"],
            model=llm,
        ),
        GEval(
            name="Supply Chain Specificity",
            criteria=(
                "Does the answer include specific technical identifiers "
                "(model numbers, SKUs, specs, lead times) that make it "
                "actionable for a procurement decision?"
            ),
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ],
            threshold=t["supply_chain_specificity"],
            model=llm,
        ),
        GEval(
            name="Answer Completeness",
            criteria=(
                "Does the answer address all aspects of the question without "
                "omitting key details present in the retrieved context?"
            ),
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.RETRIEVAL_CONTEXT,
            ],
            threshold=t["answer_completeness"],
            model=llm,
        ),
        GEval(
            name="Answer Correctness",
            criteria=(
                "Does the actual output convey the same factual information "
                "as the expected output? Penalize any missing facts, "
                "contradictions, or incorrect details compared to the expected output."
            ),
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            threshold=t["answer_correctness"],
            model=llm,
        ),
    ]


def run_evaluation(test_cases: list[LLMTestCase], llm):
    """Run DeepEval with 3 attempts and exponential backoff on transient errors."""
    metrics = _build_metrics(llm)

    logger.info("Running DeepEval evaluation...")
    logger.info("  Judge LLM : Ollama (%s)", CONFIG["model"])
    logger.info("  Test cases: %d", len(test_cases))
    logger.info("  Metrics   : %d", len(metrics))

    last_exc = None
    for attempt in range(1, 4):
        try:
            return evaluate(test_cases, metrics, async_config=AsyncConfig(run_async=False))
        except Exception as exc:
            last_exc = exc
            if attempt == 3:
                break
            wait = 2 ** attempt          # 2s, 4s
            logger.warning(
                "Attempt %d/3 failed (%s: %s). Retrying in %ds...",
                attempt, type(exc).__name__, exc, wait,
            )
            time.sleep(wait)

    raise RuntimeError(f"Evaluation failed after 3 attempts. Last error: {last_exc}") from last_exc

# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------


def build_results_df(eval_results) -> pd.DataFrame:
    rows = []
    for result in eval_results.test_results:
        ctx = result.retrieval_context
        row = {
            "question":          result.input,
            "answer":            result.actual_output,
            "ground_truth":      result.expected_output,
            "retrieval_context": " | ".join(ctx) if isinstance(ctx, list) else (ctx or ""),
            "passed":            result.success,
        }
        for md in result.metrics_data:
            key = md.name.lower().replace(" ", "_")
            row[key] = md.score
            row[f"{key}_reason"] = md.reason
        rows.append(row)
    return pd.DataFrame(rows)


def build_failures_df(eval_results) -> pd.DataFrame:
    rows = []
    for result in eval_results.test_results:
        ctx = result.retrieval_context
        ctx_str = " | ".join(ctx) if isinstance(ctx, list) else (ctx or "")
        for md in result.metrics_data:
            if md.score is not None and md.success is False:
                key = md.name.lower().replace(" ", "_")
                rows.append({
                    "question":          result.input,
                    "answer":            result.actual_output,
                    "ground_truth":      result.expected_output,
                    "retrieval_context": ctx_str,
                    "metric":            md.name,
                    "score":             md.score,
                    "threshold":         THRESHOLDS.get(key, "?"),
                    "reason":            md.reason,
                })
    return pd.DataFrame(rows)


def compute_health_score(df: pd.DataFrame) -> float:
    """Compute a weighted 0-100 health score across all metrics.

    Each metric is normalised to [0, 1] (lower-is-better metrics are
    inverted), multiplied by its weight, then scaled to 100.
    Metrics missing from the DataFrame are skipped; weights are
    re-normalised so the score is always out of 100.
    """
    score = 0.0
    total_weight = 0.0
    for col, weight in WEIGHTS.items():
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        if vals.empty:
            continue
        avg = float(vals.mean())
        normalised = (1.0 - avg) if col in _LOWER_IS_BETTER else avg
        normalised = max(0.0, min(1.0, normalised))
        score += normalised * weight
        total_weight += weight
    if total_weight == 0.0:
        return 0.0
    return round((score / total_weight) * 100, 1)


def _grade(score: float) -> str:
    if score >= 90:
        return "A - Excellent"
    if score >= 80:
        return "B - Good"
    if score >= 70:
        return "C - Acceptable"
    if score >= 60:
        return "D - Needs Work"
    return "F - Critical"


def find_previous_run(current_run_dir: Path) -> dict | None:
    """Return the run_info dict from the most recent prior run, or None."""
    output_dir = current_run_dir.parent
    current_id = current_run_dir.name
    candidates = sorted(
        d for d in output_dir.iterdir()
        if d.is_dir() and d.name != current_id and (d / "run_info.json").exists()
    )
    if not candidates:
        return None
    with open(candidates[-1] / "run_info.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _metric_passes_aggregate(col: str, avg: float) -> bool:
    threshold = THRESHOLDS.get(col, 0.5)
    if col in _LOWER_IS_BETTER:
        return avg <= threshold
    return avg >= threshold


def print_summary(
    df: pd.DataFrame,
    eval_results,
    health_score: float,
    prev_info: dict | None,
) -> None:
    cols = [c for c in METRIC_COLS if c in df.columns]

    # --- Health Score banner (manager-facing TL;DR) ---
    grade = _grade(health_score)
    logger.info("=" * 80)
    logger.info("RAG HEALTH SCORE")
    logger.info("=" * 80)
    logger.info("  Score : %.1f / 100  [%s]", health_score, grade)
    if prev_info and "health_score" in prev_info:
        prev_score = prev_info["health_score"]
        delta = health_score - prev_score
        sign = "+" if delta >= 0 else ""
        if delta > 1:
            verdict = "IMPROVED"
        elif delta < -1:
            verdict = "REGRESSED"
        else:
            verdict = "STABLE"
        logger.info(
            "  vs    : run %s  score=%.1f  (%s%.1f)  %s",
            prev_info.get("run_id", "?"), prev_score, sign, delta, verdict,
        )
    elif prev_info is None:
        logger.info("  vs    : no previous run found (first baseline)")
    logger.info("=" * 80)

    logger.info("=" * 80)
    logger.info("DEEPEVAL EVALUATION RESULTS")
    logger.info("=" * 80)

    pass_rate = df["passed"].mean() * 100
    logger.info(
        "\nOverall pass rate: %.1f%%  (%d/%d cases)",
        pass_rate, int(df["passed"].sum()), len(df),
    )

    # --- Per-question score table ---
    logger.info("\nPER-QUESTION RESULTS")
    short_headers = [METRIC_SHORT.get(c, c) for c in cols]
    header = (
        f"  {'#':>3}  {'Question':<36}  {'Pass':<5}  "
        + "  ".join(f"{s:<6}" for s in short_headers)
    )
    logger.info(header)
    logger.info("  " + "-" * (len(header) - 2))
    for idx, row in df.iterrows():
        q = str(row["question"])
        q_trunc = (q[:34] + "..") if len(q) > 36 else q
        pass_label = "PASS" if row["passed"] else "FAIL"
        scores = "  ".join(
            f"{row[c]:<6.3f}" if pd.notna(row.get(c)) else f"{'N/A':<6}"
            for c in cols
        )
        logger.info("  %3d  %-36s  %-5s  %s", idx + 1, q_trunc, pass_label, scores)

    # --- Statistical summary ---
    logger.info("\n" + "=" * 80)
    logger.info("METRIC STATISTICS")
    logger.info("=" * 80)
    for col in cols:
        vals = df[col].dropna().tolist()
        if not vals:
            continue
        avg = statistics.mean(vals)
        mn  = min(vals)
        mx  = max(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        threshold = THRESHOLDS.get(col, 0.5)
        status = "PASS" if _metric_passes_aggregate(col, avg) else "FAIL"
        direction = "(lower=better)" if col in _LOWER_IS_BETTER else ""
        logger.info(
            "  %s  %-25s  avg=%.3f  min=%.3f  max=%.3f  std=%.3f  "
            "(threshold: %s) %s",
            status, col, avg, mn, mx, std, threshold, direction,
        )

    # --- Failure diagnosis ---
    logger.info("\n" + "=" * 80)
    logger.info("FAILURE DIAGNOSIS")
    logger.info("=" * 80)

    failures_by_metric: dict[str, list] = {}
    for result in eval_results.test_results:
        for md in result.metrics_data:
            if md.score is not None and md.success is False:
                failures_by_metric.setdefault(md.name, []).append(
                    (result.input, md.reason)
                )

    if not failures_by_metric:
        logger.info("\n  All metrics passed! No failures to diagnose.\n")
    else:
        for metric_name, cases in failures_by_metric.items():
            count = len(cases)
            logger.info(
                "\n  FAIL  %s  (%d failure%s)",
                metric_name.upper(), count, "s" if count != 1 else "",
            )
            logger.info("  " + "-" * 60)
            for question, reason in cases:
                q_display = (question[:70] + "...") if len(question) > 70 else question
                r_display = reason or "No reason provided"
                logger.info("    Q: %s", q_display)
                logger.info("       -> %s", r_display)

    logger.info("\n" + "=" * 80)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(CONFIG["output_dir"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(run_dir)

    logger.info("=" * 60)
    logger.info("RAG Evaluation Framework")
    logger.info("Run ID : %s", run_id)
    logger.info("Output : %s", run_dir)
    logger.info("=" * 60)

    # --- Load dataset ---
    logger.info("Loading dataset...")
    raw_data = load_dataset(CONFIG.get("data_path"))
    dataset_label = str(CONFIG.get("data_path") or "mock")

    max_cases = CONFIG.get("max_cases")
    if max_cases is not None:
        raw_data = raw_data[: int(max_cases)]
        logger.info("Capped to %d cases (max_cases=%s)", len(raw_data), max_cases)

    # --- Build test cases ---
    test_cases = build_test_cases(raw_data)
    logger.info("Built %d test cases from %d raw records.", len(test_cases), len(raw_data))

    if not test_cases:
        logger.error("No valid test cases — aborting.")
        raise SystemExit(1)

    # --- Evaluate ---
    t0 = time.monotonic()
    llm = OllamaModel(model=CONFIG["model"])
    eval_results = run_evaluation(test_cases, llm)
    duration = time.monotonic() - t0

    # --- Build DataFrames ---
    df = build_results_df(eval_results)
    failures_df = build_failures_df(eval_results)

    # --- Health score & regression check ---
    health_score = compute_health_score(df)
    prev_info = find_previous_run(run_dir)

    # --- Console summary ---
    print_summary(df, eval_results, health_score, prev_info)

    # --- Save Excel ---
    xlsx_path = run_dir / "results.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="All Results", index=False)
        failures_df.to_excel(writer, sheet_name="Failures", index=False)
    logger.info("Excel saved → %s", xlsx_path)

    # --- Save JSON results ---
    json_path = run_dir / "results.json"
    df.to_json(json_path, orient="records", indent=2, default_handler=str)
    logger.info("JSON results saved → %s", json_path)

    # --- Save run_info.json ---
    pass_rate = float(df["passed"].mean()) if not df.empty else 0.0
    run_info = {
        "run_id":           run_id,
        "model":            CONFIG["model"],
        "dataset":          dataset_label,
        "n_cases":          len(test_cases),
        "thresholds":       THRESHOLDS,
        "pass_rate":        round(pass_rate, 4),
        "health_score":     health_score,
        "duration_seconds": round(duration, 2),
    }
    run_info_path = run_dir / "run_info.json"
    with open(run_info_path, "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)
    logger.info("Run info saved → %s", run_info_path)

    logger.info("\nDone. All outputs written to %s/", run_dir)
