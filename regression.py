"""
Regression detector -- compares two saved eval runs and fails if quality dropped.

This turns the framework into a genuine test suite: run it in CI against a baseline,
and a real quality regression returns a non-zero exit code that blocks the change.

Thresholds (not exact match): LLM-judge scores carry small natural variance, so we
flag only drops beyond a tolerance -- catching real regressions without tripping on
noise. Improvements and within-tolerance changes pass.
"""

import sys
import json
import glob
import os

RESULTS_DIR = "results"

# How much a metric may drop before we call it a regression.
THRESHOLDS = {
    "faithfulness_avg": 0.3,   # 1-5 scale
    "correctness_avg": 0.3,    # 1-5 scale
    "refusal_accuracy": 0.1,   # 0-1 scale
    "retrieval_hit": 0.1,      # 0-1 scale
    "retrieval_mrr": 0.1,      # 0-1 scale
}


def load_run(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def latest_run(config: str) -> str:
    """Most recent saved run file for a given config."""
    matches = sorted(glob.glob(os.path.join(RESULTS_DIR, f"run_{config}_*.json")))
    if not matches:
        raise FileNotFoundError(f"No saved runs found for config '{config}'")
    return matches[-1]


def compare(baseline: dict, candidate: dict) -> dict:
    """Compute per-metric deltas and whether each is a regression."""
    base = baseline["summary"]["overall"]
    cand = candidate["summary"]["overall"]

    findings = []
    regressed = False

    for metric, threshold in THRESHOLDS.items():
        b = base.get(metric, 0.0)
        c = cand.get(metric, 0.0)
        delta = round(c - b, 3)
        is_regression = delta < -threshold
        if is_regression:
            regressed = True
        findings.append({
            "metric": metric,
            "baseline": b,
            "candidate": c,
            "delta": delta,
            "threshold": threshold,
            "regression": is_regression,
        })

    return {"regressed": regressed, "findings": findings}


def main():
    # Usage: python regression.py <baseline_config> <candidate_config>
    baseline_cfg = sys.argv[1] if len(sys.argv) > 1 else "grounded"
    candidate_cfg = sys.argv[2] if len(sys.argv) > 2 else "grounded"

    baseline = load_run(latest_run(baseline_cfg))
    candidate = load_run(latest_run(candidate_cfg))

    print(f"Baseline:  {baseline_cfg}  ({baseline['config']['timestamp']})")
    print(f"Candidate: {candidate_cfg}  ({candidate['config']['timestamp']})")
    print("=" * 68)
    print(f"{'metric':<22}{'base':>8}{'cand':>8}{'delta':>8}   status")
    print("-" * 68)

    result = compare(baseline, candidate)
    for f in result["findings"]:
        status = "REGRESSION" if f["regression"] else "ok"
        arrow = "↓" if f["delta"] < 0 else ("↑" if f["delta"] > 0 else "=")
        print(f"{f['metric']:<22}{f['baseline']:>8}{f['candidate']:>8}"
              f"{f['delta']:>7}{arrow}   {status}")

    print("=" * 68)
    if result["regressed"]:
        print("RESULT: ❌ REGRESSION DETECTED -- quality dropped beyond threshold.")
        sys.exit(1)
    else:
        print("RESULT: ✅ PASS -- no regression beyond threshold.")
        sys.exit(0)


if __name__ == "__main__":
    main()