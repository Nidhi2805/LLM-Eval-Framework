"""
Loads saved evaluation runs from the results directory for the dashboard.

Kept separate from the Streamlit UI so the loading logic is testable on its own
and the UI file stays focused on presentation.
"""

import os
import json
import glob

RESULTS_DIR = "results"


def list_runs() -> list[dict]:
    """Return all saved runs, newest first, with a friendly label each."""
    runs = []
    for path in glob.glob(os.path.join(RESULTS_DIR, "run_*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue  # skip any corrupt/partial file rather than crashing
        cfg = data.get("config", {})
        label = (
            f"{cfg.get('generator_config', 'unknown')} | "
            f"{cfg.get('timestamp', os.path.basename(path))[:19]}"
        )
        runs.append({"path": path, "label": label, "data": data})

    runs.sort(key=lambda r: r["data"].get("config", {}).get("timestamp", ""),
              reverse=True)
    return runs