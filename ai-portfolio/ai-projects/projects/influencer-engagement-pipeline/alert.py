"""
alert.py — Influencer Engagement Pipeline
Mock retraining alert based on drift monitoring results.

In production, this task would send a Slack/PagerDuty/email alert
when data drift exceeds thresholds. Here it logs structured alerts
and writes an alert summary for the Streamlit dashboard.
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).parent
DRIFT_DIR = PROJECT_DIR / "data" / "drift"


def run_alert_check(drift_result: dict | None = None) -> dict:
    """
    Check drift results against alerting thresholds.
    Returns alert status and recommended action.
    """
    DRIFT_DIR.mkdir(parents=True, exist_ok=True)

    # Load drift results if not passed directly
    if drift_result is None:
        summary_path = DRIFT_DIR / "drift_summary.json"
        if not summary_path.exists():
            return {
                "alert_status": "no_data",
                "message": "No drift summary found. Run drift monitoring first.",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        drift_result = json.loads(summary_path.read_text())

    dataset_drift = drift_result.get("dataset_drift", False)
    drifted_features = drift_result.get("drifted_features", [])
    critical_features = drift_result.get("critical_features", [])
    drift_details = drift_result.get("drift_results", drift_result.get("per_feature", {}))

    # Determine alert level
    if critical_features:
        alert_status = "critical"
        severity = "CRITICAL"
        action = "Immediate retraining recommended. Critical drift in: " + ", ".join(critical_features)
    elif len(drifted_features) > 3:
        alert_status = "warning"
        severity = "WARNING"
        action = f"Multiple features drifted ({len(drifted_features)}). Schedule retraining within 1 week."
    elif drifted_features:
        alert_status = "info"
        severity = "INFO"
        action = f"Minor drift detected in {len(drifted_features)} feature(s). Monitor closely."
    else:
        alert_status = "ok"
        severity = "OK"
        action = "No significant drift detected. Model performance stable."

    alert = {
        "alert_status": alert_status,
        "severity": severity,
        "dataset_drift": dataset_drift,
        "drifted_feature_count": len(drifted_features),
        "drifted_features": drifted_features,
        "critical_features": critical_features,
        "recommended_action": action,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "thresholds": {
            "psi_warn": 0.1,
            "psi_moderate": 0.2,
            "psi_severe": 0.25,
        },
    }

    # Log the alert
    log_fn = {
        "critical": logger.critical,
        "warning": logger.warning,
        "info": logger.info,
        "ok": logger.info,
    }.get(alert_status, logger.info)

    log_fn(
        "[%s] Retraining Alert — %s | Drifted: %d features | Action: %s",
        severity, alert_status, len(drifted_features), action,
    )

    # In production: send Slack/PagerDuty notification here
    # slack_webhook.send(alert) or pagerduty.trigger(alert)

    # Save alert JSON
    alert_path = DRIFT_DIR / "alert.json"
    alert_path.write_text(json.dumps(alert, indent=2))
    logger.info("Alert saved to %s", alert_path)

    return alert


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Check drift and generate retraining alert")
    parser.add_argument(
        "--drift-summary", type=str, default=None,
        help="Path to drift_summary.json (default: data/drift/drift_summary.json)",
    )
    args = parser.parse_args()

    drift_result = None
    if args.drift_summary:
        drift_result = json.loads(Path(args.drift_summary).read_text())

    alert = run_alert_check(drift_result)

    print("\n=== Retraining Alert ===")
    print(f"  Status:     {alert['severity']}")
    print(f"  Drifted:    {alert['drifted_feature_count']} features")
    if alert["drifted_features"]:
        print(f"  Features:   {', '.join(alert['drifted_features'])}")
    if alert["critical_features"]:
        print(f"  Critical:   {', '.join(alert['critical_features'])}")
    print(f"  Action:     {alert['recommended_action']}")
    print(f"  Timestamp:  {alert['timestamp']}")
