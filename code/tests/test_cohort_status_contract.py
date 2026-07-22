import json
from pathlib import Path


def test_cohort_report_has_canonical_status():
    root = Path(__file__).resolve().parents[2]
    report = json.loads((root / "validation" / "matr_cohort_validation.json").read_text(encoding="utf-8"))
    value = report.get("status", report.get("cohort audit_final_pass", False))
    assert value is True or str(value).strip().upper() == "PASS"
