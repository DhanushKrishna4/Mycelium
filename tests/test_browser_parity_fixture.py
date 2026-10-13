import json
from pathlib import Path

import pytest

from mycelium_nca.parity import run_fixture

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "parity-v1.json"


def test_frozen_torch_reference_through_deployed_numpy_checker():
    report = run_fixture(FIXTURE)
    assert report["passed"], report
    assert {check["name"] for check in report["checks"]} == {
        "perception", "network_delta", "one_step", "rollout_8", "threshold_step"}


def test_parity_checker_reports_actual_error_and_rejects_changed_tolerance():
    fixture = json.loads(FIXTURE.read_text())
    fixture["expected"]["network_delta"][0][0][0] += .1
    report = run_fixture(fixture)
    assert not report["passed"]
    check = next(item for item in report["checks"] if item["name"] == "network_delta")
    assert check["max_absolute_error"] > .09
    fixture["atol"] = 1
    with pytest.raises(ValueError, match="tolerance"):
        run_fixture(fixture)
