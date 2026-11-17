"""The bad-band warning threshold scales with the requested discrepancy."""

import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from synapse_sr.pro import _warning_limit  # noqa: E402


def test_warning_limit_tracks_discrepancy():
    assert _warning_limit(0.5) == 5.0      # Pro default
    assert _warning_limit(4.0) == 20.0     # Flash default
    assert _warning_limit(2.0) == 10.0


def test_warning_limit_has_one_tau_floor():
    assert _warning_limit(0.1) == 5.0
    assert _warning_limit(0.0) == 5.0
    assert _warning_limit(-1.0) == 5.0
