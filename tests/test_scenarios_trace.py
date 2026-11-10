"""Mandatory correctness test (section 16):
Every number shown in scenario narratives maps to an empirical stored value.
"""
import sys
import os
import re
import pandas as pd

sys.path.insert(0, os.path.abspath("src"))
from scenarios import lookup_analog_narrative


def test_scenarios_trace():
    analog_table = pd.DataFrame([
        {
            "vol_bin": "low_vol",
            "time_bin": "afternoon",
            "move_bin": "mild_down",
            "count": 350,
            "median": 0.0015,
            "p25": -0.0010,
            "p75": 0.0040,
            "share_continuing": 0.42,
            "share_reversing": 0.58,
        }
    ])

    narrative = lookup_analog_narrative(
        analog_table=analog_table,
        vol_bin="low_vol",
        time_bin="afternoon",
        move_bin="mild_down",
        side_direction="down",
        conf_pct=64.0,
        horizon_min=30,
    )

    # Verify that numbers appearing in the narrative match table values
    assert "n=350" in narrative, "Sample count n=350 missing or mismatched"
    assert "42%" in narrative, "Continuing share 42% missing or mismatched"
    assert "58%" in narrative, "Reversing share 58% missing or mismatched"
    assert "+0.15%" in narrative, "Median return +0.15% missing or mismatched"
    print("Narrative verified:", narrative)


if __name__ == "__main__":
    test_scenarios_trace()
    print("test_scenarios_trace PASSED!")
