"""Scenario engine and analog lookup per spec section 10.3."""
import numpy as np
import pandas as pd


def predict_scenarios(quantile_models, features_row, close_price):
    """Predict Bear (q10), Base (q50), and Bull (q90) price targets.

    Ensures monotonicity: Bear <= Base <= Bull.
    """
    raw_qs = [float(quantile_models[a].predict(features_row)[0]) for a in (10, 50, 90)]
    sorted_qs = np.sort(raw_qs)
    return {
        "bear_q10": close_price * float(np.exp(sorted_qs[0])),
        "base_q50": close_price * float(np.exp(sorted_qs[1])),
        "bull_q90": close_price * float(np.exp(sorted_qs[2])),
        "ret_q10": float(sorted_qs[0]),
        "ret_q50": float(sorted_qs[1]),
        "ret_q90": float(sorted_qs[2]),
    }


def lookup_analog_narrative(analog_table, vol_bin, time_bin, move_bin, side_direction, conf_pct, horizon_min=30):
    """Generate deterministic, number-traced scenario narrative."""
    match = analog_table[
        (analog_table["vol_bin"] == vol_bin)
        & (analog_table["time_bin"] == time_bin)
        & (analog_table["move_bin"] == move_bin)
    ]
    if len(match) == 0:
        # Fallback to broader match on vol_bin and time_bin
        match = analog_table[
            (analog_table["vol_bin"] == vol_bin)
            & (analog_table["time_bin"] == time_bin)
        ]

    if len(match) > 0:
        row = match.iloc[0]
        n_samples = int(row["count"])
        cont_pct = float(row["share_continuing"] * 100)
        rev_pct = float(row["share_reversing"] * 100)
        med_ret = float(row["median"] * 100)

        narrative = (
            f"Model leans {side_direction.upper()} ({conf_pct:.0f}%) over the next {horizon_min} min. "
            f"In similar past conditions ({vol_bin}, {time_bin}; n={n_samples}), "
            f"the next move continued {side_direction} {cont_pct:.0f}% of the time "
            f"and reversed {rev_pct:.0f}% of the time (median: {med_ret:+.2f}%)."
        )
    else:
        narrative = (
            f"Model leans {side_direction.upper()} ({conf_pct:.0f}%) over the next {horizon_min} min."
        )

    return narrative

