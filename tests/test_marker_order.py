"""Tests ensuring markers passed to Lightweight Charts v4.2.0 are strictly ascending by time."""

from datetime import datetime, timezone


def sort_and_deduplicate_markers(signals_list: list[dict]) -> list[dict]:
    """Mirror exact logic implemented in src/web/app.js renderMarkers()."""
    markers = []
    for s in signals_list:
        direction = s.get("direction")
        if direction not in ("up", "down"):
            continue
        ts_iso = s["candle_ts"]
        dt = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
        epoch_sec = int(dt.timestamp())
        is_up = direction == "up"

        markers.append(
            {
                "time": epoch_sec,
                "position": "belowBar" if is_up else "aboveBar",
                "color": "#3b82f6" if is_up else "#f97316",
                "shape": "arrowUp" if is_up else "arrowDown",
                "text": f"{'UP' if is_up else 'DN'} {int(s.get('confidence', 0.5) * 100)}%",
                "id": ts_iso,
            }
        )

    # Sort strictly ascending by time
    markers.sort(key=lambda m: m["time"])

    # Deduplicate times
    deduped = []
    last_time = None
    for m in markers:
        if m["time"] != last_time:
            deduped.append(m)
            last_time = m["time"]

    return deduped


def test_marker_order_and_deduplication():
    # Intentionally out-of-order and with duplicates
    raw_signals = [
        {"candle_ts": "2026-11-12T15:30:00Z", "direction": "up", "confidence": 0.65},
        {"candle_ts": "2026-11-12T14:35:00Z", "direction": "down", "confidence": 0.60},
        {"candle_ts": "2026-11-12T16:00:00Z", "direction": "up", "confidence": 0.70},
        {"candle_ts": "2026-11-12T14:35:00Z", "direction": "down", "confidence": 0.62},  # duplicate
        {"candle_ts": "2026-11-12T15:00:00Z", "direction": "none", "confidence": 0.0},   # filtered
        {"candle_ts": "2026-11-12T14:40:00Z", "direction": "up", "confidence": 0.58},
    ]

    result = sort_and_deduplicate_markers(raw_signals)

    assert len(result) == 4
    times = [m["time"] for m in result]
    # Check strictly ascending
    assert times == sorted(times)
    assert len(times) == len(set(times))

