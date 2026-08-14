"""JST-based execution guard for the automated provisional meetings."""
from __future__ import annotations

import argparse
from datetime import datetime
from zoneinfo import ZoneInfo

JST = ZoneInfo("Asia/Tokyo")


def scheduled_meeting(now: datetime | None = None) -> str | None:
    """Return the meeting due in Japan, independent of the runner's timezone."""
    current = now or datetime.now(JST)
    if current.tzinfo is None:
        current = current.replace(tzinfo=JST)
    current = current.astimezone(JST)
    if current.weekday() < 5 and (current.hour, current.minute) == (21, 0):
        return "night"
    if current.weekday() == 5 and (current.hour, current.minute) == (6, 0):
        return "weekend"
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expect", choices=("night", "weekend"), required=True)
    args = parser.parse_args()
    actual = scheduled_meeting()
    if actual != args.expect:
        raise SystemExit(f"skip: JST schedule is {actual or 'rest'}, expected {args.expect}")
    print(f"run: {actual}")
