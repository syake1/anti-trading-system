import pandas as pd

from src.intraday import entry_confirmed, finalized_bars, parabolic_message


def test_entry_requires_parabolic_sar_transition():
    assert entry_confirmed(reversal=True, sar=True, candle=True)
    assert not entry_confirmed(reversal=True, sar=False, candle=True)


def test_entry_requires_k_reversal_and_matching_candle():
    assert not entry_confirmed(reversal=False, sar=True, candle=True)
    assert not entry_confirmed(reversal=True, sar=True, candle=False)


def test_finalized_bars_keeps_latest_completed_bar():
    index = pd.DatetimeIndex(["2026-08-21 09:00", "2026-08-21 09:15", "2026-08-21 09:30"],
                             tz="Asia/Tokyo")
    frame = pd.DataFrame({"Close": [100, 101, 102]}, index=index)

    at_0946 = finalized_bars(frame, pd.Timestamp("2026-08-21 09:46", tz="Asia/Tokyo"))

    assert at_0946.index.tolist() == index.tolist()


def test_finalized_bars_excludes_only_forming_bar():
    index = pd.DatetimeIndex(["2026-08-21 09:00", "2026-08-21 09:15", "2026-08-21 09:30"],
                             tz="Asia/Tokyo")
    frame = pd.DataFrame({"Close": [100, 101, 102]}, index=index)

    at_0940 = finalized_bars(frame, pd.Timestamp("2026-08-21 09:40", tz="Asia/Tokyo"))

    assert at_0940.index.tolist() == index[:2].tolist()


def test_discord_message_clearly_labels_parabolic_buy():
    message = parabolic_message(
        {"コード": "7203", "会社名": "トヨタ自動車"}, "買い",
        "2026-08-21T09:30:00+09:00", ["%K反転", "SAR転換", "ローソク転換"],
        pd.Timestamp("2026-08-21 09:46", tz="Asia/Tokyo"),
    )
    assert "【15分足 パラボリック買いサイン】" in message
    assert "7203 トヨタ自動車" in message
    assert "対象確定足" in message and "検出時刻" in message
