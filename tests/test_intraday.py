from src.intraday import entry_confirmed


def test_entry_requires_parabolic_sar_transition():
    assert entry_confirmed(reversal=True, sar=True, candle=True)
    assert not entry_confirmed(reversal=True, sar=False, candle=True)


def test_entry_requires_k_reversal_and_matching_candle():
    assert not entry_confirmed(reversal=False, sar=True, candle=True)
    assert not entry_confirmed(reversal=True, sar=True, candle=False)
