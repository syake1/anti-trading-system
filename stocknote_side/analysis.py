"""Integration point for stocknote's existing analysis implementation.

Deployments can replace this function (or pass ``--analyzer module:function``)
without coupling stocknote to the anti-trading-system application.
"""


def analyze_candidate(*, code, official_information, reference_information):
    """Return a conservative result when no stocknote data provider is installed."""
    return {
        "assessment": "insufficient",
        "confidence": 0.0,
        "summary": "分析に必要なstocknoteデータが不足しています。",
    }
