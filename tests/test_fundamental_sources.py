from io import BytesIO
import zipfile

import pytest

from src.fundamental_sources import parse_xbrl


def _archive(xml: str) -> bytes:
    output = BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("XBRL/PublicDoc/report.xbrl", xml)
    return output.getvalue()


def test_xbrl_normalizes_explicit_periods_and_calculates_only_valid_yoy():
    xml = """<xbrl xmlns='http://www.xbrl.org/2003/instance' xmlns:jppfs='urn:test'>
      <context id='CurrentYearDuration'><entity><identifier scheme='x'>E1</identifier></entity><period><startDate>2025-04-01</startDate><endDate>2026-03-31</endDate></period></context>
      <context id='PriorYearDuration'><entity><identifier scheme='x'>E1</identifier></entity><period><startDate>2024-04-01</startDate><endDate>2025-03-31</endDate></period></context>
      <jppfs:Revenue contextRef='CurrentYearDuration'>1200</jppfs:Revenue>
      <jppfs:Revenue contextRef='PriorYearDuration'>1000</jppfs:Revenue>
      <jppfs:OperatingIncome contextRef='CurrentYearDuration'>90</jppfs:OperatingIncome>
      <jppfs:BasicEarningsLossPerShare contextRef='CurrentYearDuration'>50</jppfs:BasicEarningsLossPerShare>
      <jppfs:NumberOfIssuedSharesAsOfFiscalYearEnd contextRef='CurrentYearDuration'>1000000</jppfs:NumberOfIssuedSharesAsOfFiscalYearEnd>
    </xbrl>"""
    result = parse_xbrl(_archive(xml))
    assert result["revenue_yoy"] == pytest.approx(20)
    assert "operating_profit_yoy" not in result
    assert result["eps"] == 50
    assert result["shares_outstanding"] == 1000000


def test_xbrl_calculates_missing_ratios_only_from_official_operands():
    xml = """<xbrl xmlns='http://www.xbrl.org/2003/instance' xmlns:jppfs='urn:test'>
      <context id='Current'><entity><identifier scheme='x'>E1</identifier></entity><period><startDate>2025-04-01</startDate><endDate>2026-03-31</endDate></period></context>
      <context id='Prior'><entity><identifier scheme='x'>E1</identifier></entity><period><startDate>2024-04-01</startDate><endDate>2025-03-31</endDate></period></context>
      <jppfs:ProfitLoss contextRef='Current'>100</jppfs:ProfitLoss>
      <jppfs:EquityAttributableToOwnersOfParent contextRef='Current'>900</jppfs:EquityAttributableToOwnersOfParent>
      <jppfs:EquityAttributableToOwnersOfParent contextRef='Prior'>700</jppfs:EquityAttributableToOwnersOfParent>
      <jppfs:Assets contextRef='Current'>2000</jppfs:Assets>
    </xbrl>"""
    result = parse_xbrl(_archive(xml))
    assert result["roe"] == pytest.approx(12.5)
    assert result["equity_ratio"] == pytest.approx(45)


def test_xbrl_does_not_guess_a_prior_period():
    xml = """<xbrl xmlns='http://www.xbrl.org/2003/instance' xmlns:jppfs='urn:test'>
      <context id='CurrentYearDuration'><entity><identifier scheme='x'>E1</identifier></entity><period><startDate>2025-04-01</startDate><endDate>2026-03-31</endDate></period></context>
      <jppfs:Revenue contextRef='CurrentYearDuration'>1200</jppfs:Revenue>
    </xbrl>"""
    assert "revenue_yoy" not in parse_xbrl(_archive(xml))


def test_xbrl_negative_prior_uses_transition_not_misleading_yoy():
    xml = """<xbrl xmlns='http://www.xbrl.org/2003/instance' xmlns:jppfs='urn:test'>
      <context id='Current'><entity><identifier scheme='x'>E1</identifier></entity><period><startDate>2025-04-01</startDate><endDate>2026-03-31</endDate></period></context>
      <context id='Prior'><entity><identifier scheme='x'>E1</identifier></entity><period><startDate>2024-04-01</startDate><endDate>2025-03-31</endDate></period></context>
      <jppfs:ProfitLoss contextRef='Current'>50</jppfs:ProfitLoss>
      <jppfs:ProfitLoss contextRef='Prior'>-100</jppfs:ProfitLoss>
    </xbrl>"""
    result = parse_xbrl(_archive(xml))
    assert result["net_profit"] == 50
    assert result["net_profit_prior"] == -100
    assert "ordinary_or_net_profit_yoy" not in result
