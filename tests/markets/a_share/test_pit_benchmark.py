from __future__ import annotations

import pandas as pd
import pytest

from quant_research_platform.markets.implementations.a_share.benchmark import (
    materialize_benchmark_weights,
    materialize_industry_history,
    normalize_benchmark_history,
)
from quant_research_platform.markets.implementations.a_share.factors import (
    calculations as factor_calculations,
)
from quant_research_platform.markets.implementations.a_share.factors import (
    definitions as factor_definitions,
)
from quant_research_platform.sources.csmar.fundamentals import (
    financial_events_from_interim,
    materialize_pit_fundamentals,
)
from quant_research_platform.sources.csmar.interim import (
    interim_catalog,
    normalize_raw_frame,
)


def _financial_interim() -> dict[str, pd.DataFrame]:
    common = {
        "Stkcd": "000001",
        "ShortName": "synthetic",
        "Typrep": "A",
    }
    return {
        "financial_balance": normalize_raw_frame(
            pd.DataFrame(
                [
                    {
                        **common,
                        "IfCorrect": "0",
                        "DeclareDate": None,
                        "Accper": "2024-06-30",
                        "A001000000": 100.0,
                        "A002000000": 40.0,
                        "A003000000": 60.0,
                    }
                ]
            ),
            "financial_balance",
        ),
        "financial_income": normalize_raw_frame(
            pd.DataFrame(
                [
                    {
                        **common,
                        "IfCorrect": "1",
                        "DeclareDate": "2024-07-02",
                        "Accper": "2024-03-31",
                        "B001101000": 30.0,
                        "B002000000": 11.0,
                    },
                    {
                        **common,
                        "IfCorrect": "0",
                        "DeclareDate": None,
                        "Accper": "2024-06-30",
                        "B001101000": 50.0,
                        "B002000000": 20.0,
                    },
                ]
            ),
            "financial_income",
        ),
        "financial_cashflow": normalize_raw_frame(
            pd.DataFrame(
                [
                    {
                        **common,
                        "IfCorrect": "0",
                        "DeclareDate": None,
                        "Accper": "2024-06-30",
                        "C001000000": 15.0,
                    }
                ]
            ),
            "financial_cashflow",
        ),
    }


def test_financial_raw_interim_and_pit_keep_latest_period_over_old_correction() -> None:
    sessions = pd.date_range("2024-06-24", periods=15, freq="B")
    disclosures = pd.DataFrame(
        {
            "asset_id": ["000001"] * 4,
            "table_id": ["FS_Combas", "FS_Comins", "FS_Comins", "FS_Comscfd"],
            "report_period": pd.to_datetime(
                ["2024-06-30", "2024-03-31", "2024-06-30", "2024-06-30"]
            ),
            "first_disclosure_date": pd.to_datetime(
                ["2024-06-28", "2024-04-20", "2024-06-28", "2024-06-28"]
            ),
        }
    )
    events = financial_events_from_interim(
        _financial_interim(), disclosures, trading_sessions=sessions
    )
    pit = materialize_pit_fundamentals(
        pd.DataFrame(
            {
                "asset_id": ["000001", "000001"],
                "timestamp": pd.to_datetime(["2024-07-02", "2024-07-03"]),
            }
        ),
        events,
    )

    net_income = pit.loc[pit["metric_id"].eq("net_income")]
    before = net_income.loc[net_income["timestamp"].eq("2024-07-02")].iloc[0]
    after = net_income.loc[net_income["timestamp"].eq("2024-07-03")].iloc[0]
    assert before["report_period"] == pd.Timestamp("2024-06-30")
    assert before["value"] == 20.0
    assert after["table_id"] == "FS_Comins"
    assert after["report_period"] == pd.Timestamp("2024-06-30")
    assert after["value"] == 20.0
    factors = factor_calculations.calculate_fundamental_factors(
        pit.loc[pit["timestamp"].eq("2024-07-03")],
        pd.DataFrame(
            {
                "asset_id": ["000001"],
                "timestamp": pd.to_datetime(["2024-07-03"]),
                "market_cap": [200.0],
            }
        ),
    ).iloc[0]
    assert factors["fundamental_raw_earnings_yield"] == 0.1
    assert factors["fundamental_raw_book_to_market"] == 0.3
    assert factors["fundamental_raw_leverage"] == 0.4
    assert all(
        definition.factor_id.startswith("fundamental_raw_")
        for definition in factor_definitions.fundamental_factor_definitions()
    )


def test_financial_multiple_correction_dates_and_report_period_growth_are_pit() -> None:
    corrected = normalize_raw_frame(
        pd.DataFrame(
            [
                {
                    "Stkcd": "000001",
                    "ShortName": "synthetic",
                    "IfCorrect": "1",
                    "DeclareDate": "2024-01-02, 2024-02-02",
                    "Accper": "2023-12-31",
                    "Typrep": "A",
                    "A001000000": 100.0,
                    "A002000000": 40.0,
                    "A003000000": 60.0,
                }
            ]
        ),
        "financial_balance",
    )
    assert corrected["correction_disclosure_date"].iloc[0] == pd.Timestamp("2024-02-02")

    metrics = {
        "total_assets": ("FS_Combas", 80.0, 100.0),
        "total_liabilities": ("FS_Combas", 32.0, 40.0),
        "total_equity": ("FS_Combas", 48.0, 60.0),
        "operating_revenue": ("FS_Comins", 24.0, 30.0),
        "net_income": ("FS_Comins", 8.0, 10.0),
        "operating_cash_flow": ("FS_Comscfd", 12.0, 15.0),
    }
    events = pd.DataFrame(
        [
            {
                "asset_id": "000001",
                "table_id": table_id,
                "metric_id": metric_id,
                "report_period": pd.Timestamp(report_period),
                "available_at": pd.Timestamp(available_at),
                "is_corrected": False,
                "value": values[position],
            }
            for metric_id, (table_id, *values) in metrics.items()
            for position, (report_period, available_at) in enumerate(
                (("2023-12-31", "2024-01-02"), ("2024-03-31", "2024-04-02"))
            )
        ]
    )
    observations = pd.DataFrame(
        {
            "asset_id": ["000001", "000001"],
            "timestamp": pd.to_datetime(["2024-04-03", "2024-04-04"]),
        }
    )
    pit = materialize_pit_fundamentals(observations, events)
    factors = factor_calculations.calculate_fundamental_factors(
        pit,
        observations.assign(market_cap=200.0),
    )

    assert factors["fundamental_raw_asset_growth"].tolist() == [0.25, 0.25]


def test_first_disclosure_uses_registered_strict_source_adapter() -> None:
    raw = pd.DataFrame(
        [
            {
                "asset_id": "000001",
                "table_id": "FS_Combas",
                "report_period": "2023-12-31",
                "first_disclosure_date": "2024-03-30",
            }
        ]
    )

    normalized = normalize_raw_frame(raw, "financial_disclosures")
    spec = interim_catalog().get("financial_disclosures")

    assert tuple(normalized.columns) == spec.required_columns
    assert normalized["source_asset_id"].iloc[0] == "000001"
    assert normalized["timestamp"].iloc[0] == pd.Timestamp("2024-03-30")
    with pytest.raises(ValueError, match="^financial_disclosures_raw_schema_invalid$"):
        normalize_raw_frame(raw.assign(vendor_unknown=1), "financial_disclosures")


def test_pit_materialization_does_not_allocate_observation_event_cross_product(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observations = pd.DataFrame(
        {
            "asset_id": ["000001"] * 100,
            "timestamp": pd.date_range("2024-06-01", periods=100, freq="D"),
        }
    )
    events = pd.DataFrame(
        {
            "asset_id": ["000001"] * 100,
            "table_id": ["FS_Combas"] * 100,
            "metric_id": ["total_assets"] * 100,
            "report_period": pd.date_range("2000-03-31", periods=100, freq="QE"),
            "available_at": pd.date_range("2000-04-01", periods=100, freq="QE"),
            "is_corrected": [False] * 100,
            "value": [float(value) for value in range(100)],
        }
    )

    def reject_cross_product(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("pit_must_not_use_dataframe_merge")

    monkeypatch.setattr(pd.DataFrame, "merge", reject_cross_product)
    output = materialize_pit_fundamentals(observations, events)

    assert len(output) == 600
    total_assets = output.loc[output["metric_id"].eq("total_assets")]
    assert len(total_assets) == 100
    assert total_assets["previous_value"].notna().all()


def test_pit_fixed_metrics_are_explicit_and_state_keeps_only_current_previous() -> None:
    events = pd.DataFrame(
        [
            {
                "asset_id": "000001",
                "table_id": "FS_Combas",
                "metric_id": "total_assets",
                "report_period": pd.Timestamp(report_period),
                "available_at": pd.Timestamp(available_at),
                "is_corrected": is_corrected,
                "value": value,
            }
            for report_period, available_at, is_corrected, value in (
                ("2023-03-31", "2023-04-01", False, 100.0),
                ("2023-09-30", "2023-10-01", False, 300.0),
                ("2023-06-30", "2023-10-02", False, 200.0),
                ("2023-03-31", "2023-10-03", True, 110.0),
            )
        ]
    )
    observations = pd.DataFrame(
        {"asset_id": ["000001"], "timestamp": [pd.Timestamp("2023-10-04")]}
    )

    pit = materialize_pit_fundamentals(observations, events)
    factors = factor_calculations.calculate_fundamental_factors(
        pit, observations.assign(market_cap=1_000.0)
    )

    assert set(pit["metric_id"]) == {
        "total_assets",
        "total_liabilities",
        "total_equity",
        "operating_revenue",
        "net_income",
        "operating_cash_flow",
    }
    assert len(pit) == 6
    total_assets = pit.loc[pit["metric_id"].eq("total_assets")].iloc[0]
    assert total_assets["report_period"] == pd.Timestamp("2023-09-30")
    assert total_assets["value"] == 300.0
    assert total_assets["previous_report_period"] == pd.Timestamp("2023-06-30")
    assert total_assets["previous_value"] == 200.0
    assert factors["fundamental_raw_asset_growth"].iloc[0] == 0.5
    assert (
        factors.drop(
            columns=[
                "market_id",
                "asset_id",
                "timestamp",
                "fundamental_raw_asset_growth",
            ]
        )
        .isna()
        .all(axis=None)
    )


def test_idx_raw_contracts_bind_csi500_csi1000_and_industry_independently() -> None:
    membership_rows = []
    weight_rows = []
    return_rows = []
    for index_code in ("000905", "000852"):
        for stock_code, weight in (("000001", 60.0), ("000002", 40.0)):
            membership_rows.append(
                {
                    "Indexcd": index_code,
                    "Chgsmp01": "2024-01-02",
                    "Chgsmp02": stock_code,
                    "Chgsmp03": "synthetic",
                    "Chgsmp04": "1",
                    "Chgsmp05": "1",
                    "Chgsmp06": "2023-12-29",
                    "Chgsmp07": "SSE",
                }
            )
            weight_rows.append(
                {
                    "Indexcd": index_code,
                    "Enddt": "2024-01-02",
                    "Stkcd": stock_code,
                    "Constdnme": "synthetic",
                    "Weight": weight,
                }
            )
        return_rows.append(
            {
                "Indexcd": index_code,
                "Idxtrd01": "2024-01-02",
                "Idxtrd02": 100.0,
                "Idxtrd03": 101.0,
                "Idxtrd04": 99.0,
                "Idxtrd05": 100.5,
                "Idxtrd06": 1.0,
                "Idxtrd07": 1.0,
                "Idxtrd08": 1.0,
                "Idxtrd09": "synthetic",
            }
        )
    history = normalize_benchmark_history(
        membership_interim=normalize_raw_frame(
            pd.DataFrame(membership_rows), "index_membership"
        ),
        weight_interim=normalize_raw_frame(pd.DataFrame(weight_rows), "index_weights"),
        return_interim=normalize_raw_frame(pd.DataFrame(return_rows), "index_market"),
        index_id_mapping={"000905": "csi500", "000852": "csi1000"},
    )
    weights = materialize_benchmark_weights(
        pd.DataFrame(
            {
                "benchmark_id": ["csi500", "csi500"],
                "constituent_asset_id": ["000001", "000002"],
                "timestamp": pd.to_datetime(["2024-01-03", "2024-01-03"]),
            }
        ),
        history["weights"],
        history["membership"],
    )
    industry = materialize_industry_history(
        pd.DataFrame(
            {
                "asset_id": ["000001"],
                "timestamp": pd.to_datetime(["2024-01-03"]),
            }
        ),
        pd.DataFrame(
            {
                "source_asset_id": ["000001"],
                "timestamp": pd.to_datetime(["2024-01-02"]),
                "csrc_2012_industry_code": ["industry-a"],
            }
        ),
    )

    assert set(history["membership"]["benchmark_id"]) == {"csi500", "csi1000"}
    assert weights["weight"].sum() == 1.0
    assert industry["csrc_2012_industry_code"].iloc[0] == "industry-a"
    removed_membership = pd.concat(
        (
            history["membership"],
            pd.DataFrame(
                [
                    {
                        "benchmark_id": "csi500",
                        "constituent_asset_id": "000001",
                        "effective_date": pd.Timestamp("2024-01-04"),
                        "is_member": False,
                    }
                ]
            ),
        ),
        ignore_index=True,
    )
    removal = materialize_benchmark_weights(
        pd.DataFrame(
            {
                "benchmark_id": ["csi500", "csi500"],
                "constituent_asset_id": ["000001", "000001"],
                "timestamp": pd.to_datetime(["2024-01-03", "2024-01-05"]),
            }
        ),
        history["weights"],
        removed_membership,
    )
    assert removal["weight"].iloc[0] == 0.6
    assert pd.isna(removal["weight"].iloc[1])
