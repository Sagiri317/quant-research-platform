from __future__ import annotations

import hashlib
from dataclasses import replace

import pandas as pd
import pytest

from quant_research_platform.contracts.targets import ProducerBinding, ProducerKind
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.profiles.research import research_profile
from quant_research_platform.research.protocols.a_share import DATA_START
from quant_research_platform.runtime.resources import runtime_resource_policy_scope
from quant_research_platform.sources.choice.fields import FINANCIAL_FIELDS
from quant_research_platform.sources.choice.raw_exports import choice_session
from quant_research_platform.sources.csmar.interim import InterimImportResult
from quant_research_platform.storage.datasets.reading import (
    open_dataset_session,
    read_dataset,
)
from quant_research_platform.storage.datasets.writing import write_dataset
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.workflows import research
from tests.helpers.choice_raw import SDK, response, write_css
from tests.markets.a_share.test_target_dag_chain import _synthetic_interim_frames


def test_financial_history_window_preserves_formal_research_and_early_requests():
    profile = research_profile()
    protocol = profile.protocol("fusion_nf_core")
    spec = profile.interim.get("financial_income")
    window = TimeWindow("2020-01-01", "2020-05-08")
    assert protocol.target_window("data materialize", spec, window) == TimeWindow(
        DATA_START, window.end
    )
    timestamps = TimeWindow(pd.Timestamp(window.start), pd.Timestamp(window.end))
    assert protocol.target_window("data materialize", spec, timestamps) == TimeWindow(
        pd.Timestamp(DATA_START), pd.Timestamp(window.end)
    )
    early = TimeWindow("2010-01-01", "2010-12-31")
    assert protocol.target_window("dataset build", spec, early) == early
    assert protocol.target_window("research run", spec, window) == TimeWindow(
        "2011-01-01", "2023-12-31"
    )
    assert (
        protocol.target_window(
            "dataset build", profile.processed.get("fundamental_pit"), window
        )
        == window
    )


@pytest.mark.parametrize("action", ["data materialize", "dataset build"])
def test_public_financial_windows_keep_choice_history_and_output_identity(
    tmp_path, action
):
    base = research_profile()
    raw_layout = StorageLayout(tmp_path / "raw-project")
    fields = tuple(field for mapping in FINANCIAL_FIELDS.values() for field in mapping)
    with choice_session(SDK(response())) as session:
        for period, first, value in (
            ("2019-03-31", "2019-04-20", 1.0),
            ("2019-12-31", "2020-04-20", 4.0),
            ("2020-03-31", "2020-04-30", 5.0),
        ):
            # Asset 2 has genuinely missing prior-year Q1; it must retain null TTM.
            codes = (
                ("000001.SZ",) if period == "2019-03-31" else ("000001.SZ", "000002.SZ")
            )
            correction = "2020-05-05" if period == "2019-12-31" else None
            write_css(
                session,
                raw_layout,
                period=period,
                end_date="2025-12-31",
                indicators=("STMTACTDATE", "ACTUALISSUINGDATE", "ITEMVALUECHANGEDATE"),
                data={code: [first, None, correction] for code in codes},
            )
            snapshots = [(first, value)]
            if correction:
                snapshots.append((correction, 6.0))
            for day, amount in snapshots:
                for branch in ("1", "3"):
                    write_css(
                        session,
                        raw_layout,
                        period=period,
                        end_date=day,
                        indicators=fields,
                        data={
                            code: [amount if branch == "1" else None] * len(fields)
                            for code in codes
                        },
                        branch=branch,
                    )

    sessions = pd.bdate_range("2019-01-02", "2020-05-08")
    frames = _synthetic_interim_frames(sessions)

    def producer(**kwargs):
        data_id = kwargs["data_ids"][0]
        spec = base.interim.get(data_id)
        layout = kwargs["layout"]
        root = layout.target_root(
            data_id, layout.interim_data_root(spec.source_id, data_id)
        )
        write_dataset(
            root,
            frames[data_id],
            spec=spec,
            instance_metadata=kwargs["instance_metadata"],
            resource_policy=kwargs["resource_policy"],
            borrowed_input=True,
        )
        return InterimImportResult({data_id: root}, (), "synthetic market source")

    bindings = {
        data_id: ProducerBinding(
            ProducerKind.INTERIM,
            producer,
            source_identity=lambda _root, selected: hashlib.sha256(
                selected.encode()
            ).hexdigest(),
        )
        for data_id in frames
    }
    profile = replace(base, producer_bindings={**base.producer_bindings, **bindings})
    layout = StorageLayout(tmp_path / "output")
    observations = []
    source_identities = []
    for start in ("2019-01-01", "2020-01-01"):
        request = dict(
            profile=profile,
            layout=layout,
            raw_root=raw_layout.project_root / "data/raw",
            start_date=start,
            end_date="2020-05-08",
        )
        result = (
            research.data_materialize(**request, data_ids=("fundamental_pit",))
            if action == "data materialize"
            else research.dataset_build(**request, dataset_id="fusion_nf_core")
        )
        assert result["status"] == "COMPLETE", result["status"]
        targets = {item["target_id"]: item for item in result["target_results"]}
        source_identities.append(targets["financial_income"]["instance_identity"])
        with runtime_resource_policy_scope() as policy:
            for data_id, expected_start in (
                ("financial_income", "2011-01-01"),
                ("fundamental_pit", start),
            ):
                spec = (
                    profile.interim
                    if data_id == "financial_income"
                    else profile.processed
                ).get(data_id)
                logical = (
                    layout.interim_data_root(spec.source_id, data_id)
                    if data_id == "financial_income"
                    else layout.processed_data_root(spec.market_id, spec.kind, data_id)
                )
                root = layout.instance_root(
                    logical, targets[data_id]["instance_identity"]
                )
                with open_dataset_session(
                    root, columns=None, time_window=None, resource_policy=policy
                ) as persisted:
                    envelope = persisted.manifest["common_manifest"]
                    assert (
                        envelope["instance_identity"]
                        == targets[data_id]["instance_identity"]
                    )
                    assert envelope["research_window"]["start"] == expected_start
                    assert envelope["research_window"]["end"] == "2020-05-08"
                saved = read_dataset(root, resource_policy=policy)
                if data_id == "financial_income":
                    assert saved.report_period.min() == pd.Timestamp("2019-03-31")
                    assert saved.is_corrected.any()
                else:
                    assert saved.timestamp.ge(start).all()
                    assert saved.timestamp.le("2020-05-08").all()
                    selected = saved.loc[saved.asset_id.eq("000001")].set_index(
                        "timestamp"
                    )
                    observations.append(
                        selected.loc[
                            pd.to_datetime(["2020-05-04", "2020-05-05", "2020-05-06"]),
                            "operating_revenue_ttm",
                        ].tolist()
                    )
                    incomplete = saved.loc[
                        saved.asset_id.eq("000002") & saved.timestamp.ge("2020-05-04")
                    ]
                    assert not incomplete.empty
                    assert incomplete.operating_revenue_ttm.isna().all()
            if action == "dataset build":
                root = layout.instance_root(
                    layout.research_dataset_root("a_share", "fusion_nf_core"),
                    targets["fusion_nf_core"]["instance_identity"],
                )
                dataset = read_dataset(root, resource_policy=policy)
                assert (
                    dataset.timestamp.ge(start).all()
                    and dataset.timestamp.le("2020-05-08").all()
                )
            assert policy.used_memory_bytes == 0
    assert observations == [[8.0, 8.0, 10.0], [8.0, 8.0, 10.0]]
    assert source_identities[0] == source_identities[1]
