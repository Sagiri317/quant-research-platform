from __future__ import annotations

import warnings
from datetime import timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from quant_research_platform.research.models.fitting import _collect_training_frame
from quant_research_platform.runtime.dataframes import (
    iter_complete_session_frames,
    iter_complete_time_frames,
)


def test_complete_session_frames_merge_boundary_and_emit_one_session() -> None:
    frames = iter(
        (
            pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(
                        ["2024-01-01", "2024-01-01", "2024-01-02"]
                    ),
                    "value": [1, 2, 3],
                }
            ),
            pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                    "value": [4, 5],
                }
            ),
        )
    )
    chunks = list(iter_complete_session_frames(frames, timestamp_column="timestamp"))
    assert [len(chunk) for chunk in chunks] == [2, 2, 1]
    assert pd.concat(chunks).value.tolist() == [1, 2, 3, 4, 5]
    assert [chunk["timestamp"].nunique() for chunk in chunks] == [1, 1, 1]


def test_complete_time_sized_frames_preserves_all_empty_chunk_schema() -> None:
    first = pd.DataFrame(
        {
            "timestamp": pd.Series(dtype="datetime64[ns]"),
            "asset_id": pd.Series(dtype="object"),
            "factor": pd.Series(dtype="float64"),
            "eligible": pd.Series(dtype="bool"),
        }
    )
    second = first.iloc[0:0].copy()
    chunks = list(
        iter_complete_time_frames(iter((first, second)), timestamp_column="timestamp")
    )
    assert len(chunks) == 1
    assert chunks[0].empty
    assert list(chunks[0].columns) == list(first.columns)
    assert chunks[0].dtypes.to_dict() == first.dtypes.to_dict()
    assert chunks[0].loc[:, ["asset_id", "factor", "eligible"]].empty


def test_complete_frames_accept_equivalent_timezone_representations() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2024-01-15T12:00:00", tz=timezone(timedelta(hours=1))),
                pd.Timestamp("2024-01-15T12:01:00", tz=ZoneInfo("Europe/Paris")),
            ],
            "asset_id": ["asset", "asset"],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        chunks = list(
            iter_complete_time_frames(iter((frame,)), timestamp_column="timestamp")
        )
    assert sum(len(chunk) for chunk in chunks) == 2


@pytest.mark.parametrize("exit_kind", ("normal", "body_error", "consumer_close"))
def test_complete_time_frames_close_upstream_after_activation(exit_kind: str) -> None:
    close_calls = 0

    class Frames:
        def __init__(self) -> None:
            first = pd.DataFrame(
                {"timestamp": pd.to_datetime(["2026-01-01"]), "value": [1]}
            )
            second = pd.DataFrame(
                {"timestamp": pd.to_datetime(["2026-01-02"]), "value": [2]}
            )
            if exit_kind == "body_error":
                first = first.rename(columns={"timestamp": "wrong"})
            self._frames = iter((first, second))

        def __iter__(self) -> Frames:
            return self

        def __next__(self) -> pd.DataFrame:
            return next(self._frames)

        def close(self) -> None:
            nonlocal close_calls
            close_calls += 1

    chunks = iter_complete_time_frames(Frames(), timestamp_column="timestamp")
    if exit_kind == "normal":
        assert len(list(chunks)) == 2
    elif exit_kind == "body_error":
        with pytest.raises(ValueError, match="missing_time_chunk_timestamp_column"):
            next(chunks)
    else:
        next(chunks)
        chunks.close()
    assert close_calls == 1


@pytest.mark.parametrize("session_chunks", (False, True))
@pytest.mark.parametrize("consumer_primary", (False, True))
def test_complete_time_frames_early_close_reports_cleanup_failures(
    session_chunks, consumer_primary
):
    from quant_research_platform.policy.lifecycle import closing_iterator

    closed = []

    class Frames:
        def __init__(self):
            self.frames = iter(
                (
                    pd.DataFrame(
                        {
                            "timestamp": pd.date_range("2026-01-01", periods=3),
                            "value": [1, 2, 3],
                        }
                    ),
                )
            )

        def __iter__(self):
            return self

        def __next__(self):
            return next(self.frames)

        def close(self):
            closed.append(True)
            raise OSError("upstream_close")

    factory = (
        iter_complete_session_frames if session_chunks else iter_complete_time_frames
    )
    chunks = factory(Frames(), timestamp_column="timestamp")
    next(chunks)
    primary = KeyboardInterrupt("time_consumer_primary")
    with pytest.raises(KeyboardInterrupt if consumer_primary else OSError) as caught:
        if consumer_primary:
            with closing_iterator(chunks):
                raise primary
        else:
            chunks.close()
    if consumer_primary:
        assert caught.value is primary
        assert any("OSError" in note for note in getattr(primary, "__notes__", ()))
    assert closed == [True]


@pytest.mark.parametrize("chunk_rows", (1, 4, 8))
def test_model_training_concat_keeps_irregularly_selected_rows(chunk_rows):
    from quant_research_platform.contracts.models import ModelSpec
    from quant_research_platform.research.models.fitting import (
        _fit_model_result,
    )
    from quant_research_platform.research.models.supervised import (
        build_estimator_from_model_spec,
    )

    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2020-01-01"] * 4 + ["2020-01-02"] * 4),
            "label_available_at": pd.Timestamp("2020-01-03"),
            "x": [float(i) for i in range(8)],
            "label": [2.0 * i + 1.0 for i in range(8)],
            "eligible": [True, False, True, True] * 2,
        }
    )
    spec = ModelSpec(
        model_id="concat_index",
        model_algorithm="linear_regression",
        input_factor_columns=("x",),
        target_column="label",
    )
    estimator = build_estimator_from_model_spec(spec)
    fitted = _fit_model_result(
        _collect_training_frame(
            (frame.iloc[i : i + chunk_rows] for i in range(0, 8, chunk_rows)),
            spec,
            fitting_cutoff="2020-01-04",
        ),
        spec,
        estimator=estimator,
    )
    assert fitted.metadata["rows_after_input_label_dropna"] == 6
    actual = fitted.prediction_model.predict(pd.DataFrame({"x": [0.0, 1.0, 4.0]}))
    assert actual.tolist() == pytest.approx([1.0, 3.0, 9.0])
