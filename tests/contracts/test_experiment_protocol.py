"""Scientific fold chronology, with final test frozen by explicit selection records."""

import pytest

from quant_research_platform.contracts.experiments import ExperimentFold
from quant_research_platform.contracts.time import TimeWindow


def test_fold_has_only_train_and_validation():
    fold = ExperimentFold(
        fold_id="fold",
        train=TimeWindow("2020-01-01", "2020-01-03"),
        validation=TimeWindow("2020-01-04", "2020-01-06"),
    )
    assert fold.train.end < fold.validation.start
    with pytest.raises(TypeError):
        ExperimentFold(
            fold_id="fold",
            train=fold.train,
            validation=fold.validation,
            test=TimeWindow("2020-01-07", "2020-01-09"),
        )


@pytest.mark.parametrize("validation_start", ["2020-01-02", "2020-01-03"])
def test_fold_rejects_training_validation_overlap(validation_start):
    with pytest.raises(ValueError, match="overlap"):
        ExperimentFold(
            fold_id="fold",
            train=TimeWindow("2020-01-01", "2020-01-03"),
            validation=TimeWindow(validation_start, "2020-01-06"),
        )


def test_fold_rejects_mixed_day_and_intraday_clocks():
    with pytest.raises(ValueError, match="time_kind"):
        ExperimentFold(
            fold_id="fold",
            train=TimeWindow("2020-01-01", "2020-01-03"),
            validation=TimeWindow("2020-01-04T01:00:00", "2020-01-06T01:00:00"),
        )
