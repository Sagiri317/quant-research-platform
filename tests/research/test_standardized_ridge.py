from __future__ import annotations

import numpy as np
import pandas as pd

from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.research.models.supervised import (
    build_estimator_from_model_spec,
)


def test_standardized_ridge_fits_scaler_on_training_rows_only() -> None:
    spec = ModelSpec(
        model_id="standardized_ridge",
        model_algorithm="standardized_ridge_regression",
        input_factor_columns=("feature",),
        label_column="target",
    )
    estimator = build_estimator_from_model_spec(spec)
    estimator.fit(pd.DataFrame({"feature": [0.0, 2.0]}), np.array([0.0, 2.0]))

    assert estimator._scaler.mean_.tolist() == [1.0]  # noqa: SLF001
    predictions = estimator.predict(pd.DataFrame({"feature": [100.0]}))
    assert predictions.shape == (1,)
