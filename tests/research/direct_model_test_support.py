"""Publish a small supplied scientific frame through the actual direct fit operation."""

from pathlib import Path
from types import SimpleNamespace

from quant_research_platform.contracts.models import model_spec_payload
from quant_research_platform.research.models.persisted import load_model
from quant_research_platform.research.models.production import fit


def fit_frame(root, frame, spec, *, cutoff=None, market_id="crypto", parameters=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "input.parquet"
    frame.to_parquet(path, index=False)
    roles = {column: "context" for column in frame}
    roles.update(timestamp="timestamp", asset_id="key", eligible="mask")
    roles.update({column: "factor" for column in spec.input_factor_columns})
    roles.update({column: "feature" for column in spec.input_feature_columns})
    roles[spec.target_column] = "label"
    daily = frame.timestamp.eq(frame.timestamp.dt.normalize()).all()

    def bound(value):
        return value.date().isoformat() if daily else value.isoformat()

    config = {
        "market_id": market_id,
        "inputs": str(path),
        "output": str(root / "model"),
        "model": model_spec_payload(spec),
        "fit_windows": [
            {"start": bound(frame.timestamp.min()), "end": bound(frame.timestamp.max())}
        ],
        "fit_cutoff": (cutoff or frame.timestamp.max()).isoformat(),
        "label_unit": "decimal_return",
        "input_spec": {
            "data_id": "synthetic",
            "frequency": "1d",
            "grain": "asset_timestamp",
            "primary_key": ["timestamp", "asset_id"],
            "partition_columns": ["year"],
            "column_roles": roles,
        },
    }
    config["model"]["random_seed"] = (
        spec.random_seed if spec.random_seed is not None else 42
    )
    if parameters is not None:
        config["candidate_parameters"] = parameters
    output = fit(config)
    result, facts = load_model(output)
    return SimpleNamespace(
        root=output, result=result, facts=facts, config=config, frame=frame
    )


def predict_frame(root, frame, case, *, window):
    from quant_research_platform.artifacts.outputs.prediction import read_predictions
    from quant_research_platform.research.predictions.runner import predict

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    source = root / "input.parquet"
    frame.to_parquet(source, index=False)
    output = predict(
        {
            "model": str(case.root),
            "inputs": str(source),
            "input_spec": case.config["input_spec"],
            "output": str(root / "prediction"),
            "prediction_window": window,
            "sample_role": "validation",
        }
    )
    prediction, metadata = read_predictions(output)
    return prediction, output, metadata
