"""Real CPU fallback training, causal sequences, and native model round trips."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.outputs.prediction import read_predictions
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.research.models.persisted import load_model
from quant_research_platform.research.models.production import fit
from quant_research_platform.research.predictions.runner import predict
from tests.correctness.test_direct_model_operations import data, fit_config, input_spec


@pytest.mark.parametrize(
    "name,value", [("preload_inputs", 1), ("initialize_at_target_mean", "true")]
)
def test_sequence_options_require_booleans(name, value):
    with pytest.raises(ValueError, match=f"sequence_parameter_invalid: {name}"):
        ModelSpec(
            model_id="lstm",
            model_algorithm="lstm_regressor",
            input_feature_columns=("x",),
            target_column="label",
            hyperparameters={name: value},
        )


def test_transformer_independent_initialization_and_seed():
    import torch

    from quant_research_platform.contracts.models import sequence_parameters
    from quant_research_platform.research.models.neural import _network

    def build(layers, independent=None):
        parameters = dict(hidden_size=4, num_layers=layers, sequence_length=3)
        if independent is not None:
            parameters["independent_layer_initialization"] = independent
        parameters = sequence_parameters("transformer_regressor", parameters)
        with torch.random.fork_rng():
            torch.manual_seed(19)
            return _network("transformer_regressor", 2, parameters)

    first = build(2, True)
    second = build(2, True)
    for name, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[name], rtol=0, atol=0)
    left, right = first.encoder.layers
    assert not torch.equal(
        left.self_attn.in_proj_weight, right.self_attn.in_proj_weight
    )
    assert (
        left.self_attn.in_proj_weight.data_ptr()
        != right.self_attn.in_proj_weight.data_ptr()
    )
    for layers, explicit in ((2, False), (1, True)):
        baseline, configured = build(layers), build(layers, explicit)
        for name, value in baseline.state_dict().items():
            torch.testing.assert_close(
                value, configured.state_dict()[name], rtol=0, atol=0
            )
    with pytest.raises(ValueError, match="independent_layer_initialization"):
        sequence_parameters(
            "transformer_regressor", {"independent_layer_initialization": "true"}
        )


def test_transformer_pre_norm_matches_residual_formula_and_legacy_parameters():
    import torch
    from torch.nn import functional as functional

    from quant_research_platform.contracts.models import sequence_parameters
    from quant_research_platform.research.models.neural import _network

    parameters = sequence_parameters(
        "transformer_regressor",
        dict(hidden_size=4, n_heads=2, num_layers=2, sequence_length=3),
    )
    legacy = {key: value for key, value in parameters.items() if key != "norm_first"}

    def build(options):
        with torch.random.fork_rng():
            torch.manual_seed(19)
            return _network("transformer_regressor", 2, options).double().eval()

    old, default = build(legacy), build(parameters)
    values = torch.arange(12, dtype=torch.float64).reshape(2, 3, 2) / 12
    lengths = torch.tensor([3, 1])
    torch.testing.assert_close(
        old(values, lengths), default(values, lengths), rtol=0, atol=0
    )
    assert old.encoder.norm is None

    model = build(
        {**parameters, "norm_first": True, "independent_layer_initialization": True}
    )
    encoded = model.projection(values) + model.position(torch.arange(3))
    padding = torch.arange(3)[None, :] >= lengths[:, None]

    def normalize(value, layer):
        return functional.layer_norm(value, (4,), layer.weight, layer.bias, 1e-5)

    for layer in model.encoder.layers:
        normalized = normalize(encoded, layer.norm1)
        attended = layer.self_attn(
            normalized,
            normalized,
            normalized,
            key_padding_mask=padding,
            need_weights=False,
        )[0]
        encoded = encoded + attended
        encoded = encoded + layer.linear2(
            functional.relu(layer.linear1(normalize(encoded, layer.norm2)))
        )
    final = normalize(encoded, model.encoder.norm)[torch.arange(2), lengths - 1]
    expected = model.output(final).squeeze(-1)
    torch.testing.assert_close(model(values, lengths), expected, rtol=1e-10, atol=1e-12)
    with pytest.raises(ValueError, match="sequence_parameter_invalid: norm_first"):
        sequence_parameters("transformer_regressor", {"norm_first": "true"})


def test_lstm_input_initialization_preserves_other_parameters_and_seed():
    import torch

    from quant_research_platform.contracts.models import sequence_parameters
    from quant_research_platform.research.models.neural import _network

    def build(enabled=None):
        parameters = dict(hidden_size=16, num_layers=2)
        if enabled is not None:
            parameters["input_gate_xavier_initialization"] = enabled
        with torch.random.fork_rng():
            torch.manual_seed(42)
            return _network(
                "lstm_regressor", 128, sequence_parameters("lstm_regressor", parameters)
            )

    baseline, disabled, configured, repeated = (
        build(),
        build(False),
        build(True),
        build(True),
    )
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(value, disabled.state_dict()[name], rtol=0, atol=0)
        torch.testing.assert_close(
            configured.state_dict()[name], repeated.state_dict()[name], rtol=0, atol=0
        )
        if name != "encoder.weight_ih_l0":
            torch.testing.assert_close(
                value, configured.state_dict()[name], rtol=0, atol=0
            )
    # Each gate has fan_in=128 and fan_out=16, rather than the packed 64 rows.
    bound = (6 / (128 + 16)) ** 0.5
    for gate in configured.encoder.weight_ih_l0.chunk(4, dim=0):
        assert float(gate.detach().abs().max()) <= bound
        assert float(gate.detach().square().mean()) == pytest.approx(
            2 / (128 + 16), rel=0.1
        )
    assert not torch.equal(
        baseline.encoder.weight_ih_l0, configured.encoder.weight_ih_l0
    )
    with pytest.raises(ValueError, match="input_gate_xavier_initialization"):
        sequence_parameters(
            "lstm_regressor", {"input_gate_xavier_initialization": "true"}
        )


@pytest.mark.parametrize(
    "layers,dropout,training", [(1, 0.4, True), (2, 0.0, True), (2, 0.4, False)]
)
def test_lstm_valid_prefix_matches_gate_equations(layers, dropout, training):
    import torch

    from quant_research_platform.research.models.neural import _network

    model = _network(
        "lstm_regressor", 2, dict(hidden_size=2, num_layers=layers, dropout=dropout)
    ).double()
    model.train(training)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(
                torch.linspace(-0.3, 0.4, parameter.numel()).reshape(parameter.shape)
            )
    reference = {
        name: value.detach().clone().requires_grad_()
        for name, value in model.named_parameters()
    }
    values = (
        torch.arange(30, dtype=torch.float64).reshape(3, 5, 2) / 30
    ).requires_grad_()
    inputs = values.detach().clone().requires_grad_()
    lengths = torch.tensor([5, 1, 3])
    finals = []
    for row, length in enumerate(lengths):
        sequence = inputs[row, :length]
        for layer in range(layers):
            hidden, cell = inputs.new_zeros(2), inputs.new_zeros(2)
            outputs = []
            for observation in sequence:
                gates = (
                    reference[f"encoder.weight_ih_l{layer}"] @ observation
                    + reference[f"encoder.bias_ih_l{layer}"]
                    + reference[f"encoder.weight_hh_l{layer}"] @ hidden
                    + reference[f"encoder.bias_hh_l{layer}"]
                )
                incoming, forget, candidate, outgoing = gates.chunk(4)
                cell = forget.sigmoid() * cell + incoming.sigmoid() * candidate.tanh()
                hidden = outgoing.sigmoid() * cell.tanh()
                outputs.append(hidden)
            sequence = torch.stack(outputs)
        finals.append(sequence[-1])
    expected = (
        torch.stack(finals) @ reference["output.weight"].T + reference["output.bias"]
    ).squeeze(-1)
    actual = model(values, lengths)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
    weights = values.new_tensor([0.4, 2.0, 1.5])
    targets = values.new_tensor([-0.2, 0.3, 0.7])
    ((actual - targets).square() * weights).sum().backward()
    ((expected - targets).square() * weights).sum().backward()
    torch.testing.assert_close(values.grad, inputs.grad, rtol=1e-10, atol=1e-12)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(
            parameter.grad, reference[name].grad, rtol=1e-10, atol=1e-12
        )
    padding = torch.arange(values.shape[1])[None, :] >= lengths[:, None]
    assert values.grad[padding].eq(0).all()


def test_lstm_recurrent_dropout_preserves_packed_training_random_stream():
    import torch

    from quant_research_platform.research.models.neural import _network

    model = _network(
        "lstm_regressor", 2, dict(hidden_size=3, num_layers=2, dropout=0.4)
    )
    values = torch.arange(30, dtype=torch.float32).reshape(3, 5, 2) / 30
    lengths = torch.tensor([5, 1, 3])
    with torch.random.fork_rng():
        torch.manual_seed(17)
        actual = model(values, lengths)
        actual_random_state = torch.random.get_rng_state()
        torch.manual_seed(17)
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            values, lengths, batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = model.encoder(packed)
        expected = model.output(hidden[-1]).squeeze(-1)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.equal(actual_random_state, torch.random.get_rng_state())


@pytest.mark.parametrize("algorithm", ["lstm_regressor", "transformer_regressor"])
@pytest.mark.parametrize("days", [[6, 7], [6, 9]])
@pytest.mark.parametrize("preload_inputs", [False, True])
def test_validation_loss_uses_observed_history_between_sample_windows(
    tmp_path, algorithm, days, preload_inputs
):
    inputs = data(tmp_path)
    frame = pd.read_parquet(inputs)
    gap = frame.timestamp.dt.day.gt(3) & ~frame.timestamp.dt.day.isin(days)
    frame.loc[gap, "x"] += frame.loc[gap, "timestamp"].dt.day * 10
    unavailable = gap & frame.timestamp.dt.day.mod(2).eq(0)
    frame.loc[unavailable, "label"] = np.nan
    frame.loc[unavailable, "label_available_at"] = pd.Timestamp("2030-01-01")
    frame.to_parquet(inputs, index=False, row_group_size=2)
    config = fit_config(tmp_path, inputs)
    config["model"].update(
        model_algorithm=algorithm,
        hyperparameters={
            "device": "cpu",
            "hidden_size": 4,
            "epochs": 1,
            "sequence_length": 3,
            "preload_inputs": preload_inputs,
        },
    )
    config["validation"] = {
        "windows": [
            {"start": f"2020-01-{day:02d}", "end": f"2020-01-{day:02d}"} for day in days
        ],
        "cutoff": f"2020-01-{days[-1]:02d}",
    }
    model_path = fit(config)
    result, _ = load_model(model_path)
    assert (
        result.metadata["resolved_hyperparameters"]["preload_inputs"] is preload_inputs
    )
    predictions, _ = read_predictions(
        predict(
            {
                "model": str(model_path),
                "inputs": {"path": str(inputs), "spec": input_spec()},
                "output": str(tmp_path / "pred"),
                "prediction_window": {
                    "start": f"2020-01-{days[0]:02d}",
                    "end": f"2020-01-{days[-1]:02d}",
                },
            }
        )
    )
    scored = predictions.loc[predictions.timestamp.dt.day.isin(days)].merge(
        frame[["timestamp", "asset_id", "label"]],
        on=["timestamp", "asset_id"],
        validate="one_to_one",
    )
    assert len(scored) == len(days) * 3
    assert result.metadata["rows_after_input_label_dropna"] == 9
    assert result.prediction_model.scaler.mean_ == pytest.approx((2.0,))
    assert result.metadata["learning_curve"]["epochs"][-1][
        "validation_loss"
    ] == pytest.approx(np.mean((scored.prediction_score - scored.label) ** 2), rel=1e-6)


@pytest.mark.parametrize("algorithm", ["lstm_regressor", "transformer_regressor"])
@pytest.mark.parametrize("grouped", [False, True])
def test_sequence_cpu_fallback_boundaries_and_native_roundtrip(
    tmp_path, algorithm, grouped, monkeypatch
):
    import torch

    from quant_research_platform.artifacts.explanations import read_explanation
    from quant_research_platform.research.models.explanations import explain

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    inputs = data(tmp_path)
    frame = pd.read_parquet(inputs)
    episodes = ["X", "Y", "X", "X", "X", "X", "X", "Y", "X"]
    frame["episode"] = frame.timestamp.dt.day.map(dict(enumerate(episodes, 1)))
    hidden = frame.timestamp.eq("2020-01-05") & frame.asset_id.eq("a")
    frame.loc[hidden, "x"] = 40.0
    frame.loc[
        hidden | (frame.timestamp.eq("2020-01-08") & frame.asset_id.eq("b")),
        "eligible",
    ] = False
    frame.loc[frame.timestamp.eq("2020-01-06"), "x"] = np.nan
    frame.to_parquet(inputs, index=False, row_group_size=2)
    spec = input_spec()
    spec["column_roles"]["episode"] = "context"
    config = fit_config(tmp_path, inputs)
    config["inputs"]["spec"] = spec
    config["model"].update(
        model_algorithm=algorithm,
        hyperparameters={
            "hidden_size": 4,
            "epochs": 1,
            "sequence_length": 4,
            "sequence_group_columns": ["episode"] if grouped else [],
            **(
                {
                    "num_layers": 2,
                    "independent_layer_initialization": True,
                    "norm_first": True,
                }
                if algorithm == "transformer_regressor"
                else {"input_gate_xavier_initialization": True}
            ),
        },
    )
    config["candidate_parameters"] = None
    config["validation"] = {
        "windows": [{"start": "2020-01-04", "end": "2020-01-09"}],
        "cutoff": "2020-01-09",
    }
    fitted = fit(config)
    result, _ = load_model(fitted)
    model = result.prediction_model
    assert result.metadata["resolved_hyperparameters"]["device"] == "cuda"
    assert result.metadata["resolved_hyperparameters"]["preload_inputs"] is False
    if algorithm == "transformer_regressor":
        assert result.metadata["resolved_hyperparameters"]["norm_first"] is True
        assert (
            result.metadata["resolved_hyperparameters"][
                "independent_layer_initialization"
            ]
            is True
        )
    else:
        assert (
            result.metadata["resolved_hyperparameters"][
                "input_gate_xavier_initialization"
            ]
            is True
        )

    # Independent identities isolate contiguous episodes even when group values
    # repeat. Default models instead skip empty rows and retain asset history.
    reference = frame.copy()
    if grouped:
        segments = [0, 1, 2, 2, 2, 3, 4, 5, 6]
        reference["asset_id"] += reference.timestamp.dt.day.map(
            dict(enumerate(segments, 1))
        ).astype(str)
    expected = model.predict(reference)[frame.timestamp.ge("2020-01-04")]
    future = frame.loc[frame.timestamp.ge("2020-01-04")].reset_index(drop=True)
    np.testing.assert_allclose(
        model.predict(future), expected, rtol=1e-6, atol=1e-6, equal_nan=True
    )
    unordered = future.iloc[::-1].assign(
        timestamp=future.iloc[::-1].timestamp.astype(str)
    )
    np.testing.assert_allclose(
        model.predict(unordered), expected[::-1], rtol=1e-6, atol=1e-6, equal_nan=True
    )
    assert model.predict(future.iloc[:0]).shape == (0,)
    changed = future.copy()
    changed.loc[changed.timestamp.ge("2020-01-07"), "x"] += 100
    before = future.timestamp.lt("2020-01-07")
    np.testing.assert_allclose(
        model.predict(changed)[before], expected[before], rtol=1e-6, atol=1e-6
    )
    changed = future.copy()
    changed.loc[changed.asset_id.eq("b"), "x"] += 100
    other = future.asset_id.ne("b")
    np.testing.assert_allclose(
        model.predict(changed)[other], expected[other], rtol=1e-6, atol=1e-6
    )

    transformed_rows = []
    transform = type(model.scaler).transform

    def measured_transform(self, values, **kwargs):
        transformed_rows.append(len(values))
        return transform(self, values, **kwargs)

    with monkeypatch.context() as measured:
        measured.setattr(type(model.scaler), "transform", measured_transform)
        streamed = model.prediction_function()
        actual = np.concatenate(
            [streamed(future.iloc[i : i + 1]) for i in range(len(future))]
        )
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6, equal_nan=True)
    # Every new observed row is scaled once, plus the initial continuous tail.
    assert sum(transformed_rows) == int(future.x.notna().sum()) + (3 if grouped else 9)
    with pytest.raises(ValueError, match="sequence_prediction_time_not_increasing"):
        streamed(future.iloc[-1:])

    valid = future.x.notna() & future.eligible
    curve = result.metadata["learning_curve"]
    assert curve["sequence_lengths"]["max"] == (1 if grouped else 3)
    errors = pd.Series(
        (expected[valid] - future.loc[valid, "label"].to_numpy()) ** 2,
        index=future.loc[valid, "timestamp"],
    )
    assert curve["validation_loss_scope"] == "epoch_end_equal_timestamp_weighted"
    assert curve["epochs"][-1]["validation_loss"] == pytest.approx(
        errors.groupby(level=0).mean().mean(),
        rel=1e-6,
    )
    window = {"start": "2020-01-05", "end": "2020-01-09"}
    selected = valid & future.timestamp.ge(window["start"])
    predictions, _ = read_predictions(
        predict(
            {
                "model": str(fitted),
                "inputs": {"path": str(inputs), "spec": spec},
                "output": str(tmp_path / "pred"),
                "prediction_window": window,
            }
        )
    )
    pd.testing.assert_frame_equal(
        predictions[["timestamp", "asset_id"]],
        future.loc[selected, ["timestamp", "asset_id"]].reset_index(drop=True),
        check_dtype=False,
    )
    np.testing.assert_allclose(
        predictions.prediction_score, expected[selected], rtol=1e-6, atol=1e-6
    )
    _, tables = read_explanation(
        explain(
            {
                "model": str(fitted),
                "inputs": {
                    "path": str(inputs),
                    "spec": {**spec, "column_units": {"label": "decimal_return"}},
                },
                "output": str(tmp_path / "explanation"),
                "methods": ["permutation"],
                "window": window,
                "sample_role": "validation",
                "output_space": "model_output",
                "selection": {"max_rows": 100, "max_dates": None},
                "permutation": {
                    "groups": {"x": ["x"]},
                    "metrics": ["mse"],
                    "axis": "within_timestamp",
                    "cutoff": "2020-01-10",
                    "repeats": 1,
                },
            }
        )
    )
    scored = future.loc[selected].copy()
    scored["squared_error"] = (expected[selected] - scored.label) ** 2
    summary = tables["permutation_summary"]
    assert summary.baseline.iloc[0] == pytest.approx(
        scored.groupby("timestamp").squared_error.mean().mean(), rel=1e-6
    )
    assert summary.observation_count.iloc[0] == len(scored)


@pytest.mark.parametrize("algorithm", ["lstm_regressor", "transformer_regressor"])
@pytest.mark.parametrize("length", [1, 3])
def test_explicit_sequence_groups_and_historical_learning_curve(
    tmp_path, algorithm, length
):
    inputs = data(tmp_path)
    frame = pd.read_parquet(inputs)
    frame["episode"] = (frame.timestamp.dt.day - 1) // 2
    frame.to_parquet(inputs, index=False)
    config = fit_config(tmp_path, inputs)
    config["inputs"]["spec"]["column_roles"]["episode"] = "context"
    config["model"].update(
        model_algorithm=algorithm,
        hyperparameters={
            "device": "cpu",
            "hidden_size": 4,
            "epochs": 2,
            "sequence_length": length,
            "sequence_group_columns": ["episode"],
        },
    )
    config["validation"] = {
        "windows": [{"start": "2020-01-04", "end": "2020-01-06"}],
        "cutoff": "2020-01-06",
    }
    if algorithm == "transformer_regressor":
        config["candidate_parameters"] = config["model"]["hyperparameters"]
        config["model"]["hyperparameters"] = {}
    result, facts = load_model(fit(config))
    assert facts["validation"] == config["validation"]
    curve = result.metadata["learning_curve"]
    assert curve["sequence_lengths"]["max"] == min(length, 2)
    assert result.prediction_model.scaler.mean_ == pytest.approx((2.0,))
    validation = frame.loc[frame.timestamp.between("2020-01-04", "2020-01-06")]
    predicted = result.prediction_model.predict(validation)
    expected_mse = np.mean((predicted - validation.label.to_numpy()) ** 2)
    assert curve["epochs"][-1]["validation_loss"] == pytest.approx(
        expected_mse, rel=1e-6
    )
    changed = validation.copy()
    changed.loc[changed.episode.eq(1), "x"] += 1000
    np.testing.assert_allclose(
        result.prediction_model.predict(changed)[validation.episode.eq(2)],
        predicted[validation.episode.eq(2)],
        rtol=1e-6,
        atol=1e-6,
    )
    config["output"] = str(tmp_path / "overlap")
    config["validation"]["windows"][0]["start"] = "2020-01-03"
    with pytest.raises(ValueError, match="model_validation_training_overlap"):
        fit(config)


@pytest.mark.parametrize("algorithm", ["lstm_regressor", "transformer_regressor"])
@pytest.mark.parametrize("grouped", [False, True])
def test_disjoint_fit_preserves_observed_gaps_without_consuming_gap_labels(
    tmp_path, algorithm, grouped
):
    from quant_research_platform.contracts.models import _model_spec
    from quant_research_platform.research.models.production import fit_model

    inputs = data(tmp_path)
    frame = pd.read_parquet(inputs)
    frame["x"] += frame.timestamp.dt.day * 10
    frame["episode"] = frame.timestamp.dt.day.map(
        {1: "A", 2: "A", 3: "A", 4: "B", 5: "A", 6: "A", 7: "A", 8: "A", 9: "A"}
    )
    frame["label_available_at"] = frame.label_available_at.astype(str)
    gap = frame.timestamp.dt.day.between(3, 5)
    frame.loc[gap, "eligible"] = False
    frame.loc[gap, "label"] = np.inf
    frame.loc[gap, "label_available_at"] = "not-a-label-clock"
    late = frame.timestamp.dt.day.eq(6) & frame.asset_id.eq("c")
    frame.loc[late, "label_available_at"] = "2020-01-08"
    frame.loc[frame.timestamp.dt.day.gt(6), "x"] = np.inf
    frame.to_parquet(inputs, index=False, row_group_size=2)
    config = fit_config(tmp_path, inputs)
    config["inputs"]["spec"]["column_roles"]["episode"] = "context"
    config["fit_windows"] = [
        {"start": "2020-01-01", "end": "2020-01-02"},
        {"start": "2020-01-06", "end": "2020-01-06"},
    ]
    config["fit_cutoff"] = "2020-01-06"
    config["model"].update(
        model_algorithm=algorithm,
        hyperparameters={
            "device": "cpu",
            "epochs": 1,
            "hidden_size": 4,
            "sequence_length": 4,
            "sequence_group_columns": ["episode"] if grouped else [],
        },
    )
    result, facts = load_model(fit(config))
    assert facts["fit_windows"] == config["fit_windows"]
    assert result.metadata["rows_after_input_label_dropna"] == 8
    assert result.metadata["label_available_max"] == "2020-01-06T00:00:00"
    expected_mean = (
        frame.loc[frame.timestamp.dt.day.isin([1, 2, 6]) & ~late]
        .groupby("timestamp")
        .x.mean()
        .mean()
    )
    assert result.prediction_model.scaler.mean_ == pytest.approx((expected_mean,))
    history = result.prediction_model.history
    for asset in ("a", "b", "c"):
        assert history.loc[history.asset_id.eq(asset), "timestamp"].dt.day.tolist() == [
            4,
            5,
            6,
        ]
    lengths = result.metadata["learning_curve"]["sequence_lengths"]
    assert lengths["max"] == (2 if grouped else 4)
    assert lengths["mean"] == pytest.approx((13 if grouped else 17) / 8)

    direct = fit_model(
        (frame.iloc[start : start + 5] for start in range(0, len(frame), 5)),
        _model_spec(config["model"]),
        fit_windows=config["fit_windows"],
        fit_cutoff=config["fit_cutoff"],
    )
    for name, value in result.prediction_model.state.items():
        np.testing.assert_allclose(
            value.numpy(), direct.prediction_model.state[name].numpy()
        )


def test_sequence_context_gaps_keep_nanosecond_window_boundaries():
    from quant_research_platform.contracts.time import TimeWindow
    from quant_research_platform.research.models.production import _fit_read_windows

    windows = (
        TimeWindow("2020-01-01", "2020-01-01"),
        TimeWindow("2020-01-02T00:00:00.000000002", "2020-01-02T00:00:00.000000003"),
    )
    reads = tuple(_fit_read_windows(windows, sequence=True))
    assert reads[0] == (windows[0], False)
    assert reads[1] == (
        TimeWindow("2020-01-02T00:00:00", "2020-01-02T00:00:00.000000001"),
        True,
    )
    assert reads[2] == (windows[1], False)
    assert tuple(_fit_read_windows(windows, sequence=False)) == (
        (windows[0], False),
        (windows[1], False),
    )
