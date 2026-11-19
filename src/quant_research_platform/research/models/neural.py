"""Sequence regressors with causal histories and CPU fallback when CUDA is absent."""

from collections import deque
from dataclasses import dataclass
from itertools import count

import numpy as np
import pandas as pd

from quant_research_platform.compute.classification import direction_classes
from quant_research_platform.contracts.models import (
    DEEPLOB_ALGORITHMS,
    sequence_context_columns,
    sequence_parameters,
)
from quant_research_platform.research.models.deeplob import complete_book_rows
from quant_research_platform.research.models.scaler import (
    equal_timestamp_sample_weights,
)


def _network(algorithm, width, parameters, *, columns=()):
    if algorithm in DEEPLOB_ALGORITHMS:
        from quant_research_platform.research.models.deeplob import network

        return network(algorithm, parameters)
    import torch
    from torch import nn

    hidden = parameters["hidden_size"]
    snapshots = set(parameters.get("snapshot_columns", ()))
    static = [i for i, column in enumerate(columns) if column in snapshots]
    dynamic = [i for i in range(width) if i not in static]

    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            if not dynamic:
                self.encoder = None
            elif algorithm == "lstm_regressor":
                self.encoder = nn.LSTM(
                    len(dynamic),
                    hidden,
                    parameters["num_layers"],
                    batch_first=True,
                    dropout=parameters["dropout"]
                    if parameters["num_layers"] > 1
                    else 0,
                )
            else:
                self.projection = nn.Linear(len(dynamic), hidden)
                self.position = nn.Embedding(parameters["sequence_length"], hidden)
                layer_parameters = dict(
                    d_model=hidden,
                    nhead=parameters["n_heads"],
                    dim_feedforward=hidden * 2,
                    dropout=parameters["dropout"],
                    batch_first=True,
                    norm_first=parameters.get("norm_first", False),
                )
                self.encoder = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(**layer_parameters),
                    parameters["num_layers"],
                    norm=nn.LayerNorm(hidden, eps=1e-5)
                    if layer_parameters["norm_first"]
                    else None,
                    enable_nested_tensor=False,
                )
                if parameters.get("independent_layer_initialization", False):
                    for index in range(1, parameters["num_layers"]):
                        self.encoder.layers[index] = nn.TransformerEncoderLayer(
                            **layer_parameters
                        )
            self.dropout = nn.Dropout(parameters.get("head_dropout", 0.0))
            self.output = nn.Linear((hidden if dynamic else 0) + len(static), 1)
            if (
                dynamic
                and algorithm == "lstm_regressor"
                and parameters.get("input_gate_xavier_initialization", False)
            ):
                for gate in self.encoder.weight_ih_l0.chunk(4, dim=0):
                    nn.init.xavier_uniform_(gate)

        def forward(self, values, lengths):
            if static:
                rows = torch.arange(len(values), device=values.device)
                snapshot = values[rows, lengths.to(values.device) - 1][:, static]
                if not dynamic:
                    return self.output(self.dropout(snapshot)).squeeze(-1)
                values = values[:, :, dynamic]
            if algorithm == "lstm_regressor":
                packed = nn.utils.rnn.pack_padded_sequence(
                    values, lengths.cpu(), batch_first=True, enforce_sorted=False
                )
                _, (hidden_state, _) = self.encoder(packed)
                final = hidden_state[-1]
            else:
                lengths = lengths.to(values.device)
                positions = torch.arange(values.shape[1], device=values.device)
                encoded = self.encoder(
                    self.projection(values) + self.position(positions),
                    src_key_padding_mask=positions[None, :] >= lengths[:, None],
                )
                final = encoded[
                    torch.arange(len(values), device=values.device), lengths - 1
                ]
            if static:
                final = torch.cat((final, snapshot), dim=1)
            return self.output(self.dropout(final)).squeeze(-1)

    return Network()


def _ordered_context(frame, columns, context_columns=()):
    """Order histories while preserving input row positions for prediction output."""
    context = frame.loc[
        :, list(dict.fromkeys(["asset_id", "timestamp", *columns, *context_columns]))
    ].reset_index(drop=True)
    if any(type(asset) is not str for asset in context.asset_id):
        raise ValueError("sequence_asset_id_invalid")
    context["timestamp"] = pd.to_datetime(context.timestamp, errors="raise")
    if (
        context_columns
        and (
            context.loc[:, context_columns].isna().any(axis=1)
            & context.loc[:, columns].notna().any(axis=1)
        ).any()
    ):
        raise ValueError("sequence_context_missing")
    if (
        context.timestamp.isna().any()
        or context.duplicated(["asset_id", "timestamp"]).any()
    ):
        raise ValueError("sequence_keys_invalid")
    return context.sort_values(["asset_id", "timestamp"], kind="stable")


def _observed_rows(frame, columns, deep):
    return (
        complete_book_rows(frame) if deep else ~frame.loc[:, columns].isna().all(axis=1)
    ).to_numpy()


def _matrix(scaler, frame, columns, valid):
    matrix = np.zeros((len(frame), len(scaler.columns)), dtype="float32")
    for start in range(0, len(frame), 16384):
        block = frame.iloc[start : start + 16384]
        selected = valid[start : start + len(block)]
        matrix[start : start + len(block)][selected] = scaler.transform(
            block.loc[selected, columns], dtype="float32"
        ).to_numpy(copy=False)
    return matrix


@dataclass
class SequencePredictor:
    algorithm: str
    parameters: dict
    scaler: object
    state: dict
    history: pd.DataFrame
    target_mean: float
    target_scale: float

    def prediction_function(self, *, probabilities=False):
        import torch
        from stable_baselines3.common.utils import get_device

        device = get_device(self.parameters["device"])
        # Loading weights must not consume the caller's random stream.
        with torch.random.fork_rng():
            network = _network(
                self.algorithm,
                len(self.scaler.columns),
                self.parameters,
                columns=self.scaler.columns,
            )
        network.load_state_dict(self.state)
        network.to(device).eval()
        columns = self.scaler.columns
        deep = self.algorithm in DEEPLOB_ALGORITHMS
        boundaries = sequence_context_columns(self.algorithm, self.parameters)
        groups = boundaries
        classification = self.algorithm == "deeplob_classifier"
        length = self.parameters["sequence_length"]
        history = {}
        previous = {
            asset: block
            for asset, block in self.history.groupby("asset_id", sort=False)
        }

        def predict(frame):
            result = np.full((len(frame), 3) if probabilities else len(frame), np.nan)
            work = _ordered_context(frame, columns, boundaries)
            if work.empty:
                return result
            parts, endpoints = [], []
            offset = 0
            assets = work.groupby("asset_id", sort=False)
            for asset, block in assets:
                first = block.timestamp.iloc[0]
                prior = previous.get(asset)
                if prior is None:
                    prior = work.iloc[:0]
                elif asset in history:
                    if first <= prior.timestamp.iloc[-1]:
                        raise ValueError("sequence_prediction_time_not_increasing")
                else:
                    prior = prior.loc[prior.timestamp.lt(first)]
                previous[asset] = block.tail(1).copy()
                # Each ordered asset history precedes its new rows.
                parts.extend((prior, block))
                offset += len(prior)
                endpoints.extend(range(offset, offset + len(block)))
                offset += len(block)
            context = pd.concat(parts, ignore_index=True)
            valid = _observed_rows(context, columns, deep)
            starts = _sequence_starts(context, valid, groups, deep)
            endpoints = np.asarray(endpoints)
            observed = np.zeros(len(context), dtype=bool)
            observed[endpoints] = valid[endpoints]
            seeds = {}
            for asset, positions in assets.indices.items():
                if asset not in history:
                    first = endpoints[positions[0]]
                    seeds[asset] = slice(starts[first], first)
                    observed[seeds[asset]] = valid[seeds[asset]]
            matrix = _matrix(self.scaler, context, columns, observed)
            for asset, seed in seeds.items():
                history[asset] = deque(matrix[seed][valid[seed]], maxlen=length)
            batches, indices = [], []

            def flush():
                if not batches:
                    return
                sizes = np.array([len(sequence) for sequence in batches])
                values = np.zeros(
                    (len(batches), int(sizes.max()), matrix.shape[1]), dtype="float32"
                )
                for i, sequence in enumerate(batches):
                    values[i, : len(sequence)] = sequence
                with torch.inference_mode():
                    output = network(
                        torch.as_tensor(values, device=device),
                        torch.as_tensor(sizes),
                    )
                    if classification:
                        output = output.softmax(dim=1)
                    output = output.cpu().numpy().astype("float64")
                if classification:
                    result[indices] = (
                        output if probabilities else output[:, 2] - output[:, 0]
                    )
                else:
                    result[indices] = output * self.target_scale + self.target_mean
                batches.clear()
                indices.clear()

            for i, asset in enumerate(work.asset_id):
                endpoint = endpoints[i]
                if starts[endpoint] == endpoint:
                    history[asset].clear()
                if not valid[endpoint]:
                    continue
                history[asset].append(matrix[endpoint].copy())
                batches.append(np.stack(history[asset]))
                indices.append(work.index[i])
                if len(batches) == self.parameters["batch_size"]:
                    flush()
            flush()
            return result

        return predict

    def predict(self, frame):
        return self.prediction_function()(frame)


def _sequence_starts(context, valid, groups, deep):
    """Start a new contiguous history on an asset/group change or required gap."""
    keys = context[["asset_id", *groups]]
    boundary = keys.ne(keys.shift()).any(axis=1).to_numpy()
    if groups or deep:
        boundary = boundary | ~valid | ~np.r_[False, valid[:-1]]
    return np.maximum.accumulate(np.where(boundary, np.arange(len(context)), 0))


def _sequence_arrays(prepared, spec, scaler, history=None):
    deep = spec.model_algorithm in DEEPLOB_ALGORITHMS
    boundaries = sequence_context_columns(spec.model_algorithm, spec.hyperparameters)
    groups = boundaries
    prefix = 0 if history is None else len(history)
    metadata = prepared.sequence_context
    observed = prepared.context_valid
    if prefix:
        metadata = pd.concat(
            [history.loc[:, metadata.columns], metadata], ignore_index=True
        )
        observed = np.r_[_observed_rows(history, spec.input_columns, deep), observed]
    context = _ordered_context(metadata, (), boundaries)
    valid = observed[context.index]
    if boundaries and (context.loc[:, boundaries].isna().any(axis=1) & valid).any():
        raise ValueError("sequence_context_missing")
    if not (groups or deep):
        context = context.loc[valid]
        valid = valid[valid]
    matrix = np.zeros((len(context), len(spec.input_columns)), dtype="float32")
    # Read in source order so Arrow take does not repeatedly combine distant chunks.
    source_order = np.argsort(context.index.to_numpy())
    for start in range(0, len(context), 16384):
        positions = source_order[start : start + 16384]
        positions = positions[valid[positions]]
        rows = context.index[positions].to_numpy()
        values = (
            prepared.feature_rows(rows - prefix)
            if not prefix
            else pd.concat(
                [
                    history.loc[:, spec.input_columns].iloc[rows[rows < prefix]],
                    prepared.feature_rows(rows[rows >= prefix] - prefix),
                ],
                ignore_index=True,
            )
        )
        matrix[positions] = scaler._transform_numeric(
            values.to_numpy(dtype="float64", copy=True), dtype="float32"
        )
    keys = ["asset_id", "timestamp"]
    endpoints = pd.MultiIndex.from_frame(context[keys]).get_indexer(
        pd.MultiIndex.from_frame(prepared.training[keys])
    )
    if (endpoints < 0).any():
        raise ValueError("sequence_training_context_missing")
    return context, matrix, endpoints, _sequence_starts(context, valid, groups, deep)


def _prepare_sequence_inputs(prepared, spec, validation=None):
    parameters = sequence_parameters(spec.model_algorithm, dict(spec.hyperparameters))
    scaler = prepared.preprocessing[1]
    context, matrix, endpoints, group_start = _sequence_arrays(prepared, spec, scaler)
    history_rows = context.groupby("asset_id", sort=False).tail(
        parameters["sequence_length"] - 1
    )
    history = pd.concat(
        [
            history_rows.loc[
                :, ~history_rows.columns.isin(spec.input_columns)
            ].reset_index(drop=True),
            pd.DataFrame(
                prepared.feature_rows(history_rows.index.to_numpy()).to_numpy(
                    dtype="float64", copy=True
                ),
                columns=spec.input_columns,
            ),
        ],
        axis=1,
    )
    validation_arrays = None
    if validation is not None:
        keys = ["asset_id", "timestamp"]
        # The context read can include the final training date. Keep its actual
        # observations once, without masking duplicates within either input.
        validation_history = history.loc[
            ~pd.MultiIndex.from_frame(history[keys]).isin(
                pd.MultiIndex.from_frame(validation.sequence_context[keys])
            )
        ]
        _, v_matrix, v_endpoints, v_starts = _sequence_arrays(
            validation, spec, scaler, validation_history
        )
        validation_arrays = v_matrix, v_endpoints, v_starts
    return matrix, endpoints, group_start, history, validation_arrays


def fit_sequence(prepared, spec, validation=None, *, fit_inputs=None):
    import torch
    from stable_baselines3.common.utils import get_device

    parameters = sequence_parameters(spec.model_algorithm, dict(spec.hyperparameters))
    device = get_device(parameters["device"])
    input_device = device if parameters["preload_inputs"] else "cpu"
    weights, scaler = prepared.preprocessing
    weight_sum = float(weights.sum())
    classification = spec.model_algorithm == "deeplob_classifier"
    matrix, endpoints, group_start, history, validation_arrays = (
        _prepare_sequence_inputs(prepared, spec, validation)
        if fit_inputs is None
        else fit_inputs
    )
    matrix = torch.as_tensor(matrix, device=input_device)
    history = history.groupby("asset_id", sort=False).tail(
        parameters["sequence_length"] - 1
    )
    target = prepared.training[spec.target_column].to_numpy(dtype="float64")
    mean = float(np.average(target, weights=weights))
    scale = float(np.sqrt(np.average((target - mean) ** 2, weights=weights))) or 1.0
    if classification:
        normalized = direction_classes(target, parameters["class_threshold"])
        mean, scale = 0.0, 1.0
    else:
        normalized = ((target - mean) / scale).astype("float32")
    if validation_arrays is not None:
        v_matrix, v_endpoints, v_starts = validation_arrays
        v_matrix = torch.as_tensor(v_matrix, device=input_device)
        v_target = validation.training[spec.target_column].to_numpy(dtype="float64")
        v_actual = (
            direction_classes(v_target, parameters["class_threshold"])
            if classification
            else ((v_target - mean) / scale).astype("float32")
        )
        validation_arrays = (v_matrix, v_endpoints, v_starts, v_actual)
        validation_weights = equal_timestamp_sample_weights(
            validation.training.timestamp
        )
        validation_mass = torch.as_tensor(
            validation_weights, dtype=torch.float64, device=device
        )
        validation_weight_sum = float(validation_weights.sum())

    def batch_errors(network, values, ends, starts, actual, selected):
        ends = ends[selected]
        lengths = np.minimum(ends - starts[ends] + 1, parameters["sequence_length"])
        positions = ends[:, None] - lengths[:, None] + 1 + np.arange(int(lengths.max()))
        positions = np.minimum(positions, ends[:, None])
        predicted = network(
            torch.as_tensor(
                values[torch.as_tensor(positions, device=values.device)], device=device
            ),
            torch.as_tensor(lengths),
        )
        actual = torch.as_tensor(actual[selected], device=device)
        return (
            torch.nn.functional.cross_entropy(
                predicted, actual.long(), reduction="none"
            )
            if classification
            else (predicted - actual).square()
        )

    curve = []
    patience = parameters.get("early_stopping_rounds")
    best_loss, best_iteration, best_state = np.inf, 0, None
    rng = np.random.default_rng(spec.random_seed)
    with torch.random.fork_rng():
        torch.manual_seed(spec.random_seed)
        network = _network(
            spec.model_algorithm, matrix.shape[1], parameters, columns=scaler.columns
        ).to(device)
        optimizer = torch.optim.AdamW(
            network.parameters(),
            lr=parameters["learning_rate"],
            weight_decay=parameters["weight_decay"],
        )
        epochs = (
            count() if parameters["epochs"] is None else range(parameters["epochs"])
        )
        for epoch in epochs:
            network.train()
            order = rng.permutation(len(target))
            loss_sum = torch.zeros((), dtype=torch.float64, device=device)
            for offset in range(0, len(order), parameters["batch_size"]):
                selected = order[offset : offset + parameters["batch_size"]]
                mass = torch.as_tensor(
                    weights[selected], dtype=torch.float32, device=device
                )
                errors = batch_errors(
                    network, matrix, endpoints, group_start, normalized, selected
                )
                loss = (errors * mass).mean()
                if not torch.isfinite(loss):
                    raise ValueError("model_training_loss_nonfinite")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                loss_sum += (errors.detach() * mass).sum().to(torch.float64)
            point = {
                "epoch": epoch + 1,
                "training_loss": float(loss_sum.cpu()) / weight_sum * scale**2,
                "validation_loss": None,
            }
            if validation_arrays is not None:
                network.eval()
                validation_sum = torch.zeros((), dtype=torch.float64, device=device)
                with torch.inference_mode():
                    for offset in range(0, len(v_actual), parameters["batch_size"]):
                        selected = np.arange(
                            offset,
                            min(offset + parameters["batch_size"], len(v_actual)),
                        )
                        errors = batch_errors(network, *validation_arrays, selected)
                        validation_sum += (
                            errors * validation_mass[offset : offset + len(selected)]
                        ).sum()
                point["validation_loss"] = (
                    float(validation_sum.cpu()) / validation_weight_sum * scale**2
                )
            if point["validation_loss"] is not None and not np.isfinite(
                point["validation_loss"]
            ):
                raise ValueError("model_validation_loss_nonfinite")
            curve.append(point)
            if patience is not None:
                if point["validation_loss"] < best_loss:
                    best_loss, best_iteration = point["validation_loss"], epoch
                    best_state = {
                        name: value.detach().cpu().clone()
                        for name, value in network.state_dict().items()
                    }
                elif epoch - best_iteration >= patience:
                    break
        if best_state is not None:
            network.load_state_dict(best_state)
        state = {
            name: value.detach().cpu() for name, value in network.state_dict().items()
        }
    if not all(torch.isfinite(value).all() for value in state.values()):
        raise ValueError("model_state_nonfinite")
    predictor = SequencePredictor(
        spec.model_algorithm, parameters, scaler, state, history, mean, scale
    )
    lengths = np.minimum(
        endpoints - group_start[endpoints] + 1, parameters["sequence_length"]
    )
    stopping = (
        {"best_iteration": best_iteration, "iterations": len(curve)}
        if patience is not None
        else None
    )
    return (
        predictor,
        {
            "metric": "log_loss" if classification else "mse",
            "training_loss_scope": "online_epoch_weighted",
            "validation_loss_scope": "epoch_end_equal_timestamp_weighted",
            "epochs": curve,
            "sequence_lengths": {
                "min": int(lengths.min()),
                "mean": float(lengths.mean()),
                "max": int(lengths.max()),
            },
        },
        stopping,
    )
