"""Sequence regressors with causal histories and CPU fallback when CUDA is absent."""

from collections import deque
from dataclasses import dataclass, replace
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


def _network(algorithm, width, parameters):
    if algorithm in DEEPLOB_ALGORITHMS:
        from quant_research_platform.research.models.deeplob import network

        return network(algorithm, parameters)
    import torch
    from torch import nn

    hidden = parameters["hidden_size"]

    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            if algorithm == "lstm_regressor":
                self.encoder = nn.LSTM(
                    width,
                    hidden,
                    parameters["num_layers"],
                    batch_first=True,
                    dropout=parameters["dropout"]
                    if parameters["num_layers"] > 1
                    else 0,
                )
            else:
                self.projection = nn.Linear(width, hidden)
                self.position = nn.Embedding(parameters["sequence_length"], hidden)
                layer = nn.TransformerEncoderLayer(
                    hidden,
                    parameters["n_heads"],
                    dim_feedforward=hidden * 2,
                    dropout=parameters["dropout"],
                    batch_first=True,
                )
                self.encoder = nn.TransformerEncoder(
                    layer, parameters["num_layers"], enable_nested_tensor=False
                )
            self.output = nn.Linear(hidden, 1)

        def forward(self, values, lengths):
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
            return self.output(final).squeeze(-1)

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
    matrix[valid] = scaler.transform(frame.loc[valid, columns]).to_numpy(
        dtype="float32"
    )
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
                self.algorithm, len(self.scaler.columns), self.parameters
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
            matrix = _matrix(self.scaler, work, columns, valid[endpoints])
            for asset, positions in assets.indices.items():
                if asset not in history:
                    first = endpoints[positions[0]]
                    seed = slice(starts[first], first)
                    initial = _matrix(
                        self.scaler, context.iloc[seed], columns, valid[seed]
                    )
                    history[asset] = deque(initial[valid[seed]], maxlen=length)
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
                history[asset].append(matrix[i].copy())
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


def _sequence_arrays(prepared, spec, scaler):
    deep = spec.model_algorithm in DEEPLOB_ALGORITHMS
    boundaries = sequence_context_columns(spec.model_algorithm, spec.hyperparameters)
    groups = boundaries
    context = _ordered_context(
        prepared.sequence_context, spec.input_columns, boundaries
    )
    valid = _observed_rows(context, spec.input_columns, deep)
    if not (groups or deep):
        context = context.loc[valid].reset_index(drop=True)
        valid = valid[valid]
    matrix = _matrix(scaler, context, spec.input_columns, valid)
    keys = ["asset_id", "timestamp"]
    endpoints = pd.MultiIndex.from_frame(context[keys]).get_indexer(
        pd.MultiIndex.from_frame(prepared.training[keys])
    )
    if (endpoints < 0).any():
        raise ValueError("sequence_training_context_missing")
    return context, matrix, endpoints, _sequence_starts(context, valid, groups, deep)


def fit_sequence(prepared, spec, validation=None):
    import torch
    from stable_baselines3.common.utils import get_device

    parameters = sequence_parameters(spec.model_algorithm, dict(spec.hyperparameters))
    device = get_device(parameters["device"])
    weights, scaler = prepared.preprocessing
    classification = spec.model_algorithm == "deeplob_classifier"
    context, matrix, endpoints, group_start = _sequence_arrays(prepared, spec, scaler)
    matrix = torch.from_numpy(matrix)
    target = prepared.training[spec.target_column].to_numpy(dtype="float64")
    mean = float(np.average(target, weights=weights))
    scale = float(np.sqrt(np.average((target - mean) ** 2, weights=weights))) or 1.0
    if classification:
        normalized = direction_classes(target, parameters["class_threshold"])
        mean, scale = 0.0, 1.0
    else:
        normalized = ((target - mean) / scale).astype("float32")
    validation_arrays = None
    if validation is not None:
        history = context.groupby("asset_id", sort=False).tail(
            parameters["sequence_length"] - 1
        )
        keys = ["asset_id", "timestamp"]
        # The context read can include the final training date. Keep its actual
        # observations once, without masking duplicates within either input.
        history = history.loc[
            ~pd.MultiIndex.from_frame(history[keys]).isin(
                pd.MultiIndex.from_frame(validation.sequence_context[keys])
            )
        ]
        validation = replace(
            validation,
            sequence_context=pd.concat(
                [history, validation.sequence_context],
                ignore_index=True,
            ),
        )
        _, v_matrix, v_endpoints, v_starts = _sequence_arrays(validation, spec, scaler)
        v_matrix = torch.from_numpy(v_matrix)
        v_target = validation.training[spec.target_column].to_numpy(dtype="float64")
        v_actual = (
            direction_classes(v_target, parameters["class_threshold"])
            if classification
            else ((v_target - mean) / scale).astype("float32")
        )
        validation_arrays = (v_matrix, v_endpoints, v_starts, v_actual)

    def batch_errors(network, values, ends, starts, actual, selected):
        ends = ends[selected]
        lengths = np.minimum(ends - starts[ends] + 1, parameters["sequence_length"])
        positions = ends[:, None] - lengths[:, None] + 1 + np.arange(int(lengths.max()))
        positions = np.minimum(positions, ends[:, None])
        predicted = network(
            torch.as_tensor(values[torch.as_tensor(positions)], device=device),
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
        network = _network(spec.model_algorithm, matrix.shape[1], parameters).to(device)
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
                "training_loss": float(loss_sum.cpu())
                / float(weights.sum())
                * scale**2,
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
                        validation_sum += errors.sum().to(torch.float64)
                point["validation_loss"] = (
                    float(validation_sum.cpu()) / len(v_actual) * scale**2
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
    history = context.groupby("asset_id", sort=False).tail(
        parameters["sequence_length"] - 1
    )
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
            "validation_loss_scope": "epoch_end_row_mean",
            "epochs": curve,
            "sequence_lengths": {
                "min": int(lengths.min()),
                "mean": float(lengths.mean()),
                "max": int(lengths.max()),
            },
        },
        stopping,
    )
