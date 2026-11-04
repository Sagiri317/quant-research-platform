"""Sequence regressors with causal histories and CPU fallback when CUDA is absent."""

from collections import deque
from dataclasses import dataclass

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
    context = frame.loc[
        :, list(dict.fromkeys(["asset_id", "timestamp", *columns, *context_columns]))
    ].copy()
    if context.asset_id.isna().any() or context.asset_id.map(type).ne(str).any():
        raise ValueError("sequence_asset_id_invalid")
    context["timestamp"] = pd.to_datetime(context.timestamp, errors="raise")
    if (
        context.timestamp.isna().any()
        or context.duplicated(["asset_id", "timestamp"]).any()
    ):
        raise ValueError("sequence_keys_invalid")
    return context.sort_values(["asset_id", "timestamp"], kind="stable").reset_index(
        drop=True
    )


def _matrix(scaler, frame, columns, deep):
    valid = (
        complete_book_rows(frame) if deep else ~frame.loc[:, columns].isna().all(axis=1)
    )
    matrix = np.zeros((len(frame), len(scaler.columns)), dtype="float32")
    matrix[valid] = scaler.transform(frame.loc[valid, columns]).to_numpy(
        dtype="float32"
    )
    return matrix, valid.to_numpy()


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
        columns = self.scaler.input_columns or self.scaler.columns
        deep = self.algorithm in DEEPLOB_ALGORITHMS
        classification = self.algorithm == "deeplob_classifier"
        length = self.parameters["sequence_length"]
        history = {}
        seed = self.history.groupby("asset_id", sort=False)
        seeds = {asset: frame for asset, frame in seed}
        last_times = {}
        segments = {}

        def predict(frame):
            result = np.full((len(frame), 3) if probabilities else len(frame), np.nan)
            work = frame.reset_index(drop=True)
            _ordered_context(work, columns, sequence_context_columns(self.algorithm))
            matrix, valid = _matrix(self.scaler, work, columns, deep)
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
                        torch.as_tensor(sizes, device=device),
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

            for i in work.sort_values(["timestamp", "asset_id"], kind="stable").index:
                asset, timestamp = work.loc[i, ["asset_id", "timestamp"]]
                timestamp = pd.Timestamp(timestamp)
                if asset not in history:
                    previous = seeds.get(asset)
                    initial = []
                    if previous is not None:
                        previous = previous.loc[previous.timestamp.lt(timestamp)]
                        if deep:
                            previous = previous.loc[
                                previous.session_date.eq(work.at[i, "session_date"])
                                & previous.segment_id.eq(work.at[i, "segment_id"])
                            ].reset_index(drop=True)
                            invalid = ~complete_book_rows(previous)
                            if invalid.any():
                                previous = previous.iloc[
                                    np.flatnonzero(invalid)[-1] + 1 :
                                ]
                        previous = (
                            previous.tail(length - 1)
                            if length > 1
                            else previous.iloc[:0]
                        )
                        initial = list(_matrix(self.scaler, previous, columns, deep)[0])
                    history[asset] = deque(initial, maxlen=length)
                if asset in last_times and timestamp <= last_times[asset]:
                    raise ValueError("sequence_prediction_time_not_increasing")
                last_times[asset] = timestamp
                if deep:
                    segment = (work.at[i, "session_date"], work.at[i, "segment_id"])
                    if asset in segments and segment != segments[asset]:
                        history[asset].clear()
                    segments[asset] = segment
                if not valid[i]:
                    if deep:
                        history[asset].clear()
                    continue
                history[asset].append(matrix[i])
                batches.append(np.stack(history[asset]))
                indices.append(i)
                if len(batches) == self.parameters["batch_size"]:
                    flush()
            flush()
            return result

        return predict

    def predict(self, frame):
        return self.prediction_function()(frame)


def fit_sequence(prepared, spec):
    import torch
    from stable_baselines3.common.utils import get_device

    parameters = sequence_parameters(spec.model_algorithm, dict(spec.hyperparameters))
    device = get_device(parameters["device"])
    weights, scaler, _ = prepared.preprocessing
    deep = spec.model_algorithm in DEEPLOB_ALGORITHMS
    classification = spec.model_algorithm == "deeplob_classifier"
    context = _ordered_context(
        prepared.sequence_context,
        spec.input_columns,
        sequence_context_columns(spec.model_algorithm),
    )
    matrix, valid = _matrix(scaler, context, spec.input_columns, deep)
    keys = pd.MultiIndex.from_frame(context[["asset_id", "timestamp"]])
    endpoints = keys.get_indexer(
        pd.MultiIndex.from_frame(prepared.training[["asset_id", "timestamp"]])
    )
    if (endpoints < 0).any():
        raise ValueError("sequence_training_context_missing")
    starts = context.groupby("asset_id", sort=False).indices
    group_start = np.empty(len(context), dtype="int64")
    for positions in starts.values():
        group_start[positions] = positions[0]
    if deep:
        boundary = (
            context[["asset_id", "session_date", "segment_id"]]
            .ne(context[["asset_id", "session_date", "segment_id"]].shift())
            .any(axis=1)
            .to_numpy()
        )
        boundary = boundary | ~valid | ~np.r_[False, valid[:-1]]
        group_start = np.maximum.accumulate(
            np.where(boundary, np.arange(len(context)), 0)
        )
    target = prepared.training[spec.target_column].to_numpy(dtype="float64")
    mean = float(np.average(target, weights=weights))
    scale = float(np.sqrt(np.average((target - mean) ** 2, weights=weights))) or 1.0
    if classification:
        normalized = direction_classes(target, parameters["class_threshold"])
        mean, scale = 0.0, 1.0
    else:
        normalized = ((target - mean) / scale).astype("float32")
    rng = np.random.default_rng(spec.random_seed)
    with torch.random.fork_rng():
        torch.manual_seed(spec.random_seed)
        network = _network(spec.model_algorithm, matrix.shape[1], parameters).to(device)
        optimizer = torch.optim.Adam(
            network.parameters(), lr=parameters["learning_rate"]
        )
        network.train()
        for _ in range(parameters["epochs"]):
            order = rng.permutation(len(target))
            for offset in range(0, len(order), parameters["batch_size"]):
                selected = order[offset : offset + parameters["batch_size"]]
                ends = endpoints[selected]
                lengths = np.minimum(
                    ends - group_start[ends] + 1, parameters["sequence_length"]
                )
                positions = (
                    ends[:, None] - lengths[:, None] + 1 + np.arange(int(lengths.max()))
                )
                positions = np.minimum(positions, ends[:, None])
                values = torch.as_tensor(matrix[positions], device=device)
                predicted = network(values, torch.as_tensor(lengths, device=device))
                actual = torch.as_tensor(normalized[selected], device=device)
                mass = torch.as_tensor(
                    weights[selected], dtype=torch.float32, device=device
                )
                errors = (
                    torch.nn.functional.cross_entropy(
                        predicted, actual.long(), reduction="none"
                    )
                    if classification
                    else (predicted - actual).square()
                )
                loss = (errors * mass).mean()
                if not torch.isfinite(loss):
                    raise ValueError("model_training_loss_nonfinite")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        state = {
            name: value.detach().cpu() for name, value in network.state_dict().items()
        }
    if not all(torch.isfinite(value).all() for value in state.values()):
        raise ValueError("model_state_nonfinite")
    history = context.groupby("asset_id", sort=False).tail(
        parameters["sequence_length"] - 1
    )
    return SequencePredictor(
        spec.model_algorithm, parameters, scaler, state, history, mean, scale
    )
