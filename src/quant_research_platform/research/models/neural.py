"""CUDA sequence regressors with causal, per-asset observation histories."""

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_research_platform.contracts.models import sequence_parameters


def _network(algorithm, width, parameters):
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


def _cuda(parameters):
    import torch

    device = torch.device(parameters["device"])
    if not torch.cuda.is_available() or (
        device.index is not None and device.index >= torch.cuda.device_count()
    ):
        raise RuntimeError("model_cuda_unavailable")
    return device


def _ordered_context(frame, columns):
    context = frame.loc[:, ["asset_id", "timestamp", *columns]].copy()
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


@dataclass
class SequencePredictor:
    algorithm: str
    parameters: dict
    scaler: object
    state: dict
    history: pd.DataFrame
    target_mean: float
    target_scale: float

    def prediction_function(self):
        import torch

        device = _cuda(self.parameters)
        # Loading weights must not consume the caller's random stream.
        with torch.random.fork_rng():
            network = _network(
                self.algorithm, len(self.scaler.columns), self.parameters
            )
        network.load_state_dict(self.state)
        network.to(device).eval()
        columns = self.scaler.input_columns
        length = self.parameters["sequence_length"]
        history = {}
        seed = self.history.groupby("asset_id", sort=False)
        seeds = {asset: frame for asset, frame in seed}
        last_times = {}

        def predict(frame):
            result = np.full(len(frame), np.nan)
            work = frame.reset_index(drop=True)
            _ordered_context(work, columns)
            matrix = self.scaler.transform(work.loc[:, columns]).to_numpy(
                dtype="float32"
            )
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
                    output = (
                        network(
                            torch.as_tensor(values, device=device),
                            torch.as_tensor(sizes, device=device),
                        )
                        .cpu()
                        .numpy()
                    )
                result[indices] = (
                    output.astype("float64") * self.target_scale + self.target_mean
                )
                batches.clear()
                indices.clear()

            for i in work.sort_values(["timestamp", "asset_id"], kind="stable").index:
                asset, timestamp = work.loc[i, ["asset_id", "timestamp"]]
                timestamp = pd.Timestamp(timestamp)
                if asset not in history:
                    previous = seeds.get(asset)
                    initial = []
                    if previous is not None:
                        previous = previous.loc[previous.timestamp.lt(timestamp)].tail(
                            length - 1
                        )
                        initial = list(
                            self.scaler.transform(previous.loc[:, columns]).to_numpy(
                                dtype="float32"
                            )
                        )
                    history[asset] = deque(initial, maxlen=length)
                if asset in last_times and timestamp <= last_times[asset]:
                    raise ValueError("sequence_prediction_time_not_increasing")
                last_times[asset] = timestamp
                if work.loc[i, list(columns)].isna().all():
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

    parameters = sequence_parameters(spec.model_algorithm, dict(spec.hyperparameters))
    device = _cuda(parameters)
    weights, scaler, _ = prepared.preprocessing
    context = _ordered_context(prepared.sequence_context, spec.input_columns)
    matrix = scaler.transform(context.loc[:, spec.input_columns]).to_numpy(
        dtype="float32"
    )
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
    target = prepared.training[spec.target_column].to_numpy(dtype="float64")
    mean = float(np.average(target, weights=weights))
    scale = float(np.sqrt(np.average((target - mean) ** 2, weights=weights))) or 1.0
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
                loss = ((predicted - actual).square() * mass).mean()
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
