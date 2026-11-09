"""DeepLOB spatial CNN, multiscale Inception, and recurrent book encoder.

Temporal convolutions use past-only padding, making short histories independent
of batch padding. Scaling and labels follow the declared QRP model recipe.
"""

from quant_research_platform.contracts.models import DEEPLOB_INPUT_COLUMNS


def complete_book_rows(frame):
    """Require the full book, independently of research-universe membership."""
    return frame.loc[:, DEEPLOB_INPUT_COLUMNS].notna().all(axis=1)


def network(algorithm, parameters):
    import torch
    from torch import nn

    channels = parameters["conv_channels"]
    inception = parameters["inception_channels"]

    def convolution(in_channels, out_channels, kernel, stride=(1, 1)):
        return nn.Sequential(
            nn.ZeroPad2d((0, 0, kernel[0] - 1, 0)),
            nn.Conv2d(in_channels, out_channels, kernel, stride=stride),
            nn.LeakyReLU(0.01),
        )

    class DeepLOB(nn.Module):
        def __init__(self):
            super().__init__()
            self.spatial = nn.Sequential(
                convolution(1, channels, (1, 2), (1, 2)),
                convolution(channels, channels, (4, 1)),
                convolution(channels, channels, (4, 1)),
                convolution(channels, channels, (1, 2), (1, 2)),
                convolution(channels, channels, (4, 1)),
                convolution(channels, channels, (4, 1)),
                convolution(channels, channels, (1, 10)),
                convolution(channels, channels, (4, 1)),
                convolution(channels, channels, (4, 1)),
            )
            self.branches = nn.ModuleList(
                [
                    nn.Sequential(
                        convolution(channels, inception, (1, 1)),
                        convolution(inception, inception, (3, 1)),
                    ),
                    nn.Sequential(
                        convolution(channels, inception, (1, 1)),
                        convolution(inception, inception, (5, 1)),
                    ),
                    nn.Sequential(
                        nn.ConstantPad2d((0, 0, 2, 0), float("-inf")),
                        nn.MaxPool2d((3, 1), stride=1),
                        convolution(channels, inception, (1, 1)),
                    ),
                ]
            )
            self.dropout = nn.Dropout(parameters["dropout"])
            self.recurrent = nn.LSTM(
                3 * inception,
                parameters["hidden_size"],
                parameters["num_layers"],
                batch_first=True,
                dropout=parameters["dropout"] if parameters["num_layers"] > 1 else 0.0,
            )
            self.output = nn.Linear(
                parameters["hidden_size"], 3 if algorithm == "deeplob_classifier" else 1
            )

        def forward(self, values, lengths):
            encoded = self.spatial(values[:, None])
            encoded = (
                torch.cat([branch(encoded) for branch in self.branches], dim=1)
                .squeeze(-1)
                .transpose(1, 2)
            )
            packed = nn.utils.rnn.pack_padded_sequence(
                self.dropout(encoded),
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, (hidden, _) = self.recurrent(packed)
            result = self.output(hidden[-1])
            return result if algorithm == "deeplob_classifier" else result.squeeze(-1)

    return DeepLOB()
