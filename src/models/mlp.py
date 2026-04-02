"""MLP prediction head.

The MLP takes a fixed-size input vector and outputs a single scalar for
regression. Input dimension varies by model and embedding strategy:

    Strategy                    ESM-2   AbLang2
    DELTA_SEQUENCE              2560    960
    DELTA_RESIDUE               1280    480
    DELTA_RESIDUE_PLUS_WILD     3840    1440   (1280+2560 or 480+960)
    DELTA_RESIDUE_REDUCED       variable (depends on reduction)
    DELTA_RESIDUE_POOLED        TBD

Pass input_dim explicitly. The first hidden layer IS the projection layer.
There is no separate input projection.
"""

from typing import List

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Multi-layer perceptron with ReLU activations and dropout.

    Parameters
    ----------
    input_dim:
        Dimension of the input embedding vector. Must match the output of
        the chosen embedding strategy and model.
    hidden_dims:
        List of hidden layer sizes, e.g. [256, 128]. The first entry is the
        projection from input_dim. An empty list gives a single linear layer.
    dropout:
        Dropout probability applied after each hidden layer (not after the
        output layer).
    output_dim:
        Output dimension. Default 1 for scalar regression.

    Example
    -------
    # ESM-2 delta sequence, two hidden layers
    mlp = MLP(input_dim=2560, hidden_dims=[256, 128], dropout=0.1)

    # AbLang2 delta residue, one hidden layer
    mlp = MLP(input_dim=480, hidden_dims=[256], dropout=0.1)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout: float,
        output_dim: int = 1,
    ):
        super().__init__()

        layers = []
        in_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x: (batch_size, input_dim) float tensor

        Returns
        -------
        (batch_size, output_dim) float tensor
        """
        return self.net(x)
