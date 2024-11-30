from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),  # Add Batch Normalization
            # nn.Dropout(0.6),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),  # Add Batch Normalization
            # nn.Dropout(0.6),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.model(x)
