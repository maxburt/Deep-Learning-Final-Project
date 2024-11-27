from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        hidden_dim: int = 128,  # Number of units in the hidden layers
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
            hidden_dim (int): number of hidden units in the fully connected layers
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.hidden_dim = hidden_dim

        # Input size: 2 (track_left) + 2 (track_right) * n_track points
        input_dim = n_track * 4
        output_dim = n_waypoints * 2  # Each waypoint has 2D coordinates (x, y)

        # Define the MLP layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        # Concatenate track_left and track_right along the last dimension
        x = torch.cat((track_left, track_right), dim=2)  # (b, n_track, 4)

        # Flatten the input: (b, n_track * 4)
        x = x.view(x.size(0), -1)

        # Pass through the fully connected layers with ReLU activations
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # Output shape: (b, n_waypoints * 2)

        # Reshape to (b, n_waypoints, 2)
        return x.view(-1, self.n_waypoints, 2)


class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.d_model = d_model

        # Embedding for the waypoints (queries)
        self.query_embed = nn.Embedding(n_waypoints, d_model)

        # Linear layer to embed track points
        self.input_embed = nn.Linear(4, d_model)  # 2 (left) + 2 (right)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output layer to project decoder output to 2D waypoints
        self.output_layer = nn.Linear(d_model, 2)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        b, n_track, _ = track_left.size()

        # Concatenate track_left and track_right along the feature dimension
        track = torch.cat([track_left, track_right], dim=2)  # (b, n_track, 4)

        # Embed the input track points
        track_embedded = self.input_embed(track)  # (b, n_track, d_model)

        # Add positional encoding (optional, can improve performance)
        track_embedded = track_embedded.permute(1, 0, 2)  # (n_track, b, d_model)

        # Generate query embeddings for waypoints
        queries = self.query_embed.weight.unsqueeze(1).repeat(1, b, 1)  # (n_waypoints, b, d_model)

        # Apply the Transformer Decoder
        decoder_output = self.decoder(queries, track_embedded)  # (n_waypoints, b, d_model)

        # Project decoder output to 2D waypoints
        waypoints = self.output_layer(decoder_output)  # (n_waypoints, b, 2)

        # Permute back to (b, n_waypoints, 2)
        return waypoints.permute(1, 0, 2)

class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        # Normalization constants
        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

        # Convolutional layers (fewer filters, aggressive downsampling)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),  # Aggressive pooling

            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Initialize fully connected layers with dynamic size calculation
        self._initialize_fc_layers()

    def _initialize_fc_layers(self):
        # Create dummy input to determine feature map size after conv layers
        dummy_input = torch.zeros(1, 3, 96, 128)  # Example input size
        dummy_output = self.conv_layers(dummy_input)
        num_features = dummy_output.view(1, -1).size(1)

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(num_features, 128),  # Use calculated feature size
            nn.ReLU(),
            nn.Linear(128, self.n_waypoints * 2),
        )

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        # Normalize the image
        x = (image - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # Apply convolutional layers
        x = self.conv_layers(x)

        # Flatten the output
        x = x.view(x.size(0), -1)

        # Apply fully connected layers
        x = self.fc_layers(x)

        # Reshape to (b, n_waypoints, 2)
        return x.view(-1, self.n_waypoints, 2)

MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
