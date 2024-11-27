"""
Usage:
    python3 -m homework.train_planner --model <model_name>
"""

import argparse
import torch
from torch.utils.data import DataLoader
from torch import nn, optim

from homework.models import MLPPlanner, TransformerPlanner, CNNPlanner, save_model
from homework.datasets.road_dataset import load_data
from homework.metrics import PlannerMetric


def parse_args():
    parser = argparse.ArgumentParser(description="Train a planner model")
    parser.add_argument(
        "--model", type=str, default="mlp_planner",
        choices=["mlp_planner", "transformer_planner", "cnn_planner"],
        help="Model to train (mlp_planner, transformer_planner, or cnn_planner)"
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Number of epochs for training"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for training and validation"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate for the optimizer"
    )
    parser.add_argument(
        "--train_path", type=str, default="drive_data/train",
        help="Path to the training dataset"
    )
    parser.add_argument(
        "--val_path", type=str, default="drive_data/val",
        help="Path to the validation dataset"
    )
    return parser.parse_args()


def train():
    args = parse_args()

    # Set up device (GPU, MPS, or CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load training and validation data
    train_loader = load_data(dataset_path=args.train_path, batch_size=args.batch_size, shuffle=True)
    val_loader = load_data(dataset_path=args.val_path, batch_size=args.batch_size, shuffle=False)

    # Initialize the model based on the argument
    if args.model == "mlp_planner":
        model = MLPPlanner(n_track=10, n_waypoints=3, hidden_dim=128).to(device)
    elif args.model == "transformer_planner":
        model = TransformerPlanner(
            n_track=10,
            n_waypoints=3,
            d_model=64,
            nhead=4,
            num_layers=2,
        ).to(device)
    elif args.model == "cnn_planner":
        model = CNNPlanner(n_waypoints=3).to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # Define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Initialize metric for evaluation
    metric = PlannerMetric()

    print("Starting training...")

    best_l1_error = float("inf")  # Track the best model
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")

        # Training phase
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            # Move batch data to the device
            waypoints = batch["waypoints"].to(device)

            if args.model == "cnn_planner":
                # For CNNPlanner, use image input
                image = batch["image"].to(device)
                predictions = model(image)
            else:
                # For MLP and TransformerPlanner, use track boundaries
                track_left = batch["track_left"].to(device)
                track_right = batch["track_right"].to(device)
                predictions = model(track_left, track_right)

            # Compute loss
            loss = loss_fn(predictions, waypoints)
            train_loss += loss.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader)
        print(f"Train Loss: {train_loss:.4f}")

        # Validation phase
        model.eval()
        metric.reset()
        with torch.no_grad():
            for batch in val_loader:
                # Move batch data to the device
                waypoints = batch["waypoints"].to(device)
                waypoints_mask = batch["waypoints_mask"].to(device)

                if args.model == "cnn_planner":
                    # For CNNPlanner, use image input
                    image = batch["image"].to(device)
                    predictions = model(image)
                else:
                    # For MLP and TransformerPlanner, use track boundaries
                    track_left = batch["track_left"].to(device)
                    track_right = batch["track_right"].to(device)
                    predictions = model(track_left, track_right)

                # Add results to metric
                metric.add(predictions, waypoints, waypoints_mask)

        # Compute validation metrics
        results = metric.compute()
        print(
            f"Validation - L1 Error: {results['l1_error']:.4f}, "
            f"Longitudinal Error: {results['longitudinal_error']:.4f}, "
            f"Lateral Error: {results['lateral_error']:.4f}"
        )

        # Save the best model
        if results["l1_error"] < best_l1_error:
            best_l1_error = results["l1_error"]
            best_model_path = save_model(model)
            print(f"New best model saved to {best_model_path}")

    print("Training complete.")


if __name__ == "__main__":
    train()