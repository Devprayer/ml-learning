import argparse
import os
from typing import List

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from mdgnn import MDGNN, generate_synthetic_stock_data


def directional_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    with torch.no_grad():
        sign_true = torch.sign(y_true)
        sign_pred = torch.sign(y_pred)
        # treat zeros as no direction; count as correct if both zero or both non-negative & non-positive match
        correct = (sign_true == sign_pred).float()
        return correct.mean().item()


def make_windows(T: int, window: int, train_ratio: float = 0.6, val_ratio: float = 0.2):
    # Produce index triplets (start, end, target_t) where end is inclusive and target is next step
    idxs = []
    for end in range(window - 1, T - 1):
        start = end - window + 1
        target_t = end + 1
        idxs.append((start, end, target_t))
    total = len(idxs)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    train = idxs[:n_train]
    val = idxs[n_train: n_train + n_val]
    test = idxs[n_train + n_val:]
    return train, val, test


def collate_window(batch):
    # Not used since we iterate windows directly
    raise NotImplementedError


def train_one_epoch(model: MDGNN, optimizer, dataset, window_idxs, device: torch.device, window: int):
    model.train()
    mse_loss = nn.MSELoss()
    losses = []
    N = dataset.num_nodes
    for (start, end, target_t) in tqdm(window_idxs, desc="Train", leave=False):
        x_seq = [dataset.features[t].to(device) for t in range(start, end + 1)]  # L of (N, F)
        edges_seq = [
            {k: (v[0].to(device), v[1].to(device)) for k, v in dataset.edges[t].items()
            } for t in range(start, end + 1)
        ]
        y_true = dataset.targets[target_t].to(device)  # (N,)

        optimizer.zero_grad()
        y_pred = model(x_seq, edges_seq)
        loss = mse_loss(y_pred, y_true)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        losses.append(loss.item())
    return sum(losses) / max(1, len(losses))


def evaluate(model: MDGNN, dataset, window_idxs, device: torch.device, window: int):
    model.eval()
    mse_loss = nn.MSELoss(reduction="mean")
    losses, dirs = [], []
    with torch.no_grad():
        for (start, end, target_t) in tqdm(window_idxs, desc="Eval", leave=False):
            x_seq = [dataset.features[t].to(device) for t in range(start, end + 1)]
            edges_seq = [
                {k: (v[0].to(device), v[1].to(device)) for k, v in dataset.edges[t].items()
                } for t in range(start, end + 1)
            ]
            y_true = dataset.targets[target_t].to(device)
            y_pred = model(x_seq, edges_seq)
            losses.append(mse_loss(y_pred, y_true).item())
            dirs.append(directional_accuracy(y_true, y_pred))
    return sum(losses) / max(1, len(losses)), sum(dirs) / max(1, len(dirs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-stocks", type=int, default=80)
    parser.add_argument("--timesteps", type=int, default=150)
    parser.add_argument("--feature-dim", type=int, default=16)
    parser.add_argument("--relations", type=int, default=3)
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    print("Generating synthetic data...")
    dataset = generate_synthetic_stock_data(
        num_stocks=args.num_stocks,
        timesteps=args.timesteps,
        feature_dim=args.feature_dim,
        num_relations=args.relations,
    )

    train_w, val_w, test_w = make_windows(dataset.num_timesteps, args.window)

    model = MDGNN(
        in_dim=dataset.feature_dim,
        hidden_dim=64,
        relation_names=dataset.relation_names,
        gnn_layers=2,
        gru_hidden_dim=64,
        mlp_hidden_dim=64,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val = float("inf")
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_mse = train_one_epoch(model, optimizer, dataset, train_w, device, args.window)
        val_mse, val_dir = evaluate(model, dataset, val_w, device, args.window)
        print(f"Epoch {epoch}: train MSE={train_mse:.4f} | val MSE={val_mse:.4f}, val DirAcc={val_dir:.3f}")
        if val_mse < best_val:
            best_val = val_mse
            torch.save({
                "model_state": model.state_dict(),
                "config": vars(args),
                "relation_names": dataset.relation_names,
            }, os.path.join("checkpoints", "mdgnn_best.pt"))

    test_mse, test_dir = evaluate(model, dataset, test_w, device, args.window)
    print(f"Test: MSE={test_mse:.4f}, DirAcc={test_dir:.3f}")
    torch.save(model.state_dict(), os.path.join("checkpoints", "mdgnn_last.pt"))


if __name__ == "__main__":
    main()