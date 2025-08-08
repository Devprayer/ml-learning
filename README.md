## MDGNN: Multi-Relational Dynamic GNN for Stock Movement Prediction

- Generates synthetic multi-relational dynamic graph data for a universe of stocks
- Trains an MDGNN (R-GCN over time + GRU) to predict next-step returns and direction

### Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train.py --epochs 5 --num-stocks 80 --timesteps 150 --device cpu
```

Arguments:
- `--num-stocks`: number of synthetic stocks
- `--timesteps`: number of time steps
- `--feature-dim`: node feature dimensionality
- `--relations`: number of relation types (>=2)
- `--window`: history window size (L)
- `--epochs`: training epochs
- `--device`: `cpu` or `cuda` (if available)

Outputs:
- Prints train/val/test metrics (MSE and directional accuracy)
- Saves last checkpoint to `checkpoints/mdgnn_last.pt`
