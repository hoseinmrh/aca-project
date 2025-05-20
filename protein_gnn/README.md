# Protein-GNN: Protein Classification with Hierarchical Graph Pooling and Structure Learning

This project is a clean, modular re-implementation of the HGP-SL (Hierarchical Graph Pooling with Structure Learning) model for protein graph classification, based on the original code in the `HGP-SL` folder.

## Structure

- `protein_gnn/`
  - `__init__.py`
  - `layers.py`         # Custom GNN layers and pooling
  - `models.py`         # Model definition
  - `sparse_softmax.py` # Sparsemax activation
  - `train.py`          # Training and evaluation logic
  - `config.py`         # Configurations and argument parsing
- `data/`               # Datasets (symlink or copy from original)
- `main.py`             # Entry point
- `requirements.txt`    # Dependencies

## Usage

```bash
python main.py --dataset PROTEINS
```

## Notes
- The code is organized for clarity and extensibility.
- Dataset structure and requirements are the same as the original HGP-SL.
- See `requirements.txt` for dependencies (PyTorch, torch-geometric, etc).

## Reference
Original HGP-SL: https://arxiv.org/abs/1911.05954
