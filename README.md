# Graph Neural Network for Protein Classification

A comprehensive comparison of different Graph Neural Network (GNN) architectures for protein classification using the PROTEINS dataset.

## Overview

This project implements and compares various GNN models for protein classification tasks. The models are evaluated on the TU Dataset PROTEINS, which contains protein structures represented as graphs where nodes are amino acids and edges represent spatial or chemical relationships.

## Project Structure

```
├── enhanced_gnn/           # Enhanced GNN implementation
│   ├── enhanced_protein_gnn.py  # Advanced protein-specific GNN
│   ├── enhanced_train.py   # Training utilities
│   └── main.py             # Main training script
├── graph_transformer/      # Graph Transformer implementation
│   ├── model.py            # Transformer-based GNN
│   └── main.py             # Training script
├── HGP_SL/                # Hierarchical Graph Pooling with Structure Learning
│   ├── models.py           # HGP-SL model implementation
│   └── main.py             # Training script
└── results/               # Evaluation results and metrics
```

## Models Implemented


### 1. Enhanced GNN (`enhanced_gnn/`)
- Protein-specific architecture combining ChebConv and EdgeConv
- Residual connections and batch normalization
- PairNorm for handling oversmoothing
- Optimized for protein classification tasks

### 2. Graph Transformer (`graph_transformer/`)
- Transformer-based architecture for graphs
- Multi-head attention mechanisms
- Global pooling for graph-level predictions

### 3. HGP-SL (`HGP_SL/`)
- Hierarchical Graph Pooling with Structure Learning
- Advanced pooling strategies
- Structure learning capabilities

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Main dependencies:
- PyTorch
- PyTorch Geometric
- scikit-learn
- pandas
- numpy
- torch_scatter

## Usage

### Training Individual Models


1. **Enhanced GNN**:
```bash
cd enhanced_gnn
python main.py
```

2. **Graph Transformer**:
```bash
cd graph_transformer
python main.py
```

3. **HGP-SL**:
```bash
cd HGP_SL
python main.py
```



### Dataset

The project uses the PROTEINS dataset from TU Dataset collection:
- **Graphs**: 1113 protein structures
- **Classes**: 2 (enzymes vs non-enzymes)
- **Node Features**: 3 (amino acid properties)


Data is automatically downloaded and split into train/validation/test sets (80%/10%/10%).

## Evaluation Metrics

All models are evaluated using:
- **Accuracy** & **Balanced Accuracy**
- **Precision**, **Recall**, **F1-Score**
- **Training Time** & **Inference Latency**
- **Memory Usage** & **Model Size**
- **Number of Parameters**


Detailed metrics are saved in `metrics.json` files within each model directory.

After runing each model and having the `metrics.json` file within each model directory, you can run the following command to get the score of each model, based on the scroing function we have defined in our report.
```bash
cd results
python calculate_score.py
```



## Authors

- Hosein Mirhoseini
- Ali Nabipour Dargah
