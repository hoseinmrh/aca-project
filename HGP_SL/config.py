# protein_gnn/config.py
import argparse

def get_config():
    parser = argparse.ArgumentParser(description='Protein Graph Classification with HGP-SL')
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--nhid', type=int, default=128)
    parser.add_argument('--sample_neighbor', action='store_true')
    parser.add_argument('--sparse_attention', action='store_true')
    parser.add_argument('--structure_learning', action='store_true')
    parser.add_argument('--pooling_ratio', type=float, default=0.5)
    parser.add_argument('--dropout_ratio', type=float, default=0.0)
    parser.add_argument('--lamb', type=float, default=1.0)
    parser.add_argument('--dataset', type=str, default='PROTEINS')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=100)
    return parser.parse_args()
