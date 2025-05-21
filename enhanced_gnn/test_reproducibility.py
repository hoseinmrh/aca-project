#!/usr/bin/env python3

"""
Test script to check if the model gives consistent results with fixed seeds.
Run this script multiple times to verify that the model produces the same output each time.
"""

import torch
import numpy as np
import time
import argparse
from set_seed import set_seed
from dataset import load_protein_dataset, get_dataloaders
from enhanced_gnn.enhanced_protein_gnn import EnhancedProteinGNN
from enhanced_train import test_model

def test_reproducibility(seed=42):
    # Set the seed 
    set_seed(seed)
    
    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load dataset with fixed seed
    print("Loading dataset...")
    dataset, train_dataset, val_dataset, test_dataset = load_protein_dataset(seed=seed)
    _, _, test_loader = get_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=32, seed=seed)
    
    # Initialize the model with fixed seed
    print("Initializing model...")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    model_params = {
        'num_nodes': dataset.num_node_features,
        'hidden_dim': 128,
        'num_classes': dataset.num_classes,
        'k': 10, 
        'readout': 'meanmax',
        'dropout': 0.3
    }
    model = EnhancedProteinGNN(**model_params).to(device)
    
    # Load pre-trained weights if available
    try:
        checkpoint = torch.load('optimal_protein_model.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded pre-trained model")
    except:
        print("No pre-trained model found, using initialized model")
    
    # Evaluate the model
    print("\nEvaluating model...")
    accuracy, precision, recall, f1 = test_model(model, test_loader, device)
    
    # Print metrics with high precision to see small differences
    print("\nMETRICS (8 decimal places):")
    print(f"Accuracy:  {accuracy:.8f}")
    print(f"Precision: {precision:.8f}")
    print(f"Recall:    {recall:.8f}")
    print(f"F1 Score:  {f1:.8f}")
    
    # Return metrics for comparison
    return accuracy, precision, recall, f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test reproducibility with fixed seeds')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--runs', type=int, default=3, help='Number of runs to test consistency')
    
    args = parser.parse_args()
    
    # Run multiple times and check consistency
    metrics_list = []
    for i in range(args.runs):
        print(f"\n=== Run {i+1}/{args.runs} ===")
        metrics = test_reproducibility(args.seed)
        metrics_list.append(metrics)
        
    # Check if all runs produced the same metrics
    consistent = all(m == metrics_list[0] for m in metrics_list)
    
    print("\n=== REPRODUCIBILITY TEST RESULTS ===")
    if consistent:
        print("SUCCESS: All runs produced identical results! Your model is reproducible.")
    else:
        print("FAILURE: Runs produced different results. Your model is NOT reproducible.")
        print("\nRun metrics (Accuracy, Precision, Recall, F1):")
        for i, metrics in enumerate(metrics_list):
            print(f"Run {i+1}: {metrics}")
