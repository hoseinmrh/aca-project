import torch
import time
import os
import json
from dataset import load_protein_dataset, get_dataloaders
from enhanced_protein_gnn import EnhancedProteinGNN
from enhanced_train import enhanced_train, test_model
from set_seed import set_seed

def train_optimal_protein_model(save_model=True, plot_results=True, seed=42):
    """
    Train the optimal protein classification model (EnhancedProtein_Medium) and evaluate it
    
    Args:
        save_model: Whether to save the trained model
        plot_results: Whether to plot the results
        seed: Random seed for reproducibility
        
    Returns:
        The trained model and evaluation metrics
    """
    # Set random seed for reproducibility
    set_seed(seed)
    
    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    dataset, train_dataset, val_dataset, test_dataset = load_protein_dataset(seed=seed)
    train_loader, val_loader, test_loader = get_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=32, seed=seed)
    
    # Initialize the best model (EnhancedProtein_Medium)
    model_params = {
        'num_nodes': dataset.num_node_features,
        'hidden_dim': 128, 
        'num_classes': dataset.num_classes,
        'k': 5, 
        'readout': 'meanmax', 
        'dropout': 0.3
    }
    model = EnhancedProteinGNN(**model_params).to(device)
    
    # Print model information
    param_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_size:,}")
    
    # Get model memory size in MB
    param_size_bytes = 0
    for param in model.parameters():
        param_size_bytes += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_mb = (param_size_bytes + buffer_size) / 1024**2
    print(f"Model size: {size_mb:.2f} MB")
    
    # Train the model with progress tracking
    print("\nStarting training...")
    start_time = time.time()
    trained_model, number_of_epochs = enhanced_train(model, train_loader, val_loader, device, patience=30, max_epochs=300)
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds")
    
    # Evaluate the model on test set
    print("\nEvaluating model on test set...")
    balanced_accuracy, precision, recall, f1, accuracy = test_model(trained_model, test_loader, device)


    inference_times = []
    trained_model.eval()
    with torch.no_grad():
        for data in test_loader:
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            batch = data.batch.to(device)
            
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start = time.time()
            _ = trained_model(x, edge_index, batch)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            end = time.time()
            
            # Calculate time per graph
            batch_size = batch[-1].item() + 1
            inference_times.append((end - start) / batch_size)
    
    avg_inference_time = sum(inference_times) / len(inference_times)
    print(f"Average inference time: {avg_inference_time*1000:.4f} ms per graph")
    
    # Collect all metrics in a single dictionary
    all_metrics = {
        # Performance metrics
        'BalancedAccuracy': balanced_accuracy,
        'Accuracy': accuracy,    
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        # Model size metrics
        'NumEpochs': number_of_epochs,
        'TrainingTimeTotal': total_time,
        'TrainingTimePerEpoch': total_time / (number_of_epochs if number_of_epochs > 0 else 1),  # Estimate based on early stopping
        # Memory usage metrics
        'PeakMemoryMB': torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == 'cuda' else 0,
        'NumberOfParameters': param_size,
        'ModelSizeMB': size_mb,
        
        
        # Time metrics
        'InferenceLatencyMS': avg_inference_time * 1000,  # Convert to milliseconds
    }
    
    # Save metrics to JSON
    save_metrics_to_json(all_metrics, "EnhancedProtein_Medium", "metrics.json")
    
    # Save the model
    if save_model:
        save_path = 'optimal_protein_model.pt'
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'model_params': model_params,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'inference_time_ms': avg_inference_time*1000,
            'all_metrics': all_metrics
        }, save_path)
        print(f"\nModel saved to {save_path}")
    
    # Return the model and metrics
    return trained_model, all_metrics

    """
    Load the optimal trained model
    
    Args:
        model_path: Path to the saved model
    
    Returns:
        The loaded model
    """
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found.")
        return None
    
    # Load model data
    checkpoint = torch.load(model_path)
    model_params = checkpoint['model_params']
    
    # Initialize the model
    model = EnhancedProteinGNN(**model_params)
    model.load_state_dict(checkpoint['model_state_dict'])

    
    return model

def save_metrics_to_json(metrics, model_name, file_path="model_metrics.json"):
    """
    Save all model metrics to a JSON file with values rounded to 3 decimal places
    
    Args:
        metrics: Dictionary of metrics to save
        model_name: Name of the model
        file_path: Path to save the JSON file
    """
    # Round all numeric values to 3 decimal places
    rounded_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            if key == "Parameters":  # Don't round parameter count
                rounded_metrics[key] = value
            else:
                rounded_metrics[key] = round(value, 3)
        else:
            rounded_metrics[key] = value
    
    

    
    # Save to file
    with open(file_path, 'w') as f:
        json.dump(rounded_metrics, f, indent=4)
    
    print(f"Metrics saved to {file_path}")

if __name__ == "__main__":

    # Train a new model
    train_optimal_protein_model(save_model=False, plot_results=False, seed=42)

