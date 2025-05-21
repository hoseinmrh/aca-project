import torch
import pandas as pd
import time
import matplotlib.pyplot as plt
import os
import json
from dataset import load_protein_dataset, get_dataloaders
from enhanced_protein_gnn import EnhancedProteinGNN
from enhanced_train import enhanced_train, test_model
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
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
    print("Initializing EnhancedProtein model...")
    model_params = {
        'num_nodes': dataset.num_node_features,
        'hidden_dim': 128,  # Medium dimension
        'num_classes': dataset.num_classes,
        'k': 5, 
        'readout': 'meanmax',  # Enhanced readout
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
    trained_model = enhanced_train(model, train_loader, val_loader, device, patience=30, max_epochs=300)
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds")
    
    # Evaluate the model on test set
    print("\nEvaluating model on test set...")
    accuracy, precision, recall, f1 = test_model(trained_model, test_loader, device)
    
    # Print metrics
    print("\n" + "="*50)
    print("FINAL MODEL PERFORMANCE:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("="*50)
    
    # Generate detailed classification report
    print("\nDetailed Classification Report:")
    y_pred = []
    y_true = []
    trained_model.eval()
    with torch.no_grad():
        for data in test_loader:
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            batch = data.batch.to(device)
            
            out = trained_model(x, edge_index, batch)
            pred = out.argmax(dim=1).cpu().numpy()
            y_pred.extend(pred)
            y_true.extend(data.y.cpu().numpy())
    
    # Print classification report
    target_names = ['Non-Enzyme', 'Enzyme']
    print(classification_report(y_true, y_pred, target_names=target_names))
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if plot_results:
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=target_names, yticklabels=target_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300)
        print("Confusion matrix saved to 'confusion_matrix.png'")
    
    # Calculate inference latency
    print("\nMeasuring inference latency...")
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
        'Accuracy': accuracy,    
        # Model size metrics
        'Parameters': param_size,
        'ModelSizeMB': size_mb,
        
        # Memory usage metrics
        'PeakMemoryMB': torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == 'cuda' else 0,
        
        # Time metrics
        'InferenceLatencyMS': avg_inference_time * 1000,  # Convert to milliseconds
        'TrainingTimeTotal': total_time,
        'TrainingTimePerEpoch': total_time / (30 if total_time < 100 else 10),  # Estimate based on early stopping
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

def load_optimal_model(model_path='optimal_protein_model.pt'):
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
    
    print(f"Loaded model with metrics:")
    print(f"Accuracy:  {checkpoint['accuracy']:.4f}")
    print(f"F1 Score:  {checkpoint['f1']:.4f}")
    print(f"Inference: {checkpoint['inference_time_ms']:.4f} ms per graph")
    
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
    
    # Check if file exists to update or create
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            try:
                all_metrics = json.load(f)
            except json.JSONDecodeError:
                all_metrics = []
    else:
        all_metrics = []
    
    # Check if this model is already in the metrics
    updated = False
    for i, m in enumerate(all_metrics):
        if m.get('Model') == model_name:
            all_metrics[i] = rounded_metrics
            updated = True
            break
    
    if not updated:
        all_metrics.append(rounded_metrics)
    
    # Save to file
    with open(file_path, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    
    print(f"Metrics saved to {file_path}")

def predict_protein(model, protein_data, device=None):
    """
    Make a prediction for a single protein graph
    
    Args:
        model: The trained model
        protein_data: A PyG data object representing the protein
        device: The device to use for inference
    
    Returns:
        The predicted class (0=Non-Enzyme, 1=Enzyme) and confidence
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        x = protein_data.x.to(device)
        edge_index = protein_data.edge_index.to(device)
        
        # Create a batch with just one graph
        batch = torch.zeros(x.size(0), dtype=torch.long).to(device)
        
        # Get prediction
        out = model(x, edge_index, batch)
        probabilities = torch.nn.functional.softmax(out, dim=1)
        
        # Get the predicted class and confidence
        pred_class = out.argmax(dim=1).item()
        confidence = probabilities[0, pred_class].item()
        
    class_name = "Enzyme" if pred_class == 1 else "Non-Enzyme"
    return pred_class, class_name, confidence

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train or use the optimal protein classification model')
    parser.add_argument('--train', action='store_true', help='Train a new model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the saved model')
    parser.add_argument('--model_path', type=str, default='optimal_protein_model.pt', 
                        help='Path to save/load the model')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducible results')
    
    args = parser.parse_args()
    
    if args.train:
        # Train a new model
        train_optimal_protein_model(save_model=True, plot_results=True, seed=args.seed)
    elif args.evaluate:
        # Load and evaluate the model
        set_seed(args.seed)  # Set seed for evaluation too
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = load_optimal_model(args.model_path)
        
        if model is not None:
            # Load test data
            dataset, _, _, test_dataset = load_protein_dataset()
            # Fix: Create test_loader properly without requiring train/val datasets
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=32,
                shuffle=False
            )
            
            # Evaluate
            model = model.to(device)
            start_time = time.time()
            accuracy, precision, recall, f1 = test_model(model, test_loader, device)
            eval_time = time.time() - start_time
            
            print("\nEvaluation on test set:")
            print(f"Accuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1 Score:  {f1:.4f}")
            
            # Measure inference latency
            print("\nMeasuring inference latency...")
            inference_times = []
            model.eval()
            with torch.no_grad():
                for data in test_loader:
                    x = data.x.to(device)
                    edge_index = data.edge_index.to(device)
                    batch = data.batch.to(device)
                    
                    torch.cuda.synchronize() if device.type == 'cuda' else None
                    start = time.time()
                    _ = model(x, edge_index, batch)
                    torch.cuda.synchronize() if device.type == 'cuda' else None
                    end = time.time()
                    
                    # Calculate time per graph
                    batch_size = batch[-1].item() + 1
                    inference_times.append((end - start) / batch_size)
            
            avg_inference_time = sum(inference_times) / len(inference_times)
            print(f"Average inference time: {avg_inference_time*1000:.4f} ms per graph")
            
            # Get model parameter count
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Get model size in MB
            param_size_bytes = 0
            for param in model.parameters():
                param_size_bytes += param.nelement() * param.element_size()
            buffer_size = 0
            for buffer in model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
            size_mb = (param_size_bytes + buffer_size) / 1024**2
            
            # Collect memory usage - this will only be peak during evaluation, not training
            peak_memory = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == 'cuda' else 0
            
            # Load the checkpoint to get training metrics if available
            try:
                checkpoint = torch.load(args.model_path)
                training_time = checkpoint.get('all_metrics', {}).get('TrainingTimeTotal', 0)
            except:
                training_time = 0
            
            # Collect all metrics
            eval_metrics = {
                # Performance metrics
                'Accuracy': accuracy,
                # Model size metrics
                'Parameters': param_count,
                'ModelSizeMB': size_mb,
                
                # Memory usage metrics
                'PeakMemoryMB': peak_memory,
                
                # Time metrics
                'InferenceLatencyMS': avg_inference_time * 1000,  # Convert to milliseconds
                'EvaluationTime': eval_time,
                'TrainingTimeTotal': training_time
            }
            
            # Save metrics to JSON
            save_metrics_to_json(eval_metrics, "EnhancedProtein_Medium_Eval", "evaluation_metrics.json")
    else:
        print("Please specify either --train or --evaluate")
        print("Example usage:")
        print("  python optimal_protein_model.py --train")
        print("  python optimal_protein_model.py --evaluate")
