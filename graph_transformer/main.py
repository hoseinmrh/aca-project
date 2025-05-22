import torch
import numpy as np
import pandas as pd
import time
import json
from dataset import load_protein_dataset, get_dataloaders
from model import ProteinGraphTransformer
from train import train
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, accuracy_score

def main():
    seed = 42
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset, train_dataset, val_dataset, test_dataset = load_protein_dataset(seed=seed)
    train_loader, val_loader, test_loader = get_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=32, seed=seed)
    in_channels = dataset.num_node_features
    out_channels = dataset.num_classes
    hidden_channels = 128
    model = ProteinGraphTransformer(in_channels, hidden_channels, out_channels, num_layers=4, heads=8, dropout=0.1).to(device)
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters.")
    param_size_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size_bytes + buffer_size) / 1024**2
    torch.cuda.reset_peak_memory_stats(device) if device.type == 'cuda' else None
    start_time = time.time()
    model, num_epochs = train(model, train_loader, val_loader, device, max_epochs=500, patience=50)
    total_time = time.time() - start_time
    peak_memory = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == 'cuda' else 0
    # Inference latency
    model.eval()
    inference_times = []
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            t0 = time.time()
            out = model(data.x, data.edge_index, data.batch)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            t1 = time.time()
            batch_size = data.y.shape[0]
            inference_times.append((t1 - t0) / batch_size * 1000)  # ms per graph
            pred = out.argmax(dim=1).cpu().numpy()
            all_preds.append(pred)
            all_labels.append(data.y.cpu().numpy())
    inference_latency_ms = float(np.mean(inference_times)) if inference_times else 0.0
    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    balanced_accuracy = balanced_accuracy_score(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    metrics = {
        'BalancedAccuracy': balanced_accuracy,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'NumEpochs': num_epochs,
        'TrainingTimeTotal': total_time,
        'TrainingTimePerEpoch': total_time / num_epochs,
        'PeakMemoryMB': peak_memory,
        'NumberOfParameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'ModelSizeMB': model_size_mb,
        'InferenceLatencyMS': inference_latency_ms
    }
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    print('Test metrics saved to metrics.json')
    print(metrics)

if __name__ == '__main__':
    main()
