import os
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from config import get_config
from models import ProteinGNN
from train import train, evaluate
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, accuracy_score
import numpy as np

def main():
    import json
    import time
    args = get_config()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    dataset = TUDataset(os.path.join('HGP-SL', 'data', args.dataset), name=args.dataset, use_node_attr=True)
    args.num_classes = dataset.num_classes
    args.num_features = dataset.num_features
    num_training = int(len(dataset) * 0.8)
    num_val = int(len(dataset) * 0.1)
    num_test = len(dataset) - (num_training + num_val)
    training_set, validation_set, test_set = torch.utils.data.random_split(dataset, [num_training, num_val, num_test])
    train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    model = ProteinGNN(args).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_time = time.time()
    total_epochs, total_time, best_val_acc = train(model, optimizer, train_loader, val_loader, args)
    train_time_per_epoch = total_time / (total_epochs)
    # Test
    test_acc, test_loss = evaluate(model, test_loader, args)
    print(f'Test set results, loss = {test_loss:.6f}, accuracy = {test_acc:.6f}')
    # Peak memory usage (GPU)
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated(args.device) / 1024**2
    else:
        peak_memory = 0.0
    # Model parameter count
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Model size in MB
    param_size_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size_bytes + buffer_size) / 1024**2
    model.eval()
    inference_times = []
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(args.device)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t0 = time.time()
            out = model(data)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t1 = time.time()
            batch_size = data.y.shape[0]  # number of graphs in this batch
            inference_times.append((t1 - t0) / batch_size * 1000)  # ms per graph
            pred = out.argmax(dim=1).detach().cpu().numpy()
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
        'NumEpochs': total_epochs,
        'TrainingTimeTotal': total_time,
        'TrainingTimePerEpoch': train_time_per_epoch,
        'PeakMemoryMB': peak_memory,
        'NumberOfParameters': num_params,
        'ModelSizeMB': model_size_mb,
        'InferenceLatencyMS': inference_latency_ms

    }
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    print('Test metrics saved to metrics.json')

if __name__ == '__main__':
    main()
