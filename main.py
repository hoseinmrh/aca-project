import torch
import pandas as pd
import time
import tracemalloc
from dataset import load_protein_dataset, get_dataloaders
from base import Abstract_GNN, LabelSmoothingCrossEntropy
from gcn import GCN
from gin import GIN
from gat import GAT
from chebnet import ChebC, ChebEdge
from train import train, test_model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_peak_memory(device):
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        return lambda: torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # in MB
    else:
        tracemalloc.start()
        return lambda: tracemalloc.get_traced_memory()[1] / (1024 ** 2)  # in MB

def measure_inference_latency(model, loader, device, n_batches=5):
    model.eval()
    times = []
    with torch.no_grad():
        for i, data in enumerate(loader):
            if i >= n_batches:
                break
            batch = data.batch.to(device)
            x = data.x.to(device)
            z = data.edge_index.to(device)
            start = time.time()
            _ = model(x, z, batch)
            end = time.time()
            times.append((end - start) / x.shape[0])  # time per sample
    return sum(times) / len(times) if times else None

def train_with_epoch_time(model, train_loader, val_loader, device):
    import copy
    from train import train, validate_model
    model = model.to(device)
    loss_function = None
    optimizer = None
    lr_scheduler = None
    best_metric = -1
    best_metric_epoch = -1
    best_val_loss = 1000
    best_model = None
    epochs = 500
    early_stop = 30
    es_counter = 0
    epoch_times = []
    for epoch in range(epochs):
        import time
        start = time.time()
        model.train()
        epoch_train_loss = 0
        for i, data in enumerate(train_loader):
            batch = data.batch.to(device)
            x = data.x.to(device)
            y = data.y.to(device)
            z = data.edge_index.to(device)
            if optimizer is None:
                loss_function = LabelSmoothingCrossEntropy()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=400, eta_min=1e-5)
            optimizer.zero_grad()
            out = model(x, z, batch)
            step_loss = loss_function(out, y)
            step_loss.backward(retain_graph=True)
            optimizer.step()
            epoch_train_loss += step_loss.item()
        epoch_train_loss = epoch_train_loss / (i + 1)
        lr_scheduler.step()
        val_loss, val_acc = validate_model(model, val_loader, device)
        end = time.time()
        epoch_times.append(end - start)
        if val_loss < best_val_loss:
            best_metric = val_acc
            best_val_loss = val_loss
            best_metric_epoch = epoch + 1
            best_model = copy.deepcopy(model)
            es_counter = 0
        else:
            es_counter += 1
        if es_counter > early_stop:
            break
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    return best_model, avg_epoch_time, len(epoch_times)

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    dataset, train_dataset, val_dataset, test_dataset = load_protein_dataset()
    train_loader, val_loader, test_loader = get_dataloaders(train_dataset, val_dataset, test_dataset)

    metrics = []

    # GCN
    GCN_ = GCN(dataset.num_node_features, 32, dataset.num_classes, readout='meanmax')
    model_size = count_parameters(GCN_)
    get_peak_memory = measure_peak_memory(device)
    gcnModel, avg_epoch_time, n_epochs = train_with_epoch_time(GCN_, train_loader, val_loader, device)
    peak_memory = get_peak_memory()
    inf_latency = measure_inference_latency(gcnModel, test_loader, device)
    gcn_accuracy, gcn_precision, gcn_recall, gcn_f1 = test_model(gcnModel, test_loader, device)
    metrics.append({
        'Model': 'GCN',
        'Parameters': model_size,
        'PeakMemoryMB': peak_memory,
        'InferenceLatency': inf_latency,
        'TrainTimePerEpochSec': avg_epoch_time,
        'Epochs': n_epochs,
        'Accuracy': gcn_accuracy,
        'Precision': gcn_precision,
        'Recall': gcn_recall,
        'F1': gcn_f1
    })

    # GIN
    GIN_ = GIN(dataset.num_node_features, 32, dataset.num_classes, readout='mean')
    model_size = count_parameters(GIN_)
    get_peak_memory = measure_peak_memory(device)
    ginModel, avg_epoch_time, n_epochs = train_with_epoch_time(GIN_, train_loader, val_loader, device)
    peak_memory = get_peak_memory()
    inf_latency = measure_inference_latency(ginModel, test_loader, device)
    gin_accuracy, gin_precision, gin_recall, gin_f1 = test_model(ginModel, test_loader, device)
    metrics.append({
        'Model': 'GIN',
        'Parameters': model_size,
        'PeakMemoryMB': peak_memory,
        'InferenceLatency': inf_latency,
        'TrainTimePerEpochSec': avg_epoch_time,
        'Epochs': n_epochs,
        'Accuracy': gin_accuracy,
        'Precision': gin_precision,
        'Recall': gin_recall,
        'F1': gin_f1
    })

    # GAT
    GAT_ = GAT(dataset.num_node_features, 32, dataset.num_classes, readout='sum', concat=True, num_heads=8)
    model_size = count_parameters(GAT_)
    get_peak_memory = measure_peak_memory(device)
    gatModel, avg_epoch_time, n_epochs = train_with_epoch_time(GAT_, train_loader, val_loader, device)
    peak_memory = get_peak_memory()
    inf_latency = measure_inference_latency(gatModel, test_loader, device)
    gat_accuracy, gat_precision, gat_recall, gat_f1 = test_model(gatModel, test_loader, device)
    metrics.append({
        'Model': 'GAT',
        'Parameters': model_size,
        'PeakMemoryMB': peak_memory,
        'InferenceLatency': inf_latency,
        'TrainTimePerEpochSec': avg_epoch_time,
        'Epochs': n_epochs,
        'Accuracy': gat_accuracy,
        'Precision': gat_precision,
        'Recall': gat_recall,
        'F1': gat_f1
    })

    # ChebNet
    chebFilterSize = 16
    ChebC_ = ChebC(dataset.num_node_features, 32, dataset.num_classes, chebFilterSize, readout='meanmax')
    model_size = count_parameters(ChebC_)
    get_peak_memory = measure_peak_memory(device)
    chebModel, avg_epoch_time, n_epochs = train_with_epoch_time(ChebC_, train_loader, val_loader, device)
    peak_memory = get_peak_memory()
    inf_latency = measure_inference_latency(chebModel, test_loader, device)
    cheb_accuracy, cheb_precision, cheb_recall, cheb_f1 = test_model(chebModel, test_loader, device)
    metrics.append({
        'Model': 'ChebNet',
        'Parameters': model_size,
        'PeakMemoryMB': peak_memory,
        'InferenceLatency': inf_latency,
        'TrainTimePerEpochSec': avg_epoch_time,
        'Epochs': n_epochs,
        'Accuracy': cheb_accuracy,
        'Precision': cheb_precision,
        'Recall': cheb_recall,
        'F1': cheb_f1
    })

    # ChebNet + EdgeConv
    ChebEdge_ = ChebEdge(dataset.num_node_features, 32, dataset.num_classes, chebFilterSize, readout='meanmax')
    model_size = count_parameters(ChebEdge_)
    get_peak_memory = measure_peak_memory(device)
    chbedgModel, avg_epoch_time, n_epochs = train_with_epoch_time(ChebEdge_, train_loader, val_loader, device)
    peak_memory = get_peak_memory()
    inf_latency = measure_inference_latency(chbedgModel, test_loader, device)
    cheb_edge_accuracy, cheb_edge_precision, cheb_edge_recall, cheb_edge_f1 = test_model(chbedgModel, test_loader, device)
    metrics.append({
        'Model': 'ChebNet + EdgeConv',
        'Parameters': model_size,
        'PeakMemoryMB': peak_memory,
        'InferenceLatency': inf_latency,
        'TrainTimePerEpochSec': avg_epoch_time,
        'Epochs': n_epochs,
        'Accuracy': cheb_edge_accuracy,
        'Precision': cheb_edge_precision,
        'Recall': cheb_edge_recall,
        'F1': cheb_edge_f1
    })

    # Save metrics to CSV
    df = pd.DataFrame(metrics)
    df.to_csv('model_metrics.csv', index=False)
    print(df)
