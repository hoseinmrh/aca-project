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
from gcn_gat_hybrid import GCN_GAT_Hybrid

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

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    dataset, train_dataset, val_dataset, test_dataset = load_protein_dataset()
    train_loader, val_loader, test_loader = get_dataloaders(train_dataset, val_dataset, test_dataset)

    metrics = []

    # GCN
    # GCN_ = GCN(dataset.num_node_features, 64, dataset.num_classes, readout='sum', dropout=0.3, num_layers=4)
    # model_size = count_parameters(GCN_)
    # get_peak_memory = measure_peak_memory(device)
    # gcnModel = train(GCN_, train_loader, val_loader, device)
    # peak_memory = get_peak_memory()
    # inf_latency = measure_inference_latency(gcnModel, test_loader, device)
    # gcn_accuracy, gcn_precision, gcn_recall, gcn_f1 = test_model(gcnModel, test_loader, device)
    # metrics.append({
    #     'Model': 'GCN',
    #     'Parameters': model_size,
    #     'PeakMemoryMB': peak_memory,
    #     'InferenceLatency': inf_latency,
    #     'TrainTimePerEpochSec': None,
    #     'Epochs': None,
    #     'Accuracy': gcn_accuracy,
    #     'Precision': gcn_precision,
    #     'Recall': gcn_recall,
    #     'F1': gcn_f1
    # })

    # # GIN
    # GIN_ = GIN(dataset.num_node_features, 64, dataset.num_classes, readout='mean', dropout=0.3)
    # model_size = count_parameters(GIN_)
    # get_peak_memory = measure_peak_memory(device)
    # ginModel = train(GIN_, train_loader, val_loader, device)
    # peak_memory = get_peak_memory()
    # inf_latency = measure_inference_latency(ginModel, test_loader, device)
    # gin_accuracy, gin_precision, gin_recall, gin_f1 = test_model(ginModel, test_loader, device)
    # metrics.append({
    #     'Model': 'GIN',
    #     'Parameters': model_size,
    #     'PeakMemoryMB': peak_memory,
    #     'InferenceLatency': inf_latency,
    #     'TrainTimePerEpochSec': None,
    #     'Epochs': None,
    #     'Accuracy': gin_accuracy,
    #     'Precision': gin_precision,
    #     'Recall': gin_recall,
    #     'F1': gin_f1
    # })

    # # GAT
    # GAT_ = GAT(dataset.num_node_features, 64, dataset.num_classes, readout='sum', concat=True, num_heads=8, dropout=0.3)
    # model_size = count_parameters(GAT_)
    # get_peak_memory = measure_peak_memory(device)
    # gatModel = train(GAT_, train_loader, val_loader, device)
    # peak_memory = get_peak_memory()
    # inf_latency = measure_inference_latency(gatModel, test_loader, device)
    # gat_accuracy, gat_precision, gat_recall, gat_f1 = test_model(gatModel, test_loader, device)
    # metrics.append({
    #     'Model': 'GAT',
    #     'Parameters': model_size,
    #     'PeakMemoryMB': peak_memory,
    #     'InferenceLatency': inf_latency,
    #     'TrainTimePerEpochSec': None,
    #     'Epochs': None,
    #     'Accuracy': gat_accuracy,
    #     'Precision': gat_precision,
    #     'Recall': gat_recall,
    #     'F1': gat_f1
    # })

    # # ChebNet
    chebFilterSize = 16
    # ChebC_ = ChebC(dataset.num_node_features, 64, dataset.num_classes, chebFilterSize, readout='sum')
    # model_size = count_parameters(ChebC_)
    # get_peak_memory = measure_peak_memory(device)
    # chebModel = train(ChebC_, train_loader, val_loader, device)
    # peak_memory = get_peak_memory()
    # inf_latency = measure_inference_latency(chebModel, test_loader, device)
    # cheb_accuracy, cheb_precision, cheb_recall, cheb_f1 = test_model(chebModel, test_loader, device)
    # metrics.append({
    #     'Model': 'ChebNet',
    #     'Parameters': model_size,
    #     'PeakMemoryMB': peak_memory,
    #     'InferenceLatency': inf_latency,
    #     'TrainTimePerEpochSec': None,
    #     'Epochs': None,
    #     'Accuracy': cheb_accuracy,
    #     'Precision': cheb_precision,
    #     'Recall': cheb_recall,
    #     'F1': cheb_f1
    # })

    # # ChebNet + EdgeConv
    ChebEdge_ = ChebEdge(dataset.num_node_features, 64, dataset.num_classes, chebFilterSize, readout='mean')
    model_size = count_parameters(ChebEdge_)
    get_peak_memory = measure_peak_memory(device)
    chbedgModel = train(ChebEdge_, train_loader, val_loader, device)
    peak_memory = get_peak_memory()
    inf_latency = measure_inference_latency(chbedgModel, test_loader, device)
    cheb_edge_accuracy, cheb_edge_precision, cheb_edge_recall, cheb_edge_f1 = test_model(chbedgModel, test_loader, device)
    metrics.append({
        'Model': 'ChebNet + EdgeConv',
        'Parameters': model_size,
        'PeakMemoryMB': peak_memory,
        'InferenceLatency': inf_latency,
        'TrainTimePerEpochSec': None,
        'Epochs': None,
        'Accuracy': cheb_edge_accuracy,
        'Precision': cheb_edge_precision,
        'Recall': cheb_edge_recall,
        'F1': cheb_edge_f1
    })

    # # GCN-GAT Hybrid
    # Hybrid_ = GCN_GAT_Hybrid(dataset.num_node_features, 48, dataset.num_classes, readout='sum', dropout=0.3, num_heads=4)
    # model_size = count_parameters(Hybrid_)
    # get_peak_memory = measure_peak_memory(device)
    # hybridModel = train(Hybrid_, train_loader, val_loader, device)
    # peak_memory = get_peak_memory()
    # inf_latency = measure_inference_latency(hybridModel, test_loader, device)
    # hybrid_accuracy, hybrid_precision, hybrid_recall, hybrid_f1 = test_model(hybridModel, test_loader, device)
    # hybrid_metrics = {
    #     'Model': 'GCN-GAT Hybrid',
    #     'Parameters': model_size,
    #     'PeakMemoryMB': peak_memory,
    #     'InferenceLatency': inf_latency,
    #     'TrainTimePerEpochSec': None,
    #     'Epochs': None,
    #     'Accuracy': hybrid_accuracy,
    #     'Precision': hybrid_precision,
    #     'Recall': hybrid_recall,
    #     'F1': hybrid_f1
    # }
    # # Save hybrid metrics to a new CSV
    # pd.DataFrame([hybrid_metrics]).to_csv('hybrid_model_metrics.csv', index=False)

    # Save metrics to CSV
    df = pd.DataFrame(metrics)
    df.to_csv('model_metrics.csv', index=False)
    print(df)
