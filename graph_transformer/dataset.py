import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

def load_protein_dataset(seed=None):
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    dataset = TUDataset(root='.', name='PROTEINS')
    if generator is not None:
        num_samples = len(dataset)
        indices = torch.randperm(num_samples, generator=generator)
        dataset = dataset[indices]
    else:
        dataset = dataset.shuffle()
    train_dataset = dataset[:int(len(dataset)*0.8)]
    val_dataset = dataset[int(len(dataset)*0.8):int(len(dataset)*0.9)]
    test_dataset = dataset[int(len(dataset)*0.9):]
    return dataset, train_dataset, val_dataset, test_dataset

def get_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=64, seed=None):
    train_generator = torch.Generator().manual_seed(seed) if seed is not None else None
    val_generator = torch.Generator().manual_seed(seed+1) if seed is not None else None
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=train_generator)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, generator=val_generator)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
