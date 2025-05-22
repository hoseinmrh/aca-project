import torch
import numpy as np
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, accuracy_score
from copy import deepcopy
from tqdm import tqdm
import torch.nn as nn
from base import LabelSmoothingCrossEntropy
from torch.optim.lr_scheduler import ReduceLROnPlateau
from set_seed import set_seed

def enhanced_train(model, train_loader, val_loader, device, patience=30, max_epochs=500, seed=None):
    """
    Enhanced training function with learning rate scheduling and improved early stopping
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to train on (cpu or cuda)
        patience: Early stopping patience
        max_epochs: Maximum number of epochs to train for
        seed: Random seed for reproducibility
        
    Returns:
        The best model based on validation loss
    """
    # Set seed if provided
    if seed is not None:
        set_seed(seed)
    model = model.to(device)
    loss_function = LabelSmoothingCrossEntropy()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    # Training tracking
    best_metric = -1
    best_metric_epoch = -1
    best_val_loss = float('inf')
    best_model = None
    es_counter = 0
    train_losses = []
    val_losses = []
    val_accs = []
    number_of_epochs = 0
    # Main training loop
    for epoch in range(max_epochs):
        number_of_epochs += 1
        model.train()
        epoch_train_loss = 0
        for i, data in enumerate(train_loader):
            batch = data.batch.to(device)
            x = data.x.to(device)
            y = data.y.to(device)
            z = data.edge_index.to(device)
            
            optimizer.zero_grad()
            out = model(x, z, batch)
            step_loss = loss_function(out, y)
            step_loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_train_loss += step_loss.item()
            
        epoch_train_loss = epoch_train_loss / (i + 1)
        train_losses.append(epoch_train_loss)
        
        # Validation
        val_loss, val_acc = validate_model(model, val_loader, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Print epoch stats
        print(f"Epoch {epoch+1}/{max_epochs}, Train Loss: {epoch_train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_metric = val_acc
            best_val_loss = val_loss
            best_metric_epoch = epoch + 1
            best_model = deepcopy(model)
            es_counter = 0
            print(f"  New best model! Val Loss: {best_val_loss:.4f}, Val Acc: {best_metric:.4f}")
        else:
            es_counter += 1
            
        # Early stopping
        if es_counter > patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
            
    print(f"Training complete. Best model at epoch {best_metric_epoch} with "
          f"Val Loss: {best_val_loss:.4f}, Val Acc: {best_metric:.4f}")
    
    return best_model, number_of_epochs

# Reuse the validate_model and test_model functions from the original train.py
def validate_model(model, val_loader, device):
    model.eval()
    val_loss = 0
    loss_func = nn.CrossEntropyLoss()
    labels = []
    preds = []
    for i, data in enumerate(val_loader):
        batch = data.batch.to(device)
        x = data.x.to(device)
        label = data.y.to(device)
        z = data.edge_index.to(device)
        out = model(x, z, batch)
        step_loss = loss_func(out, label)
        val_loss += step_loss.detach().item()
        preds.append(out.argmax(dim=1).detach().cpu().numpy())
        labels.append(label.cpu().numpy())
    preds = np.concatenate(preds).ravel()
    labels = np.concatenate(labels).ravel()
    acc = balanced_accuracy_score(labels, preds)
    loss = val_loss / (i + 1)
    return loss, acc

def test_model(model, test_loader, device):
    model.eval()
    labels = []
    preds = []
    for i, data in enumerate(test_loader):
        batch = data.batch.to(device)
        x = data.x.to(device)
        label = data.y.to(device)
        z = data.edge_index.to(device)
        out = model(x, z, batch)
        preds.append(out.argmax(dim=1).detach().cpu().numpy())
        labels.append(label.cpu().numpy())
    preds = np.concatenate(preds).ravel()
    labels = np.concatenate(labels).ravel()
    balanced_accuracy = balanced_accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    accuracy = accuracy_score(labels, preds)
    return balanced_accuracy, precision, recall, f1, accuracy
