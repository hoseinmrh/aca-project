import torch
import numpy as np
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import torch.nn as nn

def train(model, train_loader, val_loader, device, max_epochs=300, patience=30):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    loss_function = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    best_model = None
    es_counter = 0
    best_epoch = 0
    for epoch in range(max_epochs):
        best_epoch = epoch + 1
        model.train()
        epoch_loss = 0
        for data in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            data = data.to(device)
            optimizer.zero_grad()
            # Only pass edge_attr if it exists and is not None
            if hasattr(model, 'edge_encoder'):
                edge_attr = getattr(data, 'edge_attr', None)
                if edge_attr is not None:
                    out = model(data.x, data.edge_index, data.batch, edge_attr=edge_attr)
                else:
                    out = model(data.x, data.edge_index, data.batch)
            else:
                out = model(data.x, data.edge_index, data.batch)
            loss = loss_function(out, data.y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)
        val_loss, val_acc = validate(model, val_loader, device)
        print(f"Epoch {epoch+1}: Train Loss={epoch_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            es_counter = 0
        else:
            es_counter += 1
        if es_counter > patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    model.load_state_dict(best_model)
    return model, best_epoch

def validate(model, loader, device):
    model.eval()
    loss_function = nn.CrossEntropyLoss()
    val_loss = 0
    labels = []
    preds = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            # Only pass edge_attr if it exists and is not None
            if hasattr(model, 'edge_encoder'):
                edge_attr = getattr(data, 'edge_attr', None)
                if edge_attr is not None:
                    out = model(data.x, data.edge_index, data.batch, edge_attr=edge_attr)
                else:
                    out = model(data.x, data.edge_index, data.batch)
            else:
                out = model(data.x, data.edge_index, data.batch)
            loss = loss_function(out, data.y)
            val_loss += loss.item()
            preds.append(out.argmax(dim=1).cpu().numpy())
            labels.append(data.y.cpu().numpy())
    preds = np.concatenate(preds).ravel()
    labels = np.concatenate(labels).ravel()
    acc = balanced_accuracy_score(labels, preds)
    val_loss /= len(loader)
    return val_loss, acc

def test_model(model, loader, device):
    model.eval()
    labels = []
    preds = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            # Only pass edge_attr if it exists and is not None
            if hasattr(model, 'edge_encoder'):
                edge_attr = getattr(data, 'edge_attr', None)
                if edge_attr is not None:
                    out = model(data.x, data.edge_index, data.batch, edge_attr=edge_attr)
                else:
                    out = model(data.x, data.edge_index, data.batch)
            else:
                out = model(data.x, data.edge_index, data.batch)
            preds.append(out.argmax(dim=1).cpu().numpy())
            labels.append(data.y.cpu().numpy())
    preds = np.concatenate(preds).ravel()
    labels = np.concatenate(labels).ravel()
    accuracy = balanced_accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    return accuracy, precision, recall, f1
