import torch
import numpy as np
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score
from copy import deepcopy
from tqdm import tqdm
import torch.nn as nn
from base import LabelSmoothingCrossEntropy

def train(model, train_loader, val_loader, device):
    model = model.to(device)
    loss_function = LabelSmoothingCrossEntropy()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=400, eta_min=1e-5)
    best_metric = -1
    best_metric_epoch = -1
    best_val_loss = 1000
    best_model = None
    epochs = 500
    early_stop = 30
    es_counter = 0
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        for i, data in enumerate(tqdm(train_loader)):
            batch = data.batch.to(device)
            x = data.x.to(device)
            y = data.y.to(device)
            z = data.edge_index.to(device)
            optimizer.zero_grad()
            out = model(x, z, batch)
            step_loss = loss_function(out, y)
            step_loss.backward(retain_graph=True)
            optimizer.step()
            epoch_train_loss += step_loss.item()
        epoch_train_loss = epoch_train_loss / (i + 1)
        lr_scheduler.step()
        val_loss, val_acc = validate_model(model, val_loader, device)
        if val_loss < best_val_loss:
            best_metric = val_acc
            best_val_loss = val_loss
            best_metric_epoch = epoch + 1
            best_model = deepcopy(model)
            es_counter = 0
        else:
            es_counter += 1
        if es_counter > early_stop:
            break
    return best_model

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
    acc = balanced_accuracy_score(preds, labels)
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
    accuracy = balanced_accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    return accuracy, precision, recall, f1
