# protein_gnn/train.py
# Training and evaluation logic for protein GNN
import time
import torch.nn.functional as F

def train(model, optimizer, train_loader, val_loader, args):
    min_loss = float('inf')
    patience_cnt = 0
    val_loss_values = []
    best_epoch = 0
    t = time.time()
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        best_epoch += 1
        model.train()
        loss_train = 0.0
        correct = 0
        for data in train_loader:
            optimizer.zero_grad()
            data = data.to(args.device)
            out = model(data)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            pred = out.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()
        acc_train = correct / len(train_loader.dataset)
        acc_val, loss_val = evaluate(model, val_loader, args)
        print(f'Epoch: {epoch+1:04d} loss_train: {loss_train:.6f} acc_train: {acc_train:.6f} loss_val: {loss_val:.6f} acc_val: {acc_val:.6f} time: {time.time() - t:.2f}s')
        val_loss_values.append(loss_val)
        if acc_val > best_val_acc:
            best_val_acc = acc_val
            best_model_state = model.state_dict()
            patience_cnt = 0
        else:
            patience_cnt += 1
        if patience_cnt == args.patience:
            break
    total_time = time.time() - t
    print(f'Optimization Finished! Total time elapsed: {total_time:.2f}s')
    # Load best model state before returning
    model.load_state_dict(best_model_state)
    return best_epoch, total_time, best_val_acc

def evaluate(model, loader, args):
    model.eval()
    correct = 0.0
    loss_test = 0.0
    for data in loader:
        data = data.to(args.device)
        out = model(data)
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        loss_test += F.nll_loss(out, data.y).item()
    return correct / len(loader.dataset), loss_test
