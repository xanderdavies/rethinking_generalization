from tqdm import tqdm
import os
import torch
import wandb

def log_data(ep, train_err, train_loss, test_err, test_loss, no_logging=False):
    if not no_logging:
        wandb.log({
            "Loss/train": train_loss,
            "Acc/train": 1 - train_err,
            "Loss/test": test_loss,
            "Acc/test": 1 - test_err,
            "epoch": ep
        })

    print("Train Loss", train_loss)
    print("Train Accuracy", 1 - train_err)
    print("Test Accuracy", 1 - test_err)

def epoch(loader, model, criterion, device='cuda', epoch_num=0, total_epochs=100, opt=None, sched=None):
    total_loss, total_err = 0.,0.

    if opt is None:
        model.eval()
    else:
        model.train() 

    for X,y in tqdm(loader, desc=f'epoch {epoch_num}/{total_epochs}' if opt is not None else 'validating'):
        X,y = X.to(device), y.to(device)
        yp = model(X)
        loss = criterion(yp,y)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
    
    if sched is not None and opt is not None:
        sched.step()

    return total_err / len(loader.dataset), total_loss / len(loader.dataset)

def write_state_dict(path, model):
    if not os.path.isdir(os.path.dirname(path)):
        print("making dir...")
        for i in range(len(path.split("/"))-1):
            p = ('/').join(path.split("/")[:i+1])
            if p == '.':
              continue
            elif not os.path.isdir(p):
              os.mkdir(p)
    torch.save(model.state_dict(), path)