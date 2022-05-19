import torch.nn as nn
from torch.optim import SGD
from torchvision.models.resnet import resnet18
import torch.cuda as cuda
import torch 
from datetime import datetime
import argparse
from torch.optim.lr_scheduler import ExponentialLR
import wandb
from cifar10 import get_dataloaders
from utils import epoch, log_data, write_state_dict

# argparse
parser = argparse.ArgumentParser(description="""Reproducing Zhang et al.'s 'Understanding Deep Neural Networks Requires Rethinking Generalization'.""")
parser.add_argument('--batch-size', type=int, default=128, help='batch size')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
parser.add_argument('--lr-decay', type=float, default=0.95, help='learning rate decay for ExponentialLR')
parser.add_argument('--model-name', type=str, default='resnet18', help='model name (resnet18, mlp-1x512, mlp-3x512)')
parser.add_argument('--load-state', type=str, default=None, help='load state dict from file')
parser.add_argument('--wandb-project', type=str, default='rethinking-generalization', help='wandb project name')
args = parser.parse_args()

BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
LR = args.lr
MOMENTUM = args.momentum
LR_DECAY = args.lr_decay
DEVICE = 'cuda' if cuda.is_available() else 'cpu'; print("using", DEVICE)

# get model
if args.model_name == 'resnet18':
    model = resnet18(pretrained=False).to(DEVICE)
    model.fc = nn.Linear(512, 10).to(DEVICE)
elif args.model_name == 'mlp-1x512':
    model = nn.Sequential(
        nn.Linear(28*28*3, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    ).to(DEVICE)
elif args.model_name == 'mlp-3x512': # 3 hidden layers
    model = nn.Sequential(
        nn.Linear(28*28*3, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    ).to(DEVICE)
if args.load_state is not None:
    model.load_state_dict(torch.load(args.load_state))

# get data
train_loader, test_loader = get_dataloaders(BATCH_SIZE)

# opt + loss
optimizer = SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
scheduler = ExponentialLR(optimizer, LR_DECAY)
criterion = nn.CrossEntropyLoss()

# initialize and train
torch.manual_seed(0)
run = wandb.init(
    project=args.wandb_project, 
    config=args, 
    tags=[args.model_name], 
    settings=wandb.Settings(start_method="thread")
)

best_test_err = 1
for ep in range(EPOCHS):
    train_err, train_loss = epoch(train_loader, model, criterion, epoch_num=ep, total_epochs=100, opt=optimizer, sched=scheduler)
    test_err, test_loss = epoch(test_loader, model, criterion)

    if test_err < best_test_err:
        write_state_dict(f'./best_weights/{run.name}/best_weights.ckpt')
        wandb.save(f'./best_weights/{run.name}/best_weights.ckpt')
    
    log_data(ep, train_err, train_loss, test_err, test_loss)
