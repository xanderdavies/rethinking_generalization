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
from custom_relu import CustomReLU

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
parser.add_argument('--no-logging', action='store_true', help='disable logging')
parser.add_argument('--random-labels', action='store_true', help='corrupt all labels')
parser.add_argument('--use-adam', action='store_true', help='use adam instead of SGD-M')
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
        nn.Flatten(),
        nn.Linear(28*28*3, 512),
        CustomReLU(id=0),
        nn.Linear(512, 10)
    ).to(DEVICE)
elif args.model_name == 'mlp-3x512': # 3 hidden layers
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28*3, 512),
        CustomReLU(id=0),
        nn.Linear(512, 512),
        CustomReLU(id=1),
        nn.Linear(512, 512),
        CustomReLU(id=2),
        nn.Linear(512, 10)
    ).to(DEVICE)
if args.load_state is not None:
    model.load_state_dict(torch.load(args.load_state))

# get data
train_loader, test_loader = get_dataloaders(BATCH_SIZE, random_labels=args.random_labels)

# opt + loss
optimizer = SGD(model.parameters(), lr=LR, momentum=MOMENTUM) if not args.use_adam else torch.optim.Adam(model.parameters(), lr=LR)
scheduler = ExponentialLR(optimizer, LR_DECAY)
criterion = nn.CrossEntropyLoss()

# initialize and train
torch.manual_seed(0)
run = None
if not args.no_logging:
    tags = [args.model_name]
    if args.random_labels:
        tags.append('random_labels')
    if args.use_adam:
        tags.append('adam')
    run = wandb.init(
        project=args.wandb_project, 
        config=args, 
        tags=tags, 
        settings=wandb.Settings(start_method="thread"),
        name=('-').join(tags) + datetime.now().strftime("-%Y%m%d-%H%M%S")
    )

best_test_err = 1
for ep in range(EPOCHS):
    train_err, train_loss = epoch(train_loader, model, criterion, epoch_num=ep, total_epochs=EPOCHS, opt=optimizer, sched=scheduler)
    test_err, test_loss = epoch(test_loader, model, criterion)

    if test_err < best_test_err:
        write_state_dict(f'./best_weights/{run.name if run is not None else "scratch"}/best_weights.ckpt', model)
        if not args.no_logging:
            wandb.save(f'./best_weights/{run.name if run is not None else "scratch"}/best_weights.ckpt')
    
    log_data(ep, train_err, train_loss, test_err, test_loss, no_logging=args.no_logging)
