# CIFAR10 dataloaders
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

def get_dataloaders(batch_size, corruption=None):

    trans = transforms.Compose(
        [
        transforms.Resize(32),
        transforms.CenterCrop(28),
        transforms.ToTensor(),
        transforms.Normalize([0.49139968, 0.48215827 ,0.44653124], [0.24703233, 0.24348505, 0.26158768])
        ]
    ) 

    train_set = CIFAR10("./data", train=True, download=True, transform=trans)
    test_set = CIFAR10("./data", train=False, download=True, transform=trans)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader