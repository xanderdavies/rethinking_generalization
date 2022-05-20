
import torch.nn as nn
import torch.nn.functional as F
import torch
import wandb
import numpy as np

class CustomReLU(nn.Module):
    def __init__(self, inplace: bool = False, id: int = 0):
        super(CustomReLU, self).__init__()
        self.inplace = inplace
        self.neuron_counter = None
        self.id = id
        self.epoch = 0

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = F.relu(input, inplace=self.inplace)

        # count number of neurons that are active by summing across batch
        if self.training:
            if self.neuron_counter is None:
                self.neuron_counter = output.sum(dim=0)
            else:
                self.neuron_counter += output.sum(dim=0)        
        elif self.neuron_counter is not None:
            wandb.log({
                f"dead_neurons/Dead Neuron Prevalence After ReLU {self.id}": 1 - (self.neuron_counter > 0).sum().item() / np.prod(self.neuron_counter.shape),
                "epoch": self.epoch
            })
            self.neuron_counter = None
            self.epoch += 1
        return output
