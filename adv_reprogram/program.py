from abc import ABC, abstractmethod
import math
from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


AdapterFn = Callable[[torch.Tensor], torch.Tensor]
OutputMapperFn = Callable[[torch.Tensor], torch.Tensor]


class GenericProgram(nn.Module, ABC):

    # Type Annotations
    # mean: torch.Tensor
    # std: torch.Tensor
    mask: torch.Tensor

    def __init__(self, attack_dims: tuple[int, ...], victim_dims: tuple[int, ...], mask: torch.Tensor, network: nn.Module) -> None:
        super().__init__()
        self.register_buffer('mask', mask)
        self.W = nn.Parameter(torch.randn(victim_dims), requires_grad=True)
        self.network = network
        self.attack_dims = attack_dims
        self.victim_dims = victim_dims

        for param in self.network.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # h_f(.)
        X = self.adapter_fn(x).requires_grad_()
        P = torch.tanh(self.W * self.mask)
        X_adv = X + P
        # f(.)
        Y_adv = F.softmax(self.network(X_adv), dim=1)
        # h_g(.)
        return self.output_mapper(Y_adv)
    
    def get_program(self) -> torch.Tensor:
        return torch.tanh(self.W * self.mask)
    
    @abstractmethod
    def adapter_fn(self, x: torch.Tensor) -> torch.Tensor:
        """This function adapts the dimensions of x into the dimensions that are suitable as the input of the network

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: The X in the adversarial programming paper
        """
        pass

    @abstractmethod
    def output_mapper(self, y: torch.Tensor) -> torch.Tensor:
        """This function adapts the label space from the model to the label space of target tast

        Args:
            y (torch.Tensor): _description_

        Returns:
            torch.Tensor: The label y for target task
        """
        pass
        

class MNISTReprogram(GenericProgram):

    def __init__(self, attack_dims: tuple[int, ...], victim_dims: tuple[int, ...], network: nn.Module) -> None:
        mask = self._get_mask(attack_dims, victim_dims)
        super().__init__(attack_dims, victim_dims, mask, network)

    def _get_mask(self, in_dimensions: tuple[int, ...], output_dimensions: tuple[int, ...]) -> torch.Tensor:
        in_channels, in_height, in_width = in_dimensions
        w_center = math.ceil(in_width/2)
        h_center = math.ceil(in_height/2)
        w_start  = w_center - in_width // 2
        w_end    = w_start + in_width
        h_start  = h_center - in_height // 2
        h_end    = h_start + in_height

        M = torch.ones(output_dimensions, dtype=torch.float)
        M[:, h_start:h_end, w_start:w_end] = 0
        return M

    def adapter_fn(self, x: torch.Tensor) -> torch.Tensor:
        X = torch.zeros((x.shape[0], *self.victim_dims), device=x.device)

        start_x = self.victim_dims[1] // 2 - self.attack_dims[1] // 2
        end_x = start_x + self.attack_dims[1] 
        start_y = self.victim_dims[2] // 2 - self.attack_dims[2] // 2
        end_y = start_y + self.attack_dims[2]

        x = x.expand(x.shape[0], self.victim_dims[0], self.attack_dims[1], self.attack_dims[2])
        X[:, :, start_x:end_x, start_y:end_y] = x.data.clone()

        return X
    
    def output_mapper(self, y: torch.Tensor) -> torch.Tensor:
        return y[:, :10]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # h_f(.)
        X = self.adapter_fn(x).requires_grad_()
        P = torch.sigmoid(self.W * self.mask)
        X_adv = X + P
        mean = torch.tensor([0.485, 0.456, 0.406], device=X_adv.device).reshape(-1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=X_adv.device).reshape(-1, 1, 1)
        X_adv = (X_adv - mean) / std
        # f(.)
        Y_adv = F.softmax(self.network(X_adv), dim=1)
        
        # h_g(.)
        return self.output_mapper(Y_adv)