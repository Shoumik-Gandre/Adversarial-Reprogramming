from abc import ABC, abstractmethod
import math
from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F


class DomainAdapter(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        ...


class CoDomainAdapter(nn.Module):
    ...