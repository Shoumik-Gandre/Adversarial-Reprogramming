import torch
from torch.utils.data import Dataset, TensorDataset
from torchvision import transforms


def load_to_memory(dataset: Dataset) -> TensorDataset:
    x = torch.stack([inputs for (inputs, labels) in dataset])
    y = torch.tensor([labels for (inputs, labels) in dataset], dtype=torch.long)
    mean = x.mean(dim=(0, 2, 3))
    std = x.std(dim=(0, 2, 3))
    x = torch.stack([transforms.Normalize(mean, std)(image) for image in x])
    return TensorDataset(x, y)
