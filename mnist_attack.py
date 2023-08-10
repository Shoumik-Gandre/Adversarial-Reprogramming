import fire
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from adv_reprogram.program import MNISTReprogram
from torchvision import models, datasets, transforms
from torchmetrics import Accuracy


LR_DECAY = 0.05
LAMBDA = 0.05
BATCH_SIZE = 100
NUM_EPOCHS = 10


def main(dataset_root: str):

    device = torch.device('cuda')

    model = models.resnet50(weights=models.resnet.ResNet50_Weights.DEFAULT).to(device)
    model.eval()

    adv_program = MNISTReprogram(
        attack_dims=(1, 28, 28),
        victim_dims=(3, 224, 224), 
        network=model
    )

    parallel_module = nn.DataParallel(adv_program).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(parallel_module.parameters(), weight_decay=LAMBDA)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=LR_DECAY)

    train_dataset = datasets.MNIST(root=dataset_root, download=True, train=True, transform=transforms.ToTensor())
    eval_dataset = datasets.MNIST(root=dataset_root, download=True, train=False, transform=transforms.ToTensor())

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, num_workers=2)

    # Training
    for epoch in range(1, 1 + NUM_EPOCHS):

        # Train over all batches
        for inputs, labels in tqdm(train_dataloader):
            outputs = parallel_module(inputs)
            loss = criterion(outputs.to(device), labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        lr_scheduler.step()
        
        # Evaluate over all batches
        eval_accuracy = Accuracy()
        for inputs, labels in tqdm(eval_dataloader):
            outputs = parallel_module(inputs).argmax(dim=1)
            eval_accuracy(outputs, labels)
        
        print(f'epoch [{epoch}/{NUM_EPOCHS}] | Eval Accuracy = {eval_accuracy.compute()}')


if __name__ == '__main__':
    fire.Fire(main)