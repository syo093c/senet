from res_senet import SE_ResNet
import torchvision
from torchvision.transforms import transforms
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
import multiprocessing
import random

def train():
    # prepare data and dataloader
    cpu_count = multiprocessing.cpu_count()
    batch_size = 128*2*2

    transform_train = transforms.Compose([
        #transforms.RandomCrop(32, padding=4),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    dataset = torchvision.datasets.CIFAR10('.', train=True, download=True)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataset.dataset.transform = transform_train
    val_dataset.dataset.transform = transform_test

    test_data = torchvision.datasets.CIFAR10('.', train=False, download=True,transform=transform_test)


    train_loader = DataLoader(train_dataset, batch_size=batch_size,num_workers=cpu_count)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset),num_workers=cpu_count)
    test_loader = DataLoader(test_data, batch_size=batch_size,num_workers=cpu_count)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    model= SE_ResNet(depth=2+6*5,num_classes=10)
    trainer = pl.Trainer(precision='16')
    trainer.fit(model,train_loader,val_dataloaders=train_loader)

def main():
    train()

if __name__ == '__main__':
    main()
