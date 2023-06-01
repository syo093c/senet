from res_senet import SE_ResNet
import torchvision
from torchvision.transforms import transforms
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
import multiprocessing
import random
from vit import ViT
from vit import MagicNet
import torchinfo

def train():
    # prepare data and dataloader
    cpu_count = multiprocessing.cpu_count()
    batch_size = 128

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    dataset = torchvision.datasets.CIFAR10('.', train=True, download=True)

    train_size = int(0.99 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataset.dataset.transform = transform_train
    val_dataset.dataset.transform = transform_test

    test_data = torchvision.datasets.CIFAR10('.', train=False, download=True,transform=transform_test)


    train_loader = DataLoader(train_dataset, batch_size=batch_size,num_workers=cpu_count)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,num_workers=cpu_count)
    test_loader = DataLoader(test_data, batch_size=batch_size,num_workers=cpu_count)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    model= SE_ResNet(depth=2+6*3,num_classes=10)
    #model = ViT(image_size=(224,224),patch_size=(32,32),num_classes=10,dim=256,depth=6,heads=4,mlp_dim=128,channels=3)
    #model=MagicNet()

    trainer = pl.Trainer(precision='16')
    torch.set_float32_matmul_precision('medium')
    #trainer.fit(model,train_loader,val_dataloaders=val_loader)
    trainer.fit(model,train_loader,val_dataloaders=test_loader)

def test():
    model=SE_ResNet(depth=20,num_classes=10)
    torchinfo.summary(model)

def ltrain():
    batch_size = 128

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=20)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=20)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    depth = 20 # e.g. 20, 32, 44, 47, 56, 110, 1199
    model = SE_ResNet(depth=depth,num_classes=10)
    model.to('cuda')
    trainer = pl.Trainer(precision='16')
    torch.set_float32_matmul_precision('medium')
    #trainer.fit(model,train_loader,val_dataloaders=val_loader)
    trainer.fit(model,trainloader,val_dataloaders=testloader)

    torchinfo.summary(model, (3, 32, 32))


def main():
    #ltrain()
    train()
    #test()

if __name__ == '__main__':
    main()
