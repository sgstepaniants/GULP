import torchvision
import torch
import torchvision.transforms as transforms
import math
from torch.utils.data import Dataset

def get_cifar100_loaders(batch_size=64,shuffle=True,num_workers=0):

    scale_factor = 1 / math.sqrt(5389.3779)
    stats = ((0.5074,0.4867,0.4411),(0.2011 / scale_factor,0.1987 / scale_factor,0.2025 / scale_factor))
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32,padding=4,padding_mode="reflect"),
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    train_data = torchvision.datasets.CIFAR100(download=True,root="./data",transform=train_transform)
    test_data = torchvision.datasets.CIFAR100(root="./data",train=False,transform=test_transform)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               shuffle=shuffle, num_workers=num_workers)

    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                          shuffle=shuffle, num_workers=num_workers)

    return trainloader, testloader


def get_cifar10_loaders(batch_size=64,shuffle=True,num_workers=0):

    scale_factor = 1 / math.sqrt(786.3883)
    print(scale_factor)
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5 / scale_factor, 0.5 / scale_factor, 0.5 / scale_factor))])


    train_data = torchvision.datasets.CIFAR10(root='./data', train=True,
                                         download=True, transform=transform)

    test_data = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               shuffle=shuffle, num_workers=num_workers)

    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                          shuffle=shuffle, num_workers=num_workers)

    return trainloader, testloader

class Fn_on_Uniform_Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, eval_fn, discretization):
        """
        Args:
        """
        self.eval_fn = eval_fn
        self.discretization = discretization

    def __len__(self):
        return self.discretization

    def __getitem__(self, idx):
        assert(idx < self.discretization)
        assert(idx >= 0)
        if torch.is_tensor(idx):
            print('idx',idx)
            idx = idx.tolist()
            print('idx',idx)
            assert(False)

        return idx / self.discretization, self.eval_fn(idx / self.discretization)


def get_one_dimensional_loader_on_unif(eval_fn, train_discretization, test_discretization, batch_size=64,shuffle=True,num_workers=0):

    train_data = Fn_on_Uniform_Dataset(eval_fn, train_discretization)
    test_data = Fn_on_Uniform_Dataset(eval_fn, test_discretization)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               shuffle=shuffle, num_workers=num_workers)

    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                          shuffle=shuffle, num_workers=num_workers)

    return trainloader, testloader
