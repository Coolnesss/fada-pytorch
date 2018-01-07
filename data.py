import torchvision
import torch

''' Returns the MNIST dataloader '''
def mnist_dataloader(batch_size=256, train=True, cuda=False):
    dataset = torchvision.datasets.MNIST('./data', download=True, train=train, transform=torchvision.transforms.ToTensor())
    return torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=2, pin_memory=cuda)


''' Returns the SVHN dataloader '''
def svhn_dataloader(batch_size=256, train=True, cuda=False):
    dataset = torchvision.datasets.SVHN('./data', download=True, split=('train' if train else 'test'), transform=torchvision.transforms.ToTensor())
    return torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=2, pin_memory=cuda)