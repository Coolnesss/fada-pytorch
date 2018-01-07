import torchvision
import torch

''' Returns the MNIST dataloader '''
def mnist_dataloader(batch_size=256, train=True, cuda=False):
    dataset = torchvision.datasets.MNIST('./data', download=True, train=train, transform=torchvision.transforms.ToTensor())
    return torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=cuda)