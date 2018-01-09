import torchvision
import torch
import numpy as np

''' Returns the MNIST dataloader '''
def mnist_dataloader(batch_size=256, train=True, cuda=False):
    dataset = torchvision.datasets.MNIST('./data', download=True, train=train, transform=torchvision.transforms.ToTensor())
    return torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=2, pin_memory=cuda)


''' Returns the SVHN dataloader '''
def svhn_dataloader(batch_size=256, train=True, cuda=False):
    dataset = torchvision.datasets.SVHN('./data', download=True, split=('train' if train else 'test'), transform=torchvision.transforms.ToTensor())
    return torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=2, pin_memory=cuda)


''' Samples a subset from source into memory '''
def sample_data(n=2000):
    dataset = torchvision.datasets.MNIST('./data', download=True, train=True, transform=torchvision.transforms.ToTensor())
    X = torch.FloatTensor(n, 1, 28, 28)
    Y = torch.LongTensor(n)
    for i in range(n):
        x, y = dataset[i]
        X[i] = x
        Y[i] = y
    return X, Y

''' Returns a subset of the target domain such that it has n_target_samples per class '''
def create_target_samples(n=1):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((28, 28)),
        torchvision.transforms.Grayscale(),
        torchvision.transforms.ToTensor()
    ])

    dataset = torchvision.datasets.SVHN('./data', download=True, split='train', transform=transform)
    X, Y = [], []
    classes = 10 * [n]

    i = 0
    while True:
        if len(X) == n*10:
            break
        x, y = dataset[i]
        if classes[y] > 0:
            X.append(x)
            Y.append(y)
            classes[y] -= 1
        i += 1

    assert(len(X) == n*10)
    return torch.stack(X), torch.from_numpy(np.array(Y))
