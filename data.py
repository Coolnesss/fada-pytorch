import torchvision
import torch
import numpy as np

''' Returns the MNIST dataloader '''
def mnist_dataloader(batch_size=256, train=True, cuda=False):
    dataset = torchvision.datasets.MNIST('./data', download=True, train=train, transform=torchvision.transforms.ToTensor())
    return torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=2, pin_memory=cuda)


''' Returns the SVHN dataloader '''
def svhn_dataloader(batch_size=256, train=True, cuda=False):
    
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((28, 28)),
        torchvision.transforms.Grayscale(),
        torchvision.transforms.ToTensor()
    ])
    dataset = torchvision.datasets.SVHN('./data', download=True, split=('train' if train else 'test'), transform=transform)
    return torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=2, pin_memory=cuda)


''' Samples a subset from source into memory '''
def sample_data(n=2000):
    dataset = torchvision.datasets.MNIST('./data', download=True, train=True, transform=torchvision.transforms.ToTensor())
    X = torch.FloatTensor(n, 1, 28, 28)
    Y = torch.LongTensor(n)

    inds = torch.randperm(len(dataset))[:n]
    for i, index in enumerate(inds):
        x, y = dataset[index]
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

''' 
    Samples uniformly groups G1 and G3 from D_s x D_s and groups G2 and G4 from D_s x D_t  
'''
def create_groups(X_s, y_s, X_t, y_t):
    n = X_t.shape[0]
    G1, G3 = [], []
    
    # TODO optimize
    # Groups G1 and G3 come from the source domain
    for i, (x1, y1) in enumerate(zip(X_s, y_s)):
        for j, (x2, y2) in enumerate(zip(X_s, y_s)):
            if y1 == y2 and i != j and len(G1) < n:
                G1.append((x1, x2))
            if y1 != y2 and i != j and len(G3) < n:
                G3.append((x1, x2))

    G2, G4 = [], []

    # Groups G2 and G4 are mixed from the source and target domains
    for i, (x1, y1) in enumerate(zip(X_s, y_s)):
        for j, (x2, y2) in enumerate(zip(X_t, y_t)):
            if y1 == y2 and i != j and len(G2) < n:
                G2.append((x1, x2))
            if y1 != y2 and i != j and len(G4) < n:
                G4.append((x1, x2))
    
    groups = [G1, G2, G3, G4]

    # Make sure we sampled enough samples
    for g in groups:
        assert(len(g) == n)
    return groups

''' Sample groups G1, G2, G3, G4 '''
def sample_groups(n_target_samples=2):
    X_s, y_s = sample_data()
    X_t, y_t = create_target_samples(n_target_samples)
    
    print("Sampling groups")
    return create_groups(X_s, y_s, X_t, y_t), (X_s, y_s, X_t, y_t)
