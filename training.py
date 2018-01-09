import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from data import mnist_dataloader, svhn_dataloader, sample_data, create_target_samples
from models import Classifier, Encoder, DCD
from util import eval_on_test
import random

def model_fn(encoder, classifier):
    return lambda x: classifier(encoder(x))

''' Pretrain the encoder and classifier as in (a) in figure 2. '''
def pretrain(epochs=5, cuda=False):

    train_dataloader = mnist_dataloader(cuda=cuda)
    test_dataloader = mnist_dataloader(train=False, cuda=cuda)

    classifier = Classifier()
    encoder = Encoder()

    if cuda:
        classifier.cuda()
        encoder.cuda()

    ''' Jointly optimize both encoder and classifier ''' 
    optimizer = optim.Adam(list(encoder.parameters()) + list(classifier.parameters()))
    loss_fn = nn.CrossEntropyLoss()
    
    for e in range(epochs):
        
        # TODO use only 2000 samples 
        for x, y in train_dataloader:
            
            optimizer.zero_grad()

            x, y = Variable(x), Variable(y)

            if cuda:
                x, y = x.cuda(), y.cuda()

            y_pred = model_fn(encoder, classifier)(x)

            loss = loss_fn(y_pred, y)

            loss.backward()

            optimizer.step()

        print("Epoch", e, "Loss", loss.data[0], "Accuracy", eval_on_test(test_dataloader, model_fn(encoder, classifier)))
    
    return encoder, classifier

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
    

''' Train the discriminator while the encoder is frozen '''
def train_discriminator(encoder, n_target_samples=2, cuda=False, epochs=20):

    source_loader = mnist_dataloader(train=True, cuda=cuda)
    target_loader = svhn_dataloader(train=True, cuda=cuda)

    X_s, y_s = sample_data()
    X_t, y_t = create_target_samples(n_target_samples)
    
    print("Sampling groups")
    groups = create_groups(X_s, y_s, X_t, y_t)

    discriminator = DCD(D_in=128) # Takes in concatenated hidden representations
    loss_fn = nn.CrossEntropyLoss()

    # Only train DCD
    optimizer = optim.Adam(discriminator.parameters())
    
    # Size of group G2, the smallest one, times the amount of groups
    n_iters = 4 * n_target_samples

    if cuda:
        discriminator.cuda()

    print("Training DCD")
    for e in range(epochs):

        for _ in range(n_iters):
            
            # Sample a pair of samples from a group
            group = random.choice([0, 1, 2, 3])

            x1, x2 = groups[group][random.randint(0, len(groups[group]) - 1)]
            x1, x2 = Variable(x1), Variable(x2)

            # Optimize the DCD using sample drawn
            optimizer.zero_grad()

            # Concatenate encoded representations
            x_cat = torch.cat([encoder(x1.unsqueeze(0)), encoder(x2.unsqueeze(0))], 1)
            y_pred = discriminator(x_cat)

            # Label is the group
            loss = loss_fn(y_pred, Variable(torch.LongTensor([group])))

            loss.backward()

            optimizer.step()

        print("Epoch", e, "Loss", loss.data[0])
