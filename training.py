import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from data import mnist_dataloader
from models import Classifier, Encoder
from util import eval_on_test

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
