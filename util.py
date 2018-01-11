import torch
from torch.autograd import Variable

def accuracy(y_pred, y):
    return (torch.max(y_pred, 1)[1] == y).float().mean().data[0]

''' Returns the mean accuracy on the test set, given a model '''
def eval_on_test(test_dataloader, model_fn):
    acc = 0
    for x, y in test_dataloader:
        x, y = Variable(x), Variable(y)

        if torch.cuda.is_available():
            x, y = x.cuda(), y.cuda()
        acc += accuracy(model_fn(x), y)
    return round(acc / float(len(test_dataloader)), 3)

''' Converts a list of (x, x) pairs into two Tensors ''' 
def into_tensor(data, into_vars=True):
    X1 = [x[0] for x in data]
    X2 = [x[1] for x in data]
    return Variable(torch.stack(X1)), Variable(torch.stack(X2))