import torch
from torch.autograd import Variable

def accuracy(y_pred, y):
    return (torch.max(y_pred, 1)[1] == y).float().mean().data[0]

''' Returns the mean accuracy on the test set, given a model '''
def eval_on_test(test_dataloader, model_fn):
    acc = 0
    for x, y in test_dataloader:
        x, y = Variable(x), Variable(y)
        acc += accuracy(model_fn(x), y)
    return acc / float(len(test_dataloader))