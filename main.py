from training import pretrain, train_discriminator
import torch

if __name__ == '__main__':
    cuda = torch.cuda.is_available()

    encoder, classifier = pretrain(cuda=cuda, epochs=1)
    train_discriminator(encoder)