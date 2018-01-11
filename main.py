from training import pretrain, train_discriminator, train
from data import sample_groups
import torch

n_target_samples = 7
plot_accuracy = True

if __name__ == '__main__':
    cuda = torch.cuda.is_available()

    encoder, classifier = pretrain(cuda=cuda, epochs=5)

    groups, data = sample_groups(n_target_samples=n_target_samples)
    discriminator = train_discriminator(encoder, groups, n_target_samples=n_target_samples, epochs=100)

    train(encoder, discriminator, classifier, data, groups, n_target_samples=n_target_samples, cuda=cuda, epochs=150, plot_accuracy=plot_accuracy)