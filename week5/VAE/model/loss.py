import torch.nn.functional as F


def vae_loss(x, x_reconst, mu, sigma):
        # compute reconstruction loss and KL divergence
        reconst_loss = F.binary_cross_entropy(x_reconst, x, reduction='sum')
        kl_div = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1)
        # backprop and optimize
        loss = reconst_loss + kl_div
        return loss 

def mse_loss(output, target):
    return F.mse_loss(output, target)