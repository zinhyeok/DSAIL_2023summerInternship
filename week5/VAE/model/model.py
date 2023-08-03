import torch 
from torch import nn    
from torch.nn import functional as F
from base import BaseModel

#Input img -> hidden dim -> mean, std -> Parameterize -> sample z -> decode -> output img
#latent: Gaussian distribution

class VarationalAutoEncoder(BaseModel):
    def __init__(self, input_dim=28*2, hidden_dim=200, latent_dim=20):
        super(VarationalAutoEncoder, self).__init__()

        #encoder
        self.img_2hid = nn.Linear(input_dim, hidden_dim)
        self.hid_2mu = nn.Linear(hidden_dim, latent_dim)
        self.hid_2std = nn.Linear(hidden_dim, latent_dim)

        #decoder
        self.z_2hid = nn.Linear(latent_dim, hidden_dim)
        self.hid_2img = nn.Linear(hidden_dim, input_dim)


    def encoder(self, x):   
        #q_phi(z|x)
        h = F.relu(self.img_2hid(x))
        mu, sigma = self.hid_2mu(h), self.hid_2std(h)

        return mu, sigma

    def decoder(self, z):
        #p_theta(x|z)
        h = F.relu(self.z_2hid(z))
        return torch.sigmoid(self.hid_2img(h))
    
    def reparameterize(self, mu, sigma):
        #sample z
        std = torch.exp(0.5 * sigma)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, sigma = self.encoder(x)
        z_reparametrize = self.reparameterize(mu, sigma)
        x_hat = self.decoder(z_reparametrize)
        return x_hat, mu, sigma



if __name__ == "__main__":
    print("Test")
    x=torch.randn(1,28*28)
    vae = VarationalAutoEncoder(input_dim=28*28)
    x_recon, mu, sigma = vae(x)
    print(x_recon.shape)
    print(mu.shape)
    print(sigma.shape)  

