import torch
import torch.nn as nn
from constants import IMG_SIZE


class LinearVAE(nn.Module):
    def __init__(self, num_features):
        super(LinearVAE, self).__init__()
        self.num_features = num_features

        self.encoder = nn.Sequential(
            nn.Linear(in_features=IMG_SIZE ** 2, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=num_features * 2),
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=num_features, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=784),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, log_var):
        """
        samples a vector with a gaussian distribution N(mu, sigma)
        """
        standard_deviation = torch.exp(0.5 * log_var)
        # epsilon is sampled using a centered gaussian distribution
        epsilon = torch.randn_like(standard_deviation)
        return mu + (epsilon * standard_deviation)

    def forward(self, x):
        # compute encoding space distribution parameters
        parameters = self.encoder(x).view(-1, 2, self.num_features)

        # retrieve the mean `mu` and the log variance `log_var`
        mu = parameters[:, 0, :]
        log_var = parameters[:, 1, :]

        # sample the latent vector through the reparameterization trick
        sampled_latent_vector = self.reparameterize(mu, log_var)

        # decoding
        reconstruction = self.decoder(sampled_latent_vector)
        return reconstruction, mu, log_var
