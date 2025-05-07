import torch
import torch.nn as nn
from typing import Tuple, Dict, List, Optional

from protomotions.agents.common.mlp import MLP

class LangWBCModel(nn.Module):
    def __init__(
        self,
        obs_size: int,
        action_size: int,
        history_len: int,
        text_embedding_dim: int,
        latent_dim: int,
        encoder_hidden_dims: List[int],
        decoder_hidden_dims: List[int],
        **kwargs, # Catch-all for unused args from config
    ):
        super().__init__()
        self.obs_size = obs_size
        self.action_size = action_size
        self.history_len = history_len
        self.text_embedding_dim = text_embedding_dim
        self.latent_dim = latent_dim

        # --- Encoder ---
        # Input: Flattened history (history_len * obs_size) + text embedding
        encoder_input_dim = (history_len * obs_size) + text_embedding_dim
        # Output: mu and logvar for the latent space (2 * latent_dim)
        self.encoder = MLP(
            input_dim=encoder_input_dim,
            output_dim=latent_dim * 2,
            hidden_dims=encoder_hidden_dims,
            activation="relu", # Or other activation
        )

        # --- Decoder ---
        # Input: Latent variable z + current observation
        decoder_input_dim = latent_dim + obs_size
        self.decoder = MLP(
            input_dim=decoder_input_dim,
            output_dim=action_size,
            hidden_dims=decoder_hidden_dims,
            activation="relu", # Or other activation
        )

        print(f"LangWBC CVAE Model Initialized:")
        print(f"  Observation Size (single step): {obs_size}")
        print(f"  Action Size: {action_size}")
        print(f"  History Length: {history_len}")
        print(f"  Text Embedding Dim: {text_embedding_dim}")
        print(f"  Latent Dim: {latent_dim}")
        print(f"  Encoder Input Dim: {encoder_input_dim}")
        print(f"  Decoder Input Dim: {decoder_input_dim}")


    def encode(self, obs_history: torch.Tensor, text_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes observation history and text embedding into latent distribution parameters.

        Args:
            obs_history: Tensor of shape (B, history_len * obs_size) or (B, history_len, obs_size)
            text_embedding: Tensor of shape (B, text_embedding_dim)

        Returns:
            Tuple of (mu, logvar), each of shape (B, latent_dim)
        """
        if obs_history.dim() == 3: # If input is (B, hist_len, obs_size)
             obs_history = obs_history.view(obs_history.size(0), -1) # Flatten history

        encoder_input = torch.cat([obs_history, text_embedding], dim=-1)
        mu_logvar = self.encoder(encoder_input)
        mu, logvar = torch.chunk(mu_logvar, 2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Samples from the latent distribution using the reparameterization trick.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, current_obs: torch.Tensor) -> torch.Tensor:
        """
        Decodes latent variable and current observation into an action.

        Args:
            z: Latent variable tensor of shape (B, latent_dim)
            current_obs: Tensor of shape (B, obs_size)

        Returns:
            Action tensor of shape (B, action_size)
        """
        decoder_input = torch.cat([z, current_obs], dim=-1)
        action = self.decoder(decoder_input)
        return action

    def forward(self, obs_history: torch.Tensor, text_embedding: torch.Tensor, current_obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for inference (uses mean of latent distribution).
        """
        mu, _ = self.encode(obs_history, text_embedding)
        # During inference, we typically use the mean (mu) for deterministic output
        action = self.decode(mu, current_obs)
        return action

    def get_action_and_latent(self, obs_history: torch.Tensor, text_embedding: torch.Tensor, current_obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for training (samples from latent distribution).
        Returns action, mu, logvar.
        """
        mu, logvar = self.encode(obs_history, text_embedding)
        z = self.reparameterize(mu, logvar)
        action = self.decode(z, current_obs)
        return action, mu, logvar

    @staticmethod
    def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Calculates the KL divergence loss: D_KL(N(mu, sigma) || N(0, I)).
        See Appendix B from VAE paper: Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        https://arxiv.org/abs/1312.6114
        """
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return kld # Shape: (B,)
