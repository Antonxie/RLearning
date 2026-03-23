# Copyright (c) 2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
from torch import autograd


class AMPActorCritic(nn.Module):
    """Actor-Critic network for AMP (Adversarial Motion Priors) that combines
    policy learning with discriminator-based motion imitation.

    This network outputs both actions (for the policy) and values (for the critic),
    while the discriminator is trained separately to distinguish expert from policy data.
    """

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims: list = [512, 256, 128],
        critic_hidden_dims: list = [512, 256, 128],
        activation: str = "elu",
        init_noise_std: float = 1.0,
    ) -> None:
        super().__init__()

        activation_fn = self._get_activation(activation)

        actor_layers = []
        actor_layers.append(nn.Linear(num_actor_obs, actor_hidden_dims[0]))
        actor_layers.append(activation_fn)
        for i in range(len(actor_hidden_dims) - 1):
            actor_layers.append(nn.Linear(actor_hidden_dims[i], actor_hidden_dims[i + 1]))
            actor_layers.append(activation_fn)
        actor_layers.append(nn.Linear(actor_hidden_dims[-1], num_actions))
        self.actor = nn.Sequential(*actor_layers)

        critic_layers = []
        critic_layers.append(nn.Linear(num_critic_obs, critic_hidden_dims[0]))
        critic_layers.append(activation_fn)
        for i in range(len(critic_hidden_dims) - 1):
            critic_layers.append(nn.Linear(critic_hidden_dims[i], critic_hidden_dims[i + 1]))
            critic_layers.append(activation_fn)
        critic_layers.append(nn.Linear(critic_hidden_dims[-1], 1))
        self.critic = nn.Sequential(*critic_layers)

        self.log_std = nn.Parameter(torch.ones(num_actions) * torch.log(torch.tensor(init_noise_std)))
        self.num_actions = num_actions

    def _get_activation(self, name: str) -> nn.Module:
        if name == "elu":
            return nn.ELU()
        elif name == "relu":
            return nn.ReLU()
        elif name == "tanh":
            return nn.Tanh()
        elif name == "leaky_relu":
            return nn.LeakyReLU()
        else:
            return nn.ELU()

    def forward(self, actor_obs: torch.Tensor, critic_obs: torch.Tensor) -> tuple:
        """Forward pass returning both action mean and value estimate."""
        action_mean = self.actor(actor_obs)
        value = self.critic(critic_obs)
        return action_mean, value

    def act(self, obs: torch.Tensor, stochastic: bool = True) -> torch.Tensor:
        """Sample action from the policy."""
        action_mean, _ = self.forward(obs, obs)
        if stochastic:
            action_std = torch.exp(self.log_std)
            dist = torch.distributions.Normal(action_mean, action_std)
            return dist.sample()
        return action_mean


class AMPDiscriminator(nn.Module):
    """Discriminator network for AMP that distinguishes expert demonstrations from policy data.

    The discriminator is trained to maximize the log probability of distinguishing expert data
    from policy-generated data, and provides a reward signal for the policy to imitate expert behavior.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layer_sizes: list = [1024, 512],
        activation: str = "elu",
        reward_scale: float = 1.0,
        device: str = "cuda:0",
        loss_type: str = "LSGAN",
        use_minibatch_std: bool = True,
        empirical_normalization: bool = False,
    ) -> None:
        super().__init__()

        self.device = torch.device(device)
        self.input_dim = input_dim
        self.reward_scale = reward_scale
        self.use_minibatch_std = use_minibatch_std
        self.loss_type = loss_type

        activation_fn = self._get_activation(activation)

        layers = []
        curr_in_dim = input_dim
        for hidden_dim in hidden_layer_sizes:
            layers.append(nn.Linear(curr_in_dim, hidden_dim))
            layers.append(activation_fn)
            curr_in_dim = hidden_dim

        self.trunk = nn.Sequential(*layers)

        final_in_dim = hidden_layer_sizes[-1]
        if use_minibatch_std:
            final_in_dim += 1
        self.linear = nn.Linear(final_in_dim, 1)

        self.empirical_normalization = empirical_normalization
        amp_obs_dim = input_dim // 2
        if empirical_normalization:
            self.amp_normalizer = nn.LayerNorm(amp_obs_dim)
        else:
            self.amp_normalizer = nn.Identity()

        self.to(self.device)

    def _get_activation(self, name: str) -> nn.Module:
        if name == "elu":
            return nn.ELU()
        elif name == "relu":
            return nn.ReLU()
        elif name == "tanh":
            return nn.Tanh()
        elif name == "leaky_relu":
            return nn.LeakyReLU()
        else:
            return nn.ELU()

    def forward(self, observations: torch.Tensor, next_observations: torch.Tensor) -> torch.Tensor:
        """Compute discriminator scores for concatenated observations.

        Args:
            observations: Current observations [batch_size, obs_dim]
            next_observations: Next observations [batch_size, obs_dim]

        Returns:
            Discriminator logits [batch_size, 1]
        """
        obs_combined = torch.cat([observations, next_observations], dim=-1)

        if self.empirical_normalization:
            amp_obs_dim = self.input_dim // 2
            obs_combined[:, :amp_obs_dim] = self.amp_normalizer(obs_combined[:, :amp_obs_dim])
            obs_combined[:, amp_obs_dim:] = self.amp_normalizer(obs_combined[:, amp_obs_dim:])

        features = self.trunk(obs_combined)

        if self.use_minibatch_std:
            batch_size = features.shape[0]
            minibatch_std = features.std(dim=0, keepdim=True).mean()
            minibatch_std = minibatch_std.expand(batch_size, 1)
            features = torch.cat([features, minibatch_std], dim=-1)

        logits = self.linear(features)
        return logits

    def compute_reward(
        self, observations: torch.Tensor, next_observations: torch.Tensor
    ) -> torch.Tensor:
        """Compute AMP reward from observations.

        The reward is based on the discriminator's ability to distinguish expert from policy data.
        Higher reward indicates more expert-like behavior.

        Args:
            observations: Current observations
            next_observations: Next observations

        Returns:
            AMP reward tensor
        """
        with torch.no_grad():
            logits = self.forward(observations, next_observations)
            if self.loss_type == "LSGAN":
                probability = torch.sigmoid(logits)
                reward = torch.log(probability + 1e-8) - torch.log(1 - probability + 1e-8)
            elif self.loss_type == "WGAN":
                reward = logits.squeeze(-1)
            else:
                reward = -logits.squeeze(-1)
            return reward * self.reward_scale


class AMPLoss:
    """AMP loss functions for training the discriminator."""

    @staticmethod
    def gradient_penalty(
        discriminator: nn.Module,
        observations: torch.Tensor,
        next_observations: torch.Tensor,
        expert_observations: torch.Tensor,
        expert_next_observations: torch.Tensor,
        device: str = "cuda:0",
    ) -> torch.Tensor:
        """Compute gradient penalty for WGAN-GP stability.

        Args:
            discriminator: The discriminator network
            observations: Policy observations
            next_observations: Policy next observations
            expert_observations: Expert observations
            expert_next_observations: Expert next observations
            device: Device for computation

        Returns:
            Gradient penalty loss
        """
        batch_size = observations.shape[0]
        alpha = torch.rand(batch_size, 1).to(device)

        expert_data = torch.cat([expert_observations, expert_next_observations], dim=-1)
        policy_data = torch.cat([observations, next_observations], dim=-1)

        interpolated = alpha * expert_data + (1 - alpha) * policy_data
        interpolated.requires_grad_(True)

        disc_interpolated = discriminator(
            interpolated[:, :observations.shape[1]],
            interpolated[:, observations.shape[1]:]
        )

        gradients = autograd.grad(
            outputs=disc_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(disc_interpolated),
            create_graph=True,
            retain_graph=True,
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()

        return gradient_penalty

    @staticmethod
    def bce_loss(discriminator_output: torch.Tensor, is_expert: bool) -> torch.Tensor:
        """Binary cross-entropy loss for standard GAN training.

        Args:
            discriminator_output: Output of discriminator
            is_expert: Whether the data is expert data (True) or policy data (False)

        Returns:
            BCE loss
        """
        target = torch.ones_like(discriminator_output) if is_expert else torch.zeros_like(discriminator_output)
        return torch.nn.functional.binary_cross_entropy_with_logits(discriminator_output, target)

    @staticmethod
    def least_squares_loss(discriminator_output: torch.Tensor, is_expert: bool) -> torch.Tensor:
        """Least squares loss for LSGAN training.

        Args:
            discriminator_output: Output of discriminator
            is_expert: Whether the data is expert data (True) or policy data (False)

        Returns:
            Least squares loss
        """
        if is_expert:
            target = torch.ones_like(discriminator_output)
        else:
            target = torch.zeros_like(discriminator_output)
        return torch.nn.functional.mse_loss(discriminator_output, target)
