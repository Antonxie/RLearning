# Copyright (c) 2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain
from typing import Optional

from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_optimizer

from legged_lab.rsl_rl.amp_networks import AMPDiscriminator


class PPOAMP(PPO):
    """Proximal Policy Optimization with Adversarial Motion Priors (AMP).

    This algorithm extends the standard PPO with a discriminator that learns to distinguish
    expert demonstrations from policy-generated motion, providing a reward signal that
    encourages the policy to imitate expert behavior.

    Reference:
        - Peng et al. "AMP: Adversarial Motion Priors for Style-Preserving Physics-Based
          Character Animation." SIGGRAPH Asia 2021.
    """

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        discriminator: AMPDiscriminator,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        optimizer: str = "adam",
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        device: str = "cpu",
        # AMP parameters
        disc_learning_rate: float = 1e-4,
        disc_max_grad_norm: float = 1.0,
        disc_obs_buffer_size: int = 100,
        grad_penalty_scale: float = 10.0,
        disc_trunk_weight_decay: float = 1e-4,
        disc_linear_weight_decay: float = 1e-2,
        style_reward_scale: float = 1.0,
        task_style_lerp: float = 0.0,
        loss_type: str = "LSGAN",
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        """Initialize the AMP-PPO algorithm.

        Args:
            actor: The actor (policy) network.
            critic: The critic (value function) network.
            storage: The rollout storage for collecting transitions.
            discriminator: The AMP discriminator network.
            num_learning_epochs: Number of epochs to train on each rollout.
            num_mini_batches: Number of mini-batches per epoch.
            clip_param: PPO clipping parameter.
            gamma: Discount factor.
            lam: GAE lambda parameter.
            value_loss_coef: Weight for value function loss.
            entropy_coef: Weight for entropy bonus.
            learning_rate: Learning rate for policy/critic optimizer.
            max_grad_norm: Maximum gradient norm for clipping.
            optimizer: Optimizer type ('adam' or 'sgd').
            use_clipped_value_loss: Whether to use clipped value loss.
            schedule: Learning rate schedule ('fixed' or 'adaptive').
            desired_kl: Target KL divergence for adaptive schedule.
            normalize_advantage_per_mini_batch: Normalize advantages per mini-batch.
            device: Device for computation.
            disc_learning_rate: Learning rate for discriminator.
            disc_max_grad_norm: Maximum gradient norm for discriminator.
            disc_obs_buffer_size: Size of discriminator observation buffer.
            grad_penalty_scale: Scale for gradient penalty loss.
            disc_trunk_weight_decay: Weight decay for discriminator trunk.
            disc_linear_weight_decay: Weight decay for discriminator linear layer.
            style_reward_scale: Scale for AMP style reward.
            task_style_lerp: Interpolation between task and style reward.
            loss_type: Discriminator loss type ('LSGAN', 'GAN', or 'WGAN').
            rnd_cfg: Configuration for RND intrinsic reward.
            symmetry_cfg: Configuration for symmetry augmentation.
            multi_gpu_cfg: Configuration for distributed training.
        """
        self.discriminator = discriminator
        self.disc_learning_rate = disc_learning_rate
        self.disc_max_grad_norm = disc_max_grad_norm
        self.disc_obs_buffer_size = disc_obs_buffer_size
        self.grad_penalty_scale = grad_penalty_scale
        self.disc_trunk_weight_decay = disc_trunk_weight_decay
        self.disc_linear_weight_decay = disc_linear_weight_decay
        self.style_reward_scale = style_reward_scale
        self.task_style_lerp = task_style_lerp
        self.loss_type = loss_type

        self.discriminator_optimizer = optim.Adam(
            chain(
                [{"params": discriminator.trunk.parameters(), "weight_decay": disc_trunk_weight_decay}],
                [{"params": discriminator.linear.parameters(), "weight_decay": disc_linear_weight_decay}]
            ),
            lr=disc_learning_rate,
        )

        self.disc_obs_buffer = []
        self.num_disc_updates = 0

        super().__init__(
            actor=actor,
            critic=critic,
            storage=storage,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            clip_param=clip_param,
            gamma=gamma,
            lam=lam,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            optimizer=optimizer,
            use_clipped_value_loss=use_clipped_value_loss,
            schedule=schedule,
            desired_kl=desired_kl,
            normalize_advantage_per_mini_batch=normalize_advantage_per_mini_batch,
            device=device,
            rnd_cfg=rnd_cfg,
            symmetry_cfg=symmetry_cfg,
            multi_gpu_cfg=multi_gpu_cfg,
        )

    def compute_amp_reward(
        self,
        policy_obs: torch.Tensor,
        policy_next_obs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute AMP reward for the current batch.

        Args:
            policy_obs: Current observations from policy
            policy_next_obs: Next observations from policy

        Returns:
            AMP reward tensor
        """
        return self.discriminator.compute_reward(policy_obs, policy_next_obs)

    def update_discriminator(
        self,
        policy_obs: torch.Tensor,
        policy_next_obs: torch.Tensor,
        expert_obs: torch.Tensor,
        expert_next_obs: torch.Tensor,
    ) -> dict[str, float]:
        """Update the discriminator network.

        Args:
            policy_obs: Policy current observations
            policy_next_obs: Policy next observations
            expert_obs: Expert current observations
            expert_next_obs: Expert next observations

        Returns:
            Dictionary of discriminator losses
        """
        self.discriminator.train()

        policy_logits = self.discriminator(policy_obs, policy_next_obs)
        expert_logits = self.discriminator(expert_obs, expert_next_obs)

        if self.loss_type == "LSGAN":
            expert_loss = torch.nn.functional.mse_loss(
                expert_logits, torch.ones_like(expert_logits)
            )
            policy_loss = torch.nn.functional.mse_loss(
                policy_logits, torch.zeros_like(policy_logits)
            )
            disc_loss = expert_loss + policy_loss

            if self.grad_penalty_scale > 0:
                from legged_lab.rsl_rl.amp_networks import AMPLoss
                gp = AMPLoss.gradient_penalty(
                    self.discriminator,
                    policy_obs, policy_next_obs,
                    expert_obs, expert_next_obs,
                    self.device
                )
                disc_loss += self.grad_penalty_scale * gp

        elif self.loss_type == "WGAN":
            disc_loss = expert_logits.mean() - policy_logits.mean()
            if self.grad_penalty_scale > 0:
                from legged_lab.rsl_rl.amp_networks import AMPLoss
                gp = AMPLoss.gradient_penalty(
                    self.discriminator,
                    policy_obs, policy_next_obs,
                    expert_obs, expert_next_obs,
                    self.device
                )
                disc_loss += self.grad_penalty_scale * gp

        else:
            expert_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                expert_logits, torch.ones_like(expert_logits)
            )
            policy_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                policy_logits, torch.zeros_like(policy_logits)
            )
            disc_loss = expert_loss + policy_loss

        self.discriminator_optimizer.zero_grad()
        disc_loss.backward()
        nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.disc_max_grad_norm)
        self.discriminator_optimizer.step()

        self.num_disc_updates += 1

        with torch.no_grad():
            expert_acc = (torch.sigmoid(expert_logits) > 0.5).float().mean().item()
            policy_acc = (torch.sigmoid(policy_logits) < 0.5).float().mean().item()

        return {
            "disc_loss": disc_loss.item(),
            "expert_accuracy": expert_acc,
            "policy_accuracy": policy_acc,
        }

    def update(self) -> dict[str, float]:
        """Run optimization epoch and return mean losses including AMP discriminator."""
        losses = super().update()

        disc_losses = {"disc_loss": 0.0, "expert_accuracy": 0.0, "policy_accuracy": 0.0}

        if len(self.disc_obs_buffer) >= 2:
            batch = self.disc_obs_buffer[-1]
            disc_update_result = self.update_discriminator(
                batch["policy_obs"],
                batch["policy_next_obs"],
                batch["expert_obs"],
                batch["expert_next_obs"],
            )
            disc_losses = disc_update_result

        losses.update(disc_losses)
        return losses


def resolve_amp_discriminator_config(disc_cfg: dict, obs_dim: int, device: str) -> AMPDiscriminator:
    """Resolve and create the AMP discriminator from configuration.

    Args:
        disc_cfg: Discriminator configuration dictionary
        obs_dim: Observation dimension
        device: Device for computation

    Returns:
        Initialized AMPDiscriminator
    """
    hidden_dims = disc_cfg.get("hidden_dims", [1024, 512])
    activation = disc_cfg.get("activation", "elu")
    reward_scale = disc_cfg.get("reward_scale", 1.0)
    loss_type = disc_cfg.get("loss_type", "LSGAN")

    discriminator = AMPDiscriminator(
        input_dim=obs_dim * 2,
        hidden_layer_sizes=hidden_dims,
        activation=activation,
        reward_scale=reward_scale,
        device=device,
        loss_type=loss_type,
    )

    return discriminator
