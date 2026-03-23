# Copyright (c) 2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import time
import torch

from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups
from rsl_rl.utils.logger import Logger

from legged_lab.rsl_rl.amp_algorithm import PPOAMP, resolve_amp_discriminator_config


class AMPOnPolicyRunner:
    """On-policy runner for AMP (Adversarial Motion Priors) reinforcement learning.

    This runner extends the standard PPO runner with discriminator-based AMP training,
    allowing physics-based character animation that mimics expert motion data while
    learning to perform desired tasks (e.g., velocity tracking).
    """

    alg: PPOAMP

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        """Construct the runner, algorithm, and logging stack.

        Args:
            env: The vectorized environment.
            train_cfg: Training configuration dictionary.
            log_dir: Directory for logging outputs.
            device: Device for computation ('cpu' or 'cuda').
        """
        self.env = env
        self.cfg = train_cfg
        self.device = device

        self._configure_multi_gpu()

        obs = self.env.get_observations()

        alg_class = resolve_callable(self.cfg["algorithm"]["class_name"])
        self.alg = self._construct_algorithm(obs, self.env, self.cfg, self.device)

        self.logger = Logger(
            log_dir=log_dir,
            cfg=self.cfg,
            env_cfg=self.env.cfg,
            num_envs=self.env.num_envs,
            is_distributed=self.is_distributed,
            gpu_world_size=self.gpu_world_size,
            gpu_global_rank=self.gpu_global_rank,
            device=self.device,
        )

        self.current_learning_iteration = 0

    def _configure_multi_gpu(self) -> None:
        """Configure multi-GPU training if enabled."""
        self.is_distributed = False
        self.gpu_world_size = 1
        self.gpu_global_rank = 0

    def _construct_algorithm(self, obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> PPOAMP:
        """Construct the AMP-PPO algorithm with discriminator.

        Args:
            obs: Observations from the environment.
            env: The vectorized environment.
            cfg: Training configuration.
            device: Device for computation.

        Returns:
            Initialized PPOAMP algorithm.
        """
        alg_class = resolve_callable(cfg["algorithm"].pop("class_name"))

        actor_class = resolve_callable(cfg["actor"].pop("class_name"))
        critic_class = resolve_callable(cfg["critic"].pop("class_name"))

        default_sets = ["actor", "critic"]
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        actor_obs_groups = cfg["obs_groups"].get("actor", ["actor"])
        critic_obs_groups = cfg["obs_groups"].get("critic", ["critic"])

        actor: MLPModel = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        print(f"Actor Model: {actor}")

        if cfg["algorithm"].pop("share_cnn_encoders", None):
            cfg["critic"]["cnns"] = actor.cnns

        critic: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        print(f"Critic Model: {critic}")

        storage = RolloutStorage(
            "rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device
        )

        amp_cfg = cfg["algorithm"].pop("amp_cfg", {})
        disc_obs_groups = cfg.get("obs_groups", {}).get("disc", ["disc"])
        unwrapped_env = env.unwrapped if hasattr(env, "unwrapped") else env
        if hasattr(unwrapped_env, "observation_manager"):
            disc_obs = unwrapped_env.observation_manager.compute_group("disc")
            if isinstance(disc_obs, dict):
                disc_obs_dim = disc_obs.get("disc", disc_obs).shape[-1]
            else:
                disc_obs_dim = disc_obs.shape[-1]
        else:
            disc_obs_dim = obs.shape[-1]
        print(f"[INFO] Discriminator observation dimension: {disc_obs_dim}")

        discriminator = resolve_amp_discriminator_config(
            amp_cfg.get("amp_discriminator", {}),
            obs_dim=disc_obs_dim,
            device=device,
        )

        disc_learning_rate = amp_cfg.get("disc_learning_rate", 1e-4)
        disc_max_grad_norm = amp_cfg.get("disc_max_grad_norm", 1.0)
        disc_obs_buffer_size = amp_cfg.get("disc_obs_buffer_size", 100)
        grad_penalty_scale = amp_cfg.get("grad_penalty_scale", 10.0)
        disc_trunk_weight_decay = amp_cfg.get("disc_trunk_weight_decay", 1e-4)
        disc_linear_weight_decay = amp_cfg.get("disc_linear_weight_decay", 1e-2)
        style_reward_scale = amp_cfg.get("amp_discriminator", {}).get("style_reward_scale", 1.0)
        task_style_lerp = amp_cfg.get("amp_discriminator", {}).get("task_style_lerp", 0.0)
        loss_type = amp_cfg.get("loss_type", "LSGAN")

        alg: PPOAMP = alg_class(
            actor=actor,
            critic=critic,
            storage=storage,
            discriminator=discriminator,
            device=device,
            disc_learning_rate=disc_learning_rate,
            disc_max_grad_norm=disc_max_grad_norm,
            disc_obs_buffer_size=disc_obs_buffer_size,
            grad_penalty_scale=grad_penalty_scale,
            disc_trunk_weight_decay=disc_trunk_weight_decay,
            disc_linear_weight_decay=disc_linear_weight_decay,
            style_reward_scale=style_reward_scale,
            task_style_lerp=task_style_lerp,
            loss_type=loss_type,
            multi_gpu_cfg=cfg.get("multi_gpu"),
            **cfg["algorithm"],
        )

        if hasattr(alg, 'symmetry') and alg.symmetry is not None:
            alg.symmetry["_env"] = env

        return alg

    def _get_discriminator_observations(self) -> dict:
        """Get discriminator observations from the environment.

        Returns:
            Dictionary containing discriminator observations for policy and expert.
        """
        disc_obs = {"policy_obs": None, "expert_obs": None}
        unwrapped_env = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
        if hasattr(unwrapped_env, "observation_manager"):
            try:
                result = unwrapped_env.observation_manager.compute_group("disc")
                if isinstance(result, dict):
                    disc_obs["policy_obs"] = result.get("disc")
                else:
                    disc_obs["policy_obs"] = result
            except Exception as e:
                print(f"[WARN] Failed to get disc obs: {e}")
            try:
                result = unwrapped_env.observation_manager.compute_group("disc_demo")
                if isinstance(result, dict):
                    disc_obs["expert_obs"] = result.get("disc_demo")
                else:
                    disc_obs["expert_obs"] = result
            except Exception as e:
                print(f"[WARN] Failed to get disc_demo obs: {e}")
        else:
            print("[WARN] env.unwrapped has no observation_manager")
        return disc_obs

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Run the learning loop for the specified number of iterations.

        Args:
            num_learning_iterations: Number of training iterations to run.
            init_at_random_ep_len: Whether to randomize initial episode lengths.
        """
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()

        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        self.logger.init_logging_writer()

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations

        disc_obs_buffer = []
        prev_disc_policy_obs = None
        prev_disc_expert_obs = None

        for it in range(start_it, total_it):
            start = time.time()
            prev_policy_obs_for_amp = None

            with torch.inference_mode():
                for step_idx in range(self.cfg["num_steps_per_env"]):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))

                    disc_obs = self._get_discriminator_observations()
                    if disc_obs["policy_obs"] is not None and disc_obs["expert_obs"] is not None:
                        curr_policy_obs = disc_obs["policy_obs"].to(self.device)

                        if prev_policy_obs_for_amp is not None:
                            amp_rewards = self.alg.compute_amp_reward(
                                prev_policy_obs_for_amp,
                                curr_policy_obs
                            )
                            style_reward = amp_rewards.squeeze(-1) * self.alg.style_reward_scale
                            style_reward = style_reward[:rewards.shape[0]]
                            rewards = rewards + style_reward

                        prev_policy_obs_for_amp = curr_policy_obs

                    self.alg.process_env_step(obs, rewards, dones, extras)
                    self.logger.process_env_step(rewards, dones, extras, None)

                    if step_idx == 0:
                        disc_obs = self._get_discriminator_observations()
                        if disc_obs["policy_obs"] is not None and disc_obs["expert_obs"] is not None and prev_disc_policy_obs is not None and prev_disc_expert_obs is not None:
                            disc_obs_buffer.append({
                                "policy_obs": prev_disc_policy_obs,
                                "policy_next_obs": disc_obs["policy_obs"].to(self.device),
                                "expert_obs": prev_disc_expert_obs,
                                "expert_next_obs": disc_obs["expert_obs"].to(self.device),
                            })
                            if len(disc_obs_buffer) > self.alg.disc_obs_buffer_size:
                                disc_obs_buffer.pop(0)
                        if disc_obs["policy_obs"] is not None:
                            prev_disc_policy_obs = disc_obs["policy_obs"].to(self.device)
                        if disc_obs["expert_obs"] is not None:
                            prev_disc_expert_obs = disc_obs["expert_obs"].to(self.device)

            if len(disc_obs_buffer) >= 2:
                self.alg.disc_obs_buffer = disc_obs_buffer.copy()

            stop = time.time()
            collect_time = stop - start
            start = stop

            self.alg.compute_returns(obs)

            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=None,
            )

            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))

        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))
            self.logger.stop_logging_writer()

    def save(self, path: str, infos: dict | None = None) -> None:
        """Save the models and training state.

        Args:
            path: Path to save the model.
            infos: Additional information to save.
        """
        saved_dict = self.alg.save()
        saved_dict["iter"] = self.current_learning_iteration
        saved_dict["infos"] = infos
        torch.save(saved_dict, path)
        self.logger.save_model(path, self.current_learning_iteration)

    def load(
        self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None
    ) -> dict:
        """Load the models and training state.

        Args:
            path: Path to load the model from.
            load_cfg: Optional configuration for loading.
            strict: Whether to strictly enforce state dict matching.
            map_location: Device mapping for loading.

        Returns:
            Loaded information dictionary.
        """
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
        if load_iteration:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device: str | None = None) -> MLPModel:
        """Return the policy on the requested device for inference.

        Args:
            device: Device to move the policy to.

        Returns:
            Policy model for inference.
        """
        self.alg.eval_mode()
        return self.alg.get_policy().to(device)

    def add_git_repo_to_log(self, path: str) -> None:
        """Add git repository state to the log.

        Args:
            path: Path to the script being run.
        """
        pass
