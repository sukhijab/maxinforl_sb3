import collections
import copy

import torch
from torch.nn import functional as F
from stable_baselines3.sac import SAC
from typing import Optional, Union, Dict, Type
import numpy as np
import torch as th
from maxinforl_torch.models.ensembles import EnsembleMLP, Normalizer, EPS
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.utils import polyak_update
from maxinforl_torch.commons.utils import DisagreementIntrinsicReward
from stable_baselines3.common.type_aliases import MaybeCallback


class MaxInfoSAC(SAC):
    def __init__(self,
                 ensemble_model_kwargs: Dict,
                 ensemble_type: Type[torch.nn.Module] = EnsembleMLP,
                 intrinsic_reward_weights: Optional[Dict] = None,
                 normalize_ensemble_training: bool = True,
                 pred_diff: bool = True,
                 learn_rewards: bool = True,
                 normalize_dynamics_entropy: bool = True,
                 dyn_entropy_scale: Union[str, float] = 'auto',
                 init_dyn_entropy_scale: float = 1,
                 *args,
                 **kwargs
                 ):
        self.dyn_entropy_scale = dyn_entropy_scale
        self.normalize_ensemble_training = normalize_ensemble_training
        self.normalize_dynamics_entropy = normalize_dynamics_entropy
        self.pred_diff = pred_diff
        self.init_dyn_entropy_scale = init_dyn_entropy_scale
        self.learn_rewards = learn_rewards
        super().__init__(*args, **kwargs)
        self._setup_ensemble_model(
            ensemble_model_kwargs=ensemble_model_kwargs,
            intrinsic_reward_weights=intrinsic_reward_weights,
            device=self.device,
            agg_intrinsic_reward='mean',
            ensemble_type=ensemble_type,
        )
        # pick info gain with an epistemic uncertainty of 1.

    def _setup_model(self) -> None:
        super()._setup_model()
        self.actor_target = copy.deepcopy(self.actor)
        if self.dyn_entropy_scale == 'auto':
            assert self.init_dyn_entropy_scale > 0
            init_dyn_entropy_scale = self.init_dyn_entropy_scale
            self.log_dyn_entropy_scale = th.log(th.ones(1, device=self.device)
                                                * init_dyn_entropy_scale).requires_grad_(True)
            self.dyn_ent_scale_optimizer = th.optim.Adam([self.log_dyn_entropy_scale],
                                                         lr=self.lr_schedule(1))
            self.dyn_entropy_scale = None
        else:
            self.dyn_entropy_scale = th.tensor([self.dyn_entropy_scale]).squeeze()
            self.dyn_ent_scale_optimizer = None
            self.log_dyn_entropy_scale = None

    def _setup_normalizer(self, input_dim: int, output_dict: Dict, device: th.device):
        self.input_normalizer = Normalizer(input_dim=input_dim, update=self.normalize_ensemble_training,
                                           device=device)
        output_normalizers = {}
        for key, val in output_dict.items():
            output_normalizers[key] = Normalizer(input_dim=val.shape[-1], update=self.normalize_ensemble_training,
                                                 device=device)
        self.output_normalizers = output_normalizers
        self.entropy_normalizer = Normalizer(input_dim=1, update=self.normalize_dynamics_entropy,
                                             device=device)

    def _setup_ensemble_model(self,
                              ensemble_model_kwargs: Dict,
                              intrinsic_reward_weights: Dict,
                              device: th.device,
                              agg_intrinsic_reward: str = 'mean',
                              ensemble_type: Type[torch.nn.Module] = EnsembleMLP,
                              ) -> None:
        sample_obs = self.observation_space.sample()
        if isinstance(sample_obs, Dict):
            for key in sample_obs.keys():
                sample_obs[key] = np.expand_dims(sample_obs[key], 0)
        else:
            sample_obs = np.expand_dims(sample_obs, 0)
        dummy_feat = self.extract_features(
            obs_as_tensor(sample_obs,
                          self.device)
        )
        input_dim = dummy_feat.shape[-1] + self.action_space.shape[0]
        output_dict = self._get_ensemble_targets(sample_obs,
                                                 sample_obs,
                                                 rewards=torch.zeros((1, 1))
                                                 )

        self.ensemble_model = ensemble_type(
            input_dim=input_dim,
            output_dict=output_dict,
            use_entropy=True,
            **ensemble_model_kwargs,
        )

        self.ensemble_model.to(device)
        self._setup_normalizer(input_dim=input_dim, output_dict=output_dict, device=device)

        if intrinsic_reward_weights is not None:
            assert intrinsic_reward_weights.keys() == output_dict.keys()
        else:
            intrinsic_reward_weights = {k: 1.0 for k in output_dict.keys()}

        self.intrinsic_reward_model = DisagreementIntrinsicReward(
            intrinsic_reward_weights=intrinsic_reward_weights,
            ensemble_model=self.ensemble_model,
            agg_intrinsic_reward=agg_intrinsic_reward,
        )

    def extract_features(self, obs):
        with th.no_grad():
            features = self.actor.extract_features(
                obs, features_extractor=self.actor.features_extractor)
        return features

    def _get_ensemble_targets(self, next_obs: Union[th.Tensor, Dict], obs: Union[th.Tensor, Dict],
                              rewards: th.Tensor) -> Dict:
        if self.pred_diff:
            assert type(next_obs) == type(obs)
            if isinstance(next_obs, np.ndarray) or isinstance(next_obs, dict):
                next_obs = obs_as_tensor(next_obs, self.device)
                obs = obs_as_tensor(obs, self.device)
            next_obs = self.extract_features(next_obs)
            obs = self.extract_features(obs)
            if self.learn_rewards:
                return {
                    'next_obs': next_obs - obs,
                    'reward': rewards,
                }
            else:
                return {
                    'next_obs': next_obs - obs,
                }
        else:
            if isinstance(next_obs, np.ndarray) or isinstance(next_obs, dict):
                next_obs = obs_as_tensor(next_obs, self.device)
            next_obs = self.extract_features(next_obs)
            if self.learn_rewards:
                return {
                    'next_obs': next_obs,
                    'reward': rewards,
                }
            else:
                return {
                    'next_obs': next_obs,
                }

    def get_intrinsic_reward(self, inp: th.Tensor, labels: Dict) -> th.Tensor:
        # calculate intrinsic reward
        entropy = self.intrinsic_reward_model(inp=inp, labels=labels)
        if not self.ensemble_model.learn_std:
            info_gain = entropy - torch.log(torch.ones_like(entropy) * EPS)
            return info_gain
        else:
            return entropy

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        if self.dyn_ent_scale_optimizer is not None:
            optimizers += [self.dyn_ent_scale_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        dyn_scale_losses, dyn_scales = [], []
        actor_losses, critic_losses = [], []

        dynamics_info_gain, target_info_gain = [], []

        ensemble_losses = collections.defaultdict(list)

        self._update_ensemble_normalizers(batch_size)

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            features = self.extract_features(replay_data.observations)
            inp = th.cat([features, actions_pi], dim=-1)

            actions_target_pi, _ = self.actor_target.action_log_prob(replay_data.observations)
            target_inp = th.cat([features, actions_target_pi], dim=-1)

            total_inp = th.cat([inp, target_inp], dim=0)
            total_inp = self.input_normalizer.normalize(total_inp)
            dynamics_entropy = self.get_intrinsic_reward(
                inp=total_inp,
                labels=None,
            ).reshape(-1, 1)
            self.entropy_normalizer.update(dynamics_entropy.detach())
            dynamics_entropy = self.entropy_normalizer.normalize(dynamics_entropy)

            dynamics_entropy, target_dynamics_entropy = dynamics_entropy[:batch_size], \
                dynamics_entropy[batch_size:]
            # get target for entropy
            dynamics_info_gain.append(dynamics_entropy.mean().detach().item())
            target_info_gain.append(target_dynamics_entropy.mean().item())
            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            dyn_scale_loss = None
            if self.dyn_ent_scale_optimizer is not None and self.log_dyn_entropy_scale is not None:
                dyn_scale = th.exp(self.log_dyn_entropy_scale.detach())
                dyn_scale_loss = (self.log_dyn_entropy_scale * (
                        dynamics_entropy - target_dynamics_entropy).detach()).mean()
                dyn_scale_losses.append(dyn_scale_loss.item())
            else:
                dyn_scale = self.dyn_entropy_scale

            ent_coefs.append(ent_coef.item())
            dyn_scales.append(dyn_scale.item())

            total_entropy = dyn_scale * dynamics_entropy - ent_coef * log_prob
            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            if dyn_scale_loss is not None and self.dyn_ent_scale_optimizer is not None:
                self.dyn_ent_scale_optimizer.zero_grad()
                dyn_scale_loss.backward()
                self.dyn_ent_scale_optimizer.step()

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_features = self.extract_features(replay_data.next_observations)
                # get entropy of transitions
                inp = th.cat([next_features, next_actions], dim=-1)
                inp = self.input_normalizer.normalize(inp)
                next_obs_dynamics_entropy = self.get_intrinsic_reward(
                    inp=inp,
                    labels=None,
                ).reshape(-1, 1)
                next_obs_dynamics_entropy = self.entropy_normalizer.normalize(next_obs_dynamics_entropy)
                next_state_entropy = dyn_scale * next_obs_dynamics_entropy \
                                     - ent_coef * next_log_prob.reshape(-1, 1)
                # add entropy term
                next_q_values = next_q_values + next_state_entropy
                # td error + entropy term
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)  # for type checker
            critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Min over all critic networks
            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)

            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = -(total_entropy + min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

            # ensemble model training
            inp = th.cat([features, replay_data.actions], dim=-1)
            inp = self.input_normalizer.normalize(inp)

            labels = self._get_ensemble_targets(replay_data.next_observations, replay_data.observations,
                                                replay_data.rewards)
            for key, y in labels.items():
                # if gradient_step == normalization_index:
                #    self.output_normalizers[key].update(y)
                labels[key] = self.output_normalizers[key].normalize(y)
            self.ensemble_model.train()
            self.ensemble_model.optimizer.zero_grad()
            prediction = self.ensemble_model(inp)
            loss = self.ensemble_model.loss(prediction=prediction, target=labels)
            stacked_losses = []
            for key, val in loss.items():
                ensemble_losses[key].append(val.item())
                stacked_losses.append(val)
            stacked_losses = th.stack(stacked_losses)
            total_loss = stacked_losses.mean()
            total_loss.backward()
            self.ensemble_model.optimizer.step()
            self.ensemble_model.eval()

        self._n_updates += gradient_steps
        self.ensemble_model.train(False)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/dyn_scale", np.mean(dyn_scales))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        for key, val in ensemble_losses.items():
            self.logger.record(f"train/ensemble_loss_{key}", np.mean(val))
            self.logger.record(f"train/out_normalizer_mean_{key}", np.mean(
                self.output_normalizers[key].mean.cpu().numpy()))
            self.logger.record(f"train/out_normalizer_std_{key}", np.mean(
                self.output_normalizers[key].std.cpu().numpy()))
        self.logger.record(f"train/inp_normalizer_mean", np.mean(self.input_normalizer.mean.cpu().numpy()))
        self.logger.record(f"train/inp_normalizer_std", np.mean(self.input_normalizer.std.cpu().numpy()))
        self.logger.record("train/dynamics_info_gain", np.mean(dynamics_info_gain))
        self.logger.record("train/target_info_gain", np.mean(target_info_gain))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
        if len(dyn_scale_losses) > 0:
            self.logger.record("train/dyn_scale_losses", np.mean(dyn_scale_losses))

    def _update_ensemble_normalizers(self, batch_size: int):
        if self.replay_buffer.size() >= batch_size:
            replay_data = self.replay_buffer.sample(self.batch_size, env=self._vec_normalize_env)
            features = self.extract_features(replay_data.observations)
            inp = th.cat([features, replay_data.actions], dim=-1)
            self.input_normalizer.update(inp)

            labels = self._get_ensemble_targets(replay_data.next_observations, replay_data.observations,
                                                replay_data.rewards)
            for key, y in labels.items():
                self.output_normalizers[key].update(y)

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 4,
            tb_log_name: str = "MaxInfoSac",
            reset_num_timesteps: bool = True,
            progress_bar: bool = False,
    ):
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )