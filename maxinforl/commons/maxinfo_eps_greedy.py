import collections
import copy

import torch
from torch.nn import functional as F
from stable_baselines3.sac import SAC
from typing import Optional, Union, Dict, Type, Tuple
from gymnasium import spaces
import numpy as np
import torch as th
from maxinforl.models.ensembles import EnsembleMLP, Normalizer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import Schedule, RolloutReturn, \
    TrainFreq, MaybeCallback, TrainFrequencyUnit
from stable_baselines3.common.utils import obs_as_tensor, should_collect_more_steps
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.utils import get_schedule_fn, polyak_update
from maxinforl.commons.utils import BaseIntrinsicReward
from stable_baselines3.common.noise import ActionNoise


class MaxInfoEpsGreedy(SAC):
    def __init__(self,
                 ensemble_model_kwargs: Dict,
                 intrinsic_reward_weights: Optional[Dict] = None,
                 agg_intrinsic_reward: str = 'sum',
                 normalize_ensemble_training: bool = True,
                 pred_diff: bool = True,
                 intrinsic_reward_model: Optional[Type[BaseIntrinsicReward]] = None,
                 exploration_freq: Union[int, Schedule] = 5,
                 *args,
                 **kwargs
                 ):
        self.normalize_ensemble_training = normalize_ensemble_training
        self.pred_diff = pred_diff
        self.exploration_freq = exploration_freq
        self.exploration_ent_coef_optimizer: Optional[th.optim.Adam] = None
        self.exploration_log_ent_coef = None
        super().__init__(*args, **kwargs)
        self._setup_ensemble_model(
            ensemble_model_kwargs=ensemble_model_kwargs,
            intrinsic_reward_weights=intrinsic_reward_weights,
            intrinsic_reward_model=intrinsic_reward_model,
            device=self.device,
            agg_intrinsic_reward=agg_intrinsic_reward,
        )
        self._setup_exploration_freq_schedule()

    def _setup_exploration_freq_schedule(self) -> None:
        """Transform to callable if needed."""
        self.exploration_freq_fn = get_schedule_fn(self.exploration_freq)

    def _update_exploration_weight(self) -> None:
        self.exploration_freq = int(self.exploration_freq_fn(self._current_progress_remaining))

    def _setup_normalizer(self, input_dim: int, output_dict: Dict, device: th.device):
        self.input_normalizer = Normalizer(input_dim=input_dim, update=self.normalize_ensemble_training,
                                           device=device)
        output_normalizers = {}
        for key, val in output_dict.items():
            output_normalizers[key] = Normalizer(input_dim=val.shape[-1], update=self.normalize_ensemble_training,
                                                 device=device)
        self.output_normalizers = output_normalizers

    def _setup_ensemble_model(self,
                              ensemble_model_kwargs: Dict,
                              intrinsic_reward_weights: Dict,
                              intrinsic_reward_model: Union[Type[BaseIntrinsicReward], None],
                              device: th.device,
                              agg_intrinsic_reward: str = 'sum',
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
        self.ensemble_model = EnsembleMLP(
            input_dim=input_dim,
            output_dict=output_dict,
            **ensemble_model_kwargs,
        )
        self.ensemble_model.to(device)
        self._setup_normalizer(input_dim=input_dim, output_dict=output_dict, device=device)

        if intrinsic_reward_weights is not None:
            assert intrinsic_reward_weights.keys() == output_dict.keys()
        else:
            intrinsic_reward_weights = {k: 1.0 for k in output_dict.keys()}

        if intrinsic_reward_model:
            self.intrinsic_reward_model = intrinsic_reward_model(
                intrinsic_reward_weights=intrinsic_reward_weights,
                ensemble_model=self.ensemble_model,
                agg_intrinsic_reward=agg_intrinsic_reward,
            )
        else:
            self.intrinsic_reward_model = None

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
            return {
                'next_obs': next_obs - obs,
                'reward': rewards,
            }
        else:
            if isinstance(next_obs, np.ndarray) or isinstance(next_obs, dict):
                next_obs = obs_as_tensor(next_obs, self.device)
            next_obs = self.extract_features(next_obs)
            return {
                'next_obs': next_obs,
                'reward': rewards,
            }

    def get_intrinsic_reward(self, inp: th.Tensor, labels: Dict) -> th.Tensor:
        # calculate intrinsic reward
        if self.intrinsic_reward_model is None:
            return th.zeros(inp.shape[0], device=self.device)
        else:
            return self.intrinsic_reward_model(inp=inp, labels=labels)

    def _setup_model(self) -> None:
        super()._setup_model()
        self.exploration_policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            **self.policy_kwargs,
        )
        self.exploration_policy.to(self.device)
        self._create_exploration_aliases()

        self.exploration_batch_norm_stats = copy.deepcopy(self.batch_norm_stats)
        self.exploration_batch_norm_stats_target = copy.deepcopy(self.batch_norm_stats_target)
        self.exploration_target_entropy = copy.deepcopy(self.target_entropy)

        if self.ent_coef_optimizer is not None:
            self.exploration_log_ent_coef = copy.deepcopy(self.log_ent_coef)
            self.exploration_ent_coef_optimizer = th.optim.Adam([self.exploration_log_ent_coef],
                                                                lr=self.lr_schedule(1))
        else:
            self.exploration_ent_coef_tensor = copy.deepcopy(self.ent_coef_tensor)

    def _create_exploration_aliases(self):
        self.exploration_actor = self.exploration_policy.actor
        self.exploration_critic = self.exploration_policy.critic
        self.exploration_critic_target = self.exploration_policy.critic_target

    def predict_with_exploration_policy(
            self,
            observation: Union[np.ndarray, Dict[str, np.ndarray]],
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:

        return self.exploration_policy.predict(observation, state, episode_start, deterministic)

    def _sample_action_from_exploration_policy(
            self,
            learning_starts: int,
            action_noise: Optional[ActionNoise] = None,
            n_envs: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:

        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # Warmup phase
            unscaled_action = np.array([self.action_space.sample() for _ in range(n_envs)])
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            assert self._last_obs is not None, "self._last_obs was not set"
            unscaled_action, _ = self.predict_with_exploration_policy(self._last_obs, deterministic=False)

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, spaces.Box):
            scaled_action = self.exploration_policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.exploration_policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action

    def collect_rollouts_with_explore_exploit(
            self,
            env: VecEnv,
            callback: BaseCallback,
            train_freq: TrainFreq,
            replay_buffer: ReplayBuffer,
            action_noise: Optional[ActionNoise] = None,
            learning_starts: int = 0,
            log_interval: Optional[int] = None,
            exploit: bool = True,
    ) -> RolloutReturn:

        if exploit:
            return self.collect_rollouts(
                env=env,
                callback=callback,
                train_freq=train_freq,
                replay_buffer=replay_buffer,
                action_noise=action_noise,
                learning_starts=learning_starts,
                log_interval=log_interval,
            )

        # if self.verbose:
        #    print('Collecting data with exploration policy')
        # Switch to eval mode (this affects batch norm / dropout)

        self.exploration_policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        if self.use_sde:
            self.exploration_actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True
        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.exploration_actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action_from_exploration_policy(
                learning_starts, action_noise, env.num_envs
            )

            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if not callback.on_step():
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes,
                                     continue_training=False)

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones,
                                   infos)  # type: ignore[arg-type]

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self._dump_logs()
        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        self.exploration_policy.set_training_mode(True)
        self.ensemble_model.train(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer,
                      self.exploration_actor.optimizer, self.exploration_critic.optimizer
                      ]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer, self.exploration_ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        exploration_ent_coef_losses, exploration_ent_coefs = [], []
        exploration_actor_losses, exploration_critic_losses = [], []
        batch_intrinsic_reward = []

        ensemble_losses = collections.defaultdict(list)

        self._update_ensemble_normalizers(batch_size)

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()
                self.exploration_actor.noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

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

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
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
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Lots of repeated code for training exploration agent

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.exploration_actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.exploration_ent_coef_optimizer is not None and self.exploration_log_ent_coef is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.exploration_log_ent_coef.detach())
                ent_coef_loss = -(self.exploration_log_ent_coef * (log_prob +
                                                                   self.exploration_target_entropy).detach()).mean()
                exploration_ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.exploration_ent_coef_tensor

            exploration_ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None and self.exploration_ent_coef_optimizer is not None:
                self.exploration_ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.exploration_ent_coef_optimizer.step()

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.exploration_actor.action_log_prob(replay_data.next_observations)
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(self.exploration_critic_target(
                    replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                # relabel reward with exploration reward
                labels = self._get_ensemble_targets(replay_data.next_observations, replay_data.observations,
                                                    replay_data.rewards)
                features = self.extract_features(replay_data.observations)
                inp = th.cat([features, replay_data.actions], dim=-1)
                # normalize inputs when gradient step == normalization_index which is randomly sampled.
                # if gradient_step == normalization_index:
                #    self.input_normalizer.update(inp)
                inp = self.input_normalizer.normalize(inp)
                for key, y in labels.items():
                    # if gradient_step == normalization_index:
                    #    self.output_normalizers[key].update(y)
                    labels[key] = self.output_normalizers[key].normalize(y)
                rewards = self.get_intrinsic_reward(
                    inp=inp,
                    labels=labels
                ).reshape(-1, 1)
                target_q_values = rewards + (1 - replay_data.dones) * self.gamma * next_q_values

                batch_intrinsic_reward.append(rewards.mean().item())

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.exploration_critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)  # for type checker
            exploration_critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

            # Optimize the critic
            self.exploration_critic.optimizer.zero_grad()
            critic_loss.backward()
            self.exploration_critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Min over all critic networks
            q_values_pi = th.cat(self.exploration_critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            exploration_actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.exploration_actor.optimizer.zero_grad()
            actor_loss.backward()
            self.exploration_actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

                # repeat for exploration critic
                polyak_update(self.exploration_critic.parameters(), self.exploration_critic_target.parameters(),
                              self.tau)
                polyak_update(self.exploration_batch_norm_stats, self.exploration_batch_norm_stats_target, 1.0)

            # ensemble model training
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

        self._n_updates += gradient_steps
        self.ensemble_model.train(False)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/exploration_ent_coefs", np.mean(exploration_ent_coefs))
        self.logger.record("train/exploration_actor_losses", np.mean(exploration_actor_losses))
        self.logger.record("train/exploration_critic_losses", np.mean(exploration_critic_losses))
        for key, val in ensemble_losses.items():
            self.logger.record(f"train/ensemble_loss_{key}", np.mean(val))
            self.logger.record(f"train/out_normalizer_mean_{key}", np.mean(
                self.output_normalizers[key].mean.cpu().numpy()))
            self.logger.record(f"train/out_normalizer_std_{key}", np.mean(
                self.output_normalizers[key].std.cpu().numpy()))
        self.logger.record(f"train/inp_normalizer_mean", np.mean(self.input_normalizer.mean.cpu().numpy()))
        self.logger.record(f"train/inp_normalizer_std", np.mean(self.input_normalizer.std.cpu().numpy()))
        self.logger.record("train/batch_intrinsic_reward", np.mean(batch_intrinsic_reward))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
            self.logger.record("train/exploration_ent_coef_losses", np.mean(exploration_ent_coef_losses))

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
            tb_log_name: str = "run",
            reset_num_timesteps: bool = True,
            progress_bar: bool = False,
    ):
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None, "You must set the environment before calling learn()"
        assert isinstance(self.train_freq, TrainFreq)  # check done in _setup_learn()

        rounds = 0
        exploit_steps = 0
        while self.num_timesteps < total_timesteps:
            self._update_exploration_weight()
            if self.exploration_freq > 0:
                exploit = not (rounds % self.exploration_freq == 0)
            else:
                exploit = True

            prev_steps = self.num_timesteps

            # print(self.exploration_freq, self._current_progress_remaining, exploit)
            rollout = self.collect_rollouts_with_explore_exploit(
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
                exploit=exploit,
            )

            if not rollout.continue_training:
                break

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else rollout.episode_timesteps
                # Special case when the user passes `gradient_steps=0`
                if gradient_steps > 0:
                    self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)

            rounds += 1
            exploit_steps += exploit * (self.num_timesteps - prev_steps)
            self.logger.record("time/exploitation_steps", exploit_steps)
            self.logger.record("time/exploration_steps", self.num_timesteps - exploit_steps)

        callback.on_training_end()

        return self