import torch
import torch as th
import numpy as np
from typing import Optional, Tuple, Dict, Union
from gymnasium import spaces
from stable_baselines3.sac.policies import MlpPolicy
from torch.distributions import Normal
from stable_baselines3.common.type_aliases import PyTorchObs


class OACPolicy(MlpPolicy):
    def __init__(self,
                 *args,
                 **kwargs):
        if 'beta_ub' in kwargs:
            beta_ub = kwargs.pop('beta_ub')
        else:
            beta_ub = 4.66
        if 'beta_lb' in kwargs:
            beta_lb = kwargs.pop('beta_lb')
        else:
            beta_lb = -3.65

        if 'shift_multiplier' in kwargs:
            shift_multiplier = kwargs.pop('shift_multiplier')
        else:
            shift_multiplier = 6.86

        self.beta_ub = beta_ub
        self.shift_multiplier = shift_multiplier
        self.beta_lb = beta_lb
        super().__init__(*args, **kwargs)

    def predict(
            self,
            observation: Union[np.ndarray, Dict[str, np.ndarray]],
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        # Check for common mistake that the user does not mix Gym/VecEnv API
        # Tuple obs are not supported by SB3, so we can safely do that check
        if isinstance(observation, tuple) and len(observation) == 2 and isinstance(observation[1], dict):
            raise ValueError(
                "You have passed a tuple to the predict() function instead of a Numpy array or a Dict. "
                "You are probably mixing Gym API with SB3 VecEnv API: `obs, info = env.reset()` (Gym) "
                "vs `obs = vec_env.reset()` (SB3 VecEnv). "
                "See related issue https://github.com/DLR-RM/stable-baselines3/issues/1694 "
                "and documentation for more information: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api"
            )

        obs_tensor, vectorized_env = self.obs_to_tensor(observation)

        actions = self._predict(obs_tensor, deterministic=deterministic)
        # Convert to numpy, and reshape to the original action shape
        actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))  # type: ignore[misc]

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)  # type: ignore[assignment, arg-type]
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low, self.action_space.high)  # type: ignore[assignment, arg-type]

        # Remove batch dimension if needed
        if not vectorized_env:
            assert isinstance(actions, np.ndarray)
            actions = actions.squeeze(axis=0)

        return actions, state

    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        if deterministic:
            with torch.no_grad():
                return self.actor(observation, deterministic)
        else:
            pre_tanh_mean, log_std, kwargs = self.actor.get_action_dist_params(observation)
            std = log_std.exp()
            q1, q2 = self.critic(observation, torch.tanh(pre_tanh_mean))
            assert q1.requires_grad and q2.requires_grad, "q1 or q2 does not require grad"
            mu_q = 0.5 * (q1 + q2)
            sigma_q = 0.5 * torch.abs(q1 - q2)

            q_ub = mu_q + self.beta_ub * sigma_q

            # Obtain the gradient of Q_UB wrt to a
            # with a evaluated at mu_t
            grad = torch.autograd.grad(q_ub.sum(), pre_tanh_mean)
            grad = grad[0]

            assert grad is not None
            assert pre_tanh_mean.shape == grad.shape

            # Obtain Sigma_T (the covariance of the normal distribution)
            sigma_t = torch.pow(std, 2)

            # The dividor is (g^T Sigma g) ** 0.5
            # Sigma is diagonal, so this works out to be
            # ( sum_{i=1}^k (g^(i))^2 (sigma^(i))^2 ) ** 0.5
            denom = torch.sqrt(
                torch.sum(
                    torch.mul(torch.pow(grad, 2), sigma_t)
                )
            ) + 1e-6

            # Obtain the change in mu
            mu_c = self.shift_multiplier * torch.mul(sigma_t, grad) / denom

            assert mu_c.shape == pre_tanh_mean.shape

            mu_e = pre_tanh_mean + mu_c

            # Construct the tanh normal distribution and sample the exploratory action from it
            assert mu_e.shape == std.shape

            dist = Normal(mu_e, std)

            action = torch.tanh(dist.sample())
            return action.detach()

