from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from gymnasium.wrappers.common import TimeLimit
import wandb
from stable_baselines3.sac import SAC
from wandb.integration.sb3 import WandbCallback
from maxinforl.commons import MaxInfoEpsGreedy, MaxInfoSAC, MaxRNDSAC, OACPolicy
from maxinforl.commons.utils import DisagreementIntrinsicReward, random_exploration_schedule
import numpy as np
import argparse
import yaml
from pathlib import Path
import os


def experiment(
        alg: str = 'maxinfo_eps_greedy',
        domain_name: str = 'cartpole-swingup_sparse',
        logs_dir: str = './logs/',
        project_name: str = 'Test',
        total_steps: int = 250_000,
        num_envs: int = 1,
        action_cost: float = 0.1,
        train_freq: int = 1,
        action_repeat: int = 2,
        seed: int = 0,
        features: int = 256,
):
    env_name, task = domain_name.split('-')
    from maxinforl.envs.dm2gym import DMCGym
    from maxinforl.envs.action_repeat import ActionRepeat
    from maxinforl.envs.action_cost import ActionCost
    tb_dir = logs_dir + 'runs'

    config = dict(
        alg=alg,
        total_steps=total_steps,
        num_envs=num_envs,
        action_cost=action_cost,
        domain_name=domain_name,
        train_freq=train_freq,
        action_repeat=action_repeat,
        features=features,
    )

    run = wandb.init(
        dir=logs_dir,
        project=project_name,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
        config=config,
    )

    env = lambda: TimeLimit(
        ActionRepeat(
            ActionCost(DMCGym(
                domain=env_name,
                task=task,
                render_mode='rgb_array',
            ), action_cost=action_cost),
            repeat=action_repeat, return_total_reward=True),
        max_episode_steps=1_000)

    vec_env = make_vec_env(env, n_envs=num_envs, seed=seed)
    eval_env = make_vec_env(env, n_envs=num_envs, seed=seed + 1_000)

    callback = EvalCallback(eval_env,
                            log_path=logs_dir,
                            best_model_save_path=logs_dir,
                            eval_freq=1_000,
                            n_eval_episodes=5,
                            deterministic=True,
                            render=False
                            )

    algorithm_kwargs = {
        'policy': 'MlpPolicy',
        'train_freq': train_freq,
        'verbose': 1,
        'tensorboard_log': f"{tb_dir}/{run.id}",
        'gradient_steps': -1,
        'learning_starts': 500 * num_envs,
    }

    if alg == 'sac':
        algorithm = SAC(
            env=vec_env,
            seed=seed,
            **algorithm_kwargs
        )
    elif alg == 'oac':
        algorithm_kwargs['policy'] = OACPolicy
        algorithm_kwargs['policy_kwargs'] = {'beta_lb': -3.65,
                                'beta_ub': 4.66, 'shift_multiplier': 6.86}
        algorithm = SAC(
            env=vec_env,
            seed=seed,
            **algorithm_kwargs
        )

    elif alg == 'maxinfo_eps_greedy':

        ensemble_model_kwargs = {
            'learn_std': False,
            'features': (features, features),
            'optimizer_kwargs': {'lr': 3e-4, 'weight_decay': 0.0},
            'use_entropy': False,
        }
        exploration_freq = random_exploration_schedule(eps_init=1.0, eps_final=0.1,
                                                       exploration_fraction=0.25, seed=seed)

        int_reward = DisagreementIntrinsicReward

        algorithm = MaxInfoEpsGreedy(
            env=vec_env,
            ensemble_model_kwargs=ensemble_model_kwargs,
            intrinsic_reward_model=int_reward,
            exploration_freq=exploration_freq,
            seed=seed,
            **algorithm_kwargs
        )
    elif alg == 'maxinfosac':
        ensemble_model_kwargs = {
            'learn_std': False,
            'optimizer_kwargs': {'lr': 3e-4, 'weight_decay': 0.0},
            'features': (features, features),
        }
        algorithm = MaxInfoSAC(
            env=vec_env,
            seed=seed,
            ensemble_model_kwargs=ensemble_model_kwargs,
            **algorithm_kwargs
        )
    elif alg == 'maxinfooac':
        algorithm_kwargs['policy'] = OACPolicy
        algorithm_kwargs['policy_kwargs'] = {'beta_lb': -3.65,
                                'beta_ub': 1.0, 'shift_multiplier': 2.0}
        ensemble_model_kwargs = {
            'learn_std': False,
            'optimizer_kwargs': {'lr': 3e-4, 'weight_decay': 0.0},
            'features': (features, features),
        }

        algorithm = MaxInfoSAC(
            env=vec_env,
            seed=seed,
            ensemble_model_kwargs=ensemble_model_kwargs,
            **algorithm_kwargs
        )
    elif alg == 'maxrndsac':
        import torch.nn
        target_model_kwargs = {
            'learn_std': False,
            'features': (32, 32),
            'optimizer_kwargs': {'lr': 3e-4, 'weight_decay': 0.0},
            'act_fn': torch.nn.Tanh(),
        }

        ensemble_model_kwargs = {
            'learn_std': False,
            'features': (features, features),
            'optimizer_kwargs': {'lr': 3e-4, 'weight_decay': 0.0},
            'act_fn': torch.nn.ReLU(),
        }

        algorithm = MaxRNDSAC(
            env=vec_env,
            seed=seed,
            target_model_kwargs=target_model_kwargs,
            ensemble_model_kwargs=ensemble_model_kwargs,
            embedding_dim=1,
            model_init_gain=np.sqrt(2),
            **algorithm_kwargs
        )

    else:
        raise NotImplementedError

    algorithm.learn(
        total_timesteps=total_steps,
        callback=[WandbCallback(), callback],
    )


def main(args):
    """"""
    from pprint import pprint
    print(args)
    print('\n ------------------------------------ \n')

    """ Experiment core """
    np.random.seed(args.seed)
    ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(ROOT_PATH, 'configs.yaml')
    env_config = yaml.safe_load(Path(config_path).read_text())
    env_config = env_config[args.domain_name]

    experiment(
        alg=args.alg,
        logs_dir=args.logs_dir,
        project_name=args.project_name,
        domain_name=args.domain_name,
        action_cost=args.action_cost,
        seed=args.seed,
        **env_config,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MTTest')

    # general experiment args
    parser.add_argument('--alg', type=str, default='maxinfooac')
    parser.add_argument('--logs_dir', type=str, default='./logs/')
    parser.add_argument('--project_name', type=str, default='MCTest')
    parser.add_argument('--domain_name', type=str, default='cartpole-swingup_sparse')
    parser.add_argument('--action_cost', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    main(args)
