from abc import ABC
from typing import Dict, Optional, List
import torch
import numpy as np


class BaseIntrinsicReward(ABC):
    def __init__(self, ensemble_model: torch.nn.Module,
                 intrinsic_reward_weights: Dict, agg_intrinsic_reward: str = 'sum'):
        self.intrinsic_reward_weights = intrinsic_reward_weights
        self.agg_intrinsic_reward = agg_intrinsic_reward
        self.ensemble_model = ensemble_model

    def __call__(self, inp: torch.Tensor, labels: Dict):
        raise NotImplementedError

    def aggregate_intrinsic_reward(self, intrinsic_rewards: torch.Tensor):
        if self.agg_intrinsic_reward == 'sum':
            intrinsic_rewards = intrinsic_rewards.sum(dim=0)
        elif self.agg_intrinsic_reward == 'max':
            intrinsic_rewards = intrinsic_rewards.max(dim=0)
        elif self.agg_intrinsic_reward == 'mean':
            intrinsic_rewards = intrinsic_rewards.mean(dim=0)
        else:
            raise NotImplementedError
        return intrinsic_rewards


class CuriosityIntrinsicReward(BaseIntrinsicReward):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, inp: torch.Tensor, labels: Dict):
        self.ensemble_model.eval()
        predictions = self.ensemble_model(inp)
        curiosity = torch.stack([
            # take mean of ensemble as prediction
            ((predictions[key].mean(dim=-1) - y) ** 2
             ).mean(dim=-1) * self.intrinsic_reward_weights[key]  # take mean error over the dimension of output
            for key, y in labels.items()])
        curiosity = self.aggregate_intrinsic_reward(curiosity)
        return curiosity


class DisagreementIntrinsicReward(BaseIntrinsicReward):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, inp: torch.Tensor, labels: Dict):
        self.ensemble_model.eval()
        predictions = self.ensemble_model(inp)
        disg = self.ensemble_model.get_disagreement(predictions)
        disg = torch.stack([val * self.intrinsic_reward_weights[key] for key, val in disg.items()])
        disg = self.aggregate_intrinsic_reward(disg)
        return disg


def sigmoid_schedule(init_value: float = 0.0,
                     point_for_pure_exploitation: float = 0.75,
                     slope: float = 20,
                     ):
    """

    :param init_value: initial value for exploration, if 0.0 -> equal exploration and exploitation is done
    :param point_for_pure_exploitation: point of progress after which exploration_weight gets nearly 0
    :return:
    """
    assert point_for_pure_exploitation < 1
    max_value = 100

    # progress is 1 at the beginning and 0 at the end
    def exploration_weight(progress: float):
        # value is min_value when progress is 1 and 0 when point_for_equal_weight is reached
        value = - slope * (progress - 1) + init_value
        if progress <= point_for_pure_exploitation:
            value = max_value
        # when value is small --> we get nearly 1 and when value gets big we get 0.
        return 1 / (1 + np.exp(value))

    return exploration_weight


def explore_till(
        exploration_till_progress: float = 0.75,
):
    def exploration_weight(progress: float):
        if progress > exploration_till_progress:
            return 1.0
        else:
            return 0.0

    return exploration_weight


def exploration_frequency_schedule(
        thressholds: Optional[List[List]] = None,
):
    if thressholds is not None:
        thressholds = sorted(thressholds, reverse=True)
    else:
        return 1_000_000

    # example: thressholds = [[1 1], [0.75, 2], [0.5, 5], [0.25, 100]]
    def exploration_freq(progress: float):
        # progress starts at 1 and goes to 0.
        current_freq = 1_000_000
        for thresshold in thressholds:
            # if progress = 1 -> returns 1 till progress is <= 0.75
            if progress <= thresshold[0]:
                current_freq = thresshold[1]
            else:
                return current_freq
        return current_freq

    return exploration_freq


def random_exploration_schedule(eps_init: float = 1.0, eps_final: float = 0.001,
                                exploration_fraction: float = 0.0,
                                seed: int = 0,
                                ):
    assert exploration_fraction < 1.0
    assert exploration_fraction >= 0.0
    np.random.seed(seed=seed)

    # example: thressholds = [[1 1], [0.75, 2], [0.5, 5], [0.25, 100]]
    def exploration_freq(progress: float):
        # progress starts at 1 and goes to 0.
        eps = (eps_init - eps_final) * (progress - exploration_fraction) / (1 - exploration_fraction) + eps_final
        u = np.random.uniform(low=0, high=1)
        explore = u <= eps
        if explore:
            return 1
        else:
            return -1

    return exploration_freq
