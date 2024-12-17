# MaxInfoRL: Boosting exploration in RL through information gain maximization

A Pytorch implementation of [MaxInfoRL][paper], a simple, flexible, and scalable class of reinforcement learning algorithms that enhance exploration in RL by automatically combining intrinsic and extrinsic rewards. For a jax implementation, visit this [jax repository][jaxrepo].

To learn more:

- [Research paper][paper]
- [Project website][website]
- [Twitter summary][tweet]

## MaxInfoRL

MaxInfoRL boosts exploration in RL by combining extrinsic rewards with intrinsic 
exploration bonuses derived from information gain of the underlying MDP.
MaxInfoRL naturally trades off maximization of the value function with that of the entropy over states, rewards,
and actions. MaxInfoRL is very general and can be combined with a variety
of off-policy model-free RL methods for continuous state-action spaces. We provide implementations of 
**MaxInfoSac, MaxRNDSAC, MaxInfoOAC, $\epsilon$--MaxInfoRL**. 
Our implementations build up on the stable-baselines3 package.

# Instructions

## Installation

```sh
pip install -e .
```

## Training

Training script:

```sh
python examples/dmc/experiment.py \
  --project_name maxinforl \
  --alg maxinfosac \
  --domain_name cartpole-swingup_sparse
```

You can run sac, oac, maxinfosac, maxinfooac, maxrndsac, or maxinfo_eps_greedy by specifying the alg flag.

[paper]: https://arxiv.org/abs/2412.12098
[website]: https://sukhijab.github.io/projects/maxinforl/
[tweet]: https://sukhijab.github.io/
[jaxrepo]: https://github.com/sukhijab/maxinforl_jax

## Custom environments

This repo relies on stable-baselines3 to load environments, natively supporting Gym environments. If your environment is registered in Gym, you can directly use it (just adjust the configs.yaml file accordingly). 

# Citation
If you find MaxInfoRL useful for your research, please cite this work:
```
@article{sukhija2024maxinforl,
  title={MaxInfoRL: Boosting exploration in reinforcement learning through information gain maximization},
  author={Sukhija, Bhavya and Coros, Stelian and Krause, Andreas and Abbeel, Pieter and Sferrazza, Carmelo},
  journal={arXiv preprint arXiv:2412.12098},
  year={2024}
}
```

# References
This codebase contains some files adapted from other sources:
* Stable-Baselines3: https://github.com/DLR-RM/stable-baselines3