# MaxInfoRL: Boosting exploration in RL through information gain maximization

An implementation of [MaxInfoRL][paper], a simple, flexible, and scalable reinforcement
learning algorithm that enhances exploration in RL by automatically combining intrinsic and extrinsic rewards.



If you find this code useful, please reference in your paper:

```
@article{sukhija2024maxinforl,
  title={MaxInfoRL: Boosting exploration in reinforcement learning through information gain maximization},
  author={Sukhija, Bhavya and Coros, Stelian and Krause, Andreas and Abbeel, Pieter and Sferrazza, Carmelo},
  journal={ArXiv},
  year={2024}
}
```

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



# Instructions

## Installation

```sh
pip install -e .
```

Training script:

```sh
python examples/dmc/experiment.py \
  --project_name maxinforl \
  --alg maxinfosac \
  --domain_name cartpole-swingup_sparse
```


All hyperparameters are listed in the `examples/state_based//configs.yaml` and `examples/vision_based//configs.yaml` 
files. You can override them if needed.

[jax]: https://github.com/google/jax#pip-installation-gpu-cuda
[paper]: https://openreview.net/pdf?id=R4q3cY3kQf
[website]: https://sukhijab.github.io/
[tweet]: https://sukhijab.github.io/
