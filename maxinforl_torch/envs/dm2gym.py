import logging
import os

import numpy as np
from dm_control import suite
from dm_env import specs
from gymnasium.core import Env
from gymnasium.spaces import Box

"""A simple OpenCV based viewer for dm_control images"""

import cv2
import uuid


# Code taken from: https://github.com/imgeorgiev/dmc2gymnasium/blob/main/dmc2gymnasium/DMCGym.py
# and for rendering: https://github.com/zuoxingdong/dm2gym/blob/master/dm2gym/envs/opencv_image_viewer.py

class OpenCVImageViewer():
    """A simple OpenCV highgui based dm_control image viewer

    This class is meant to be a drop-in replacement for
    `gym.envs.classic_control.rendering.SimpleImageViewer`
    """

    def __init__(self, *, escape_to_exit=False):
        """Construct the viewing window"""
        self._escape_to_exit = escape_to_exit
        self._window_name = str(uuid.uuid4())
        cv2.namedWindow(self._window_name, cv2.WINDOW_AUTOSIZE)
        self._isopen = True

    def __del__(self):
        """Close the window"""
        cv2.destroyWindow(self._window_name)
        self._isopen = False

    def imshow(self, img):
        """Show an image"""
        # Convert image to BGR format
        cv2.imshow(self._window_name, img[:, :, [2, 1, 0]])
        # Listen for escape key, then exit if pressed
        if cv2.waitKey(1) in [27] and self._escape_to_exit:
            exit()

    @property
    def isopen(self):
        """Is the window open?"""
        return self._isopen

    def close(self):
        pass


def _spec_to_box(spec, dtype=np.float32):
    def extract_min_max(s):
        assert s.dtype == np.float64 or s.dtype == np.float32
        dim = int(np.prod(s.shape))
        if type(s) == specs.Array:
            bound = np.inf * np.ones(dim, dtype=np.float32)
            return -bound, bound
        elif type(s) == specs.BoundedArray:
            zeros = np.zeros(dim, dtype=np.float32)
            return s.minimum + zeros, s.maximum + zeros
        else:
            logging.error("Unrecognized type")

    mins, maxs = [], []
    for s in spec:
        mn, mx = extract_min_max(s)
        mins.append(mn)
        maxs.append(mx)
    low = np.concatenate(mins, axis=0).astype(dtype)
    high = np.concatenate(maxs, axis=0).astype(dtype)
    assert low.shape == high.shape
    return Box(low, high, dtype=dtype)


def _flatten_obs(obs, dtype=np.float32):
    obs_pieces = []
    for v in obs.values():
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0).astype(dtype)


class DMCGym(Env):
    def __init__(
            self,
            domain,
            task,
            task_kwargs={},
            environment_kwargs={},
            rendering="egl",
            render_height=64,
            render_width=64,
            render_camera_id=0,
            render_mode: str = 'rgb_array',
            seed: int = 0,
    ):
        """TODO comment up"""

        # for details see https://github.com/deepmind/dm_control
        assert rendering in ["glfw", "egl", "osmesa"]
        os.environ["MUJOCO_GL"] = rendering

        self._env = suite.load(
            domain,
            task,
            task_kwargs,
            environment_kwargs,
        )

        # placeholder to allow built in gymnasium rendering
        self.render_mode = render_mode
        self.render_height = render_height
        self.render_width = render_width
        self.render_camera_id = render_camera_id

        self._observation_space = _spec_to_box(self._env.observation_spec().values())
        self._action_space = _spec_to_box([self._env.action_spec()])
        self.viewer = None

        # set seed if provided with task_kwargs
        self._observation_space.seed(seed)
        self._action_space.seed(seed)

    def __getattr__(self, name):
        """Add this here so that we can easily access attributes of the underlying env"""
        return getattr(self._env, name)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def reward_range(self):
        """DMC always has a per-step reward range of (0, 1)"""
        return 0, 1

    def step(self, action):
        if action.dtype.kind == "f":
            action = action.astype(np.float32)
        assert self._action_space.contains(action)
        timestep = self._env.step(action)
        observation = _flatten_obs(timestep.observation)
        reward = timestep.reward
        termination = False  # we never reach a goal
        truncation = timestep.last()
        info = {"discount": timestep.discount}
        return observation, reward, termination, truncation, info

    def reset(self, seed=None, options=None):
        if seed is not None:
            if not isinstance(seed, np.random.RandomState):
                seed = np.random.RandomState(seed)
            self._env.task._random = seed

        if options:
            logging.warn("Currently doing nothing with options={:}".format(options))
        timestep = self._env.reset()
        observation = _flatten_obs(timestep.observation)
        info = {}
        return observation, info

    def render(self):
        img = self._env.physics.render(height=self.render_height, width=self.render_width,
                                       camera_id=self.render_camera_id)
        if self.render_mode == 'rgb_array':
            return img
        elif self.render_mode == 'human':
            if self.viewer is None:
                self.viewer = OpenCVImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen
        else:
            raise NotImplementedError


if __name__ == '__main__':

    env = DMCGym(
        domain='reacher',
        task='easy',
        render_mode='human',
    )
    observation, _ = env.reset()
    for i in range(1000):
        observation, reward, termination, truncation, info = env.step(env.action_space.sample())
        env.render()
