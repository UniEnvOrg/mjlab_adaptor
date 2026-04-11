from pathlib import Path
import os
import sys
import numpy as np
import pytest
import torch

import mjlab.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from unienv_mjlab import FromMJLabEnv

TEST_ENV_NAME = "Mjlab-Velocity-Flat-Unitree-G1"

def construct_env(
    num_envs: int = 4,
    episode_length_s: float = 20.0,
    render_mode: str | None = None,
) -> FromMJLabEnv:
    cfg = load_env_cfg(TEST_ENV_NAME)
    cfg.scene.num_envs = num_envs
    cfg.episode_length_s = episode_length_s
    mjlab_env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode=render_mode)
    return FromMJLabEnv(mjlab_env)

def test_mjlab_reset_step_and_masked_reset():
    env = construct_env(num_envs=4)

    _, obs, info = env.reset(seed=0)
    assert obs in env.observation_space
    assert env.render() is None
    assert env.batch_size == 4
    assert env.device == "cpu"
    assert env.render_fps > 0

    action = env.sample_action()
    assert action in env.action_space

    obs, reward, terminated, truncated, info = env.step(action)
    assert obs in env.observation_space
    assert reward.shape == (4,)
    assert terminated.shape == (4,)
    assert truncated.shape == (4,)
    assert isinstance(info, dict)

    env.close()

@pytest.mark.skipif(
    os.environ.get("MJLAB_ENABLE_RENDER_TESTS") != "1",
    reason="Render test requires an available EGL/OpenGL context.",
)
def test_mjlab_rgb_array_render_smoke():
    env = construct_env(num_envs=1, render_mode="rgb_array")

    env.reset(seed=0)
    frame = env.render()
    assert isinstance(frame, np.ndarray)
    assert frame.dtype == np.uint8

    env.close()
