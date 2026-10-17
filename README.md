# MJLab Adaptor

This repository contains a UniEnv adaptor for [`mjlab`](https://github.com/mujocolab/mjlab).

## Installation

```bash
pip install unienv_mjlab
```

## Usage

```python
import mjlab.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

from unienv_mjlab import FromMJLabEnv

cfg = load_env_cfg("Mjlab-Velocity-Flat-Unitree-G1")
cfg.scene.num_envs = 4

mjlab_env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode=None)
env = FromMJLabEnv(mjlab_env)

ctx, obs, info = env.reset(seed=0)
for _ in range(10):
    action = env.sample_action()
    obs, reward, terminated, truncated, info = env.step(action)
```

## Notes

- `mjlab` environments are vectorized by default.
- `mjlab.step()` auto-resets done environments before returning observations. This means that you don't need to call `reset()` after an episode ends.
- The converted spaces carry unienv space metadata read from the mjlab managers: concatenated observation groups list their terms (`terms`: name, shape, slice) and per-element `names` / `units` (e.g. `joint_pos/left_knee_joint` in `rad`), camera terms are described as `CHW` images / depth normalized to `[0, 1]`, and the action space records what each action term controls (`control`, `mode`, joint `names`, `scale`, `offset`).
