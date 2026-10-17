from typing import Any, Dict, Mapping, Optional, SupportsFloat, Tuple, Union

import torch
import numpy as np
from mjlab.envs import ManagerBasedRlEnv

from unienv_interface.backends.pytorch import (
    PyTorchArrayType,
    PyTorchComputeBackend,
    PyTorchDeviceType,
    PyTorchDtypeType,
    PyTorchRNGType,
)
from unienv_interface.env_base.env import Env
from unienv_interface.space.space_utils import batch_utils as sbu
from unienv_interface.space.space_utils.metadata_utils import annotate
from .metadata import mjlab_action_metadata, mjlab_observation_metadata
from .space_utils import from_mjlab_space


def mask_mjlab_data(data: Any, mask: torch.Tensor) -> Any:
    return PyTorchComputeBackend.map_fn_over_arrays(data, lambda x: x[mask])


MJLabData = Union[torch.Tensor, Mapping[str, "MJLabData"]]


class FromMJLabEnv(
    Env[
        PyTorchComputeBackend.ARRAY_TYPE,
        None,
        MJLabData,
        MJLabData,
        np.ndarray,
        PyTorchComputeBackend.DEVICE_TYPE,
        PyTorchComputeBackend.DTYPE_TYPE,
        PyTorchComputeBackend.RNG_TYPE,
    ]
):
    backend = PyTorchComputeBackend

    @property
    def metadata(self) -> Dict[str, Any]:
        return self.env.metadata

    @property
    def render_mode(self) -> Optional[str]:
        return self.env.render_mode

    @property
    def render_fps(self) -> Optional[int]:
        render_fps = 1 / self.env.step_dt
        return int(round(render_fps))

    @property
    def device(self) -> PyTorchDeviceType:
        return self.env.device

    @property
    def batch_size(self) -> Optional[int]:
        return self.env.num_envs

    def __init__(self, env: ManagerBasedRlEnv) -> None:
        self.env = env

        # ========== Convert spaces ==========
        # mjlab declares unbounded boxes; the meaning of the values (units, joint
        # names, cameras, ...) is read from the action / observation managers.
        self.action_space = from_mjlab_space(
            env.action_space, device=self.device
        ).with_metadata(mjlab_action_metadata(env))
        self.observation_space = annotate(
            from_mjlab_space(env.observation_space, device=self.device),
            mjlab_observation_metadata(env),
        )
        self.context_space = None

        self.rng = torch.Generator(device=self.device)

    def reset(
        self,
        *,
        mask: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Tuple[None, MJLabData, Dict[str, Any]]:
        env_ids = None
        if mask is not None:
            env_ids = torch.nonzero(mask, as_tuple=True)[0]

        obs, info = self.env.reset(seed=seed, env_ids=env_ids, **kwargs)

        if mask is not None:
            obs = mask_mjlab_data(obs, mask)
            info = mask_mjlab_data(info, mask)

        return None, obs, info

    def step(
        self,
        action: MJLabData,
    ) -> Tuple[MJLabData, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the wrapped MJLab environment.

        Note:
            MJLab currently performs auto-reset for finished sub-environments inside
            ``step()``, so the returned observations may already correspond to the
            reset state for entries where ``terminated`` or ``truncated`` is true.
            See: https://github.com/mujocolab/mjlab/issues/900
        """
        obs, rew, terminated, truncated, info = self.env.step(action)
        return obs, rew, terminated, truncated, info

    def render(self) -> Optional[Any]:
        return self.env.render()

    def close(self) -> None:
        self.env.close()

    def has_wrapper_attr(self, name: str) -> bool:
        return hasattr(self, name) or hasattr(self.env, name)

    def get_wrapper_attr(self, name: str) -> Any:
        if hasattr(self, name):
            return getattr(self, name)
        if hasattr(self.env, name):
            return getattr(self.env, name)
        raise AttributeError(f"Attribute {name} not found in the environment.")

    def set_wrapper_attr(self, name: str, value: Any):
        if hasattr(self, name):
            setattr(self, name, value)
        elif hasattr(self.env, name):
            setattr(self.env, name, value)
        else:
            raise AttributeError(f"Attribute {name} not found in the environment.")
