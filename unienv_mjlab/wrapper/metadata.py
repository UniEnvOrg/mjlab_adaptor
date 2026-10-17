"""Space metadata for mjlab manager-based environments.

mjlab declares every observation and action space as an unbounded ``Box``; what
the values mean lives in the managers that produce them. This module reads the
observation manager's term configs and the action manager's terms and describes
the converted unienv spaces with the shared metadata vocabulary (see
``unienv_interface.space.space_utils.metadata_utils``):

* a group that keeps its terms apart (``concatenate_terms=False``) gets one
  description per term leaf;
* a concatenated group gets a group-level description: the keys all its terms
  agree on, per-element ``names`` (``"<term>/<element>"``) / ``units`` when the
  terms are concatenated along the last axis, and a ``terms`` list with the
  name, shape and ``[start, stop)`` slice (along the concatenation axis) of
  every term together with its own description;
* the action space gets the description of its action term(s): what is
  controlled, how, per-element names and the term's ``scale`` / ``offset``.

Terms are recognised by their function; terms this module does not know are
still listed (name, shape, slice) but not otherwise described.
"""

from __future__ import annotations

import inspect
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import mujoco
import numpy as np
import torch

from unienv_interface.space.space_utils.metadata_utils import (
    action_metadata,
    depth_metadata,
    image_metadata,
    metadata,
    proprio_metadata,
)

__all__ = ["mjlab_observation_metadata", "mjlab_action_metadata"]

_XYZ = ["x", "y", "z"]

# Keys of the term descriptions that a concatenated group keeps when all its
# terms agree on them. Per-element keys (names / units lists / range) are
# combined separately; ``format`` is not kept as concatenating images changes
# their channel count.
_GROUP_COMMON_KEYS = ("kind", "frame", "source", "fps", "layout", "normalized")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _float32(value: float) -> float:
    """``value`` as the shortest float that round-trips through float32 (0.2, not 0.2000000029)."""
    return float(str(np.float32(value)))


def _to_json_number_list(value: Any) -> Any:
    """``float`` or a per-element list from a float / (num_envs, dim) or (dim,) tensor.

    Returns ``None`` when the value differs between environments.
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.ndim == 0:
            return _float32(float(value))
        if value.ndim == 2:
            if not bool(torch.all(value == value[:1])):
                return None
            value = value[0]
        values = [_float32(v) for v in value.tolist()]
        if len(values) > 0 and all(v == values[0] for v in values):
            return values[0]
        return values
    return None


def _param(term_cfg: Any, name: str) -> Any:
    """A term parameter, falling back to the default of the term function's argument."""
    params = term_cfg.params or {}
    if name in params:
        return params[name]
    try:
        parameter = inspect.signature(term_cfg.func).parameters.get(name)
    except (TypeError, ValueError):
        return None
    if parameter is None or parameter.default is inspect.Parameter.empty:
        return None
    return parameter.default


def _joint_names_and_units(
    env: Any, asset_cfg: Any, velocity: bool
) -> Tuple[Optional[List[str]], Optional[List[str]]]:
    """Names and units of the joints selected by ``asset_cfg`` (an mjlab ``SceneEntityCfg``)."""
    if asset_cfg is None:
        return None, None
    entity = env.scene[asset_cfg.name]
    joints = [j for j in entity.spec.joints if j.type != mujoco.mjtJoint.mjJNT_FREE]
    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, slice):
        selected = joints[joint_ids]
    else:
        selected = [joints[int(i)] for i in joint_ids]
    names = [j.name.split("/")[-1] for j in selected]
    units: Optional[List[str]] = []
    for joint in selected:
        if joint.type == mujoco.mjtJoint.mjJNT_HINGE:
            units.append("rad/s" if velocity else "rad")
        elif joint.type == mujoco.mjtJoint.mjJNT_SLIDE:
            units.append("m/s" if velocity else "m")
        else:  # ball joints have several qpos / dof entries per joint
            units = None
            break
    return names, units


def _action_names(env: Any, action_name: Optional[str] = None) -> Optional[List[str]]:
    """Per-element names of the (whole, or one term of the) action vector."""
    manager = env.action_manager
    term_names = [action_name] if action_name is not None else list(manager.active_terms)
    per_term = [_action_term_element_names(manager.get_term(name)) for name in term_names]
    flat = [name for names in per_term for name in names]
    if action_name is None and len(term_names) > 1 and len(set(flat)) != len(flat):
        flat = [
            f"{term}/{name}" for term, names in zip(term_names, per_term) for name in names
        ]
    return flat


# ---------------------------------------------------------------------------
# Observation terms
# ---------------------------------------------------------------------------

_SENSOR_DESCRIPTIONS: Dict[str, Dict[str, Any]] = {
    # site sensors: measured in the site frame
    "accelerometer": dict(units="m/s^2", frame="link_local", names=_XYZ),
    "velocimeter": dict(units="m/s", frame="link_local", names=_XYZ),
    "gyro": dict(units="rad/s", frame="link_local", names=_XYZ),
    "force": dict(units="N", frame="link_local", names=_XYZ, kind="force"),
    "torque": dict(units="N*m", frame="link_local", names=_XYZ, kind="force"),
    "rangefinder": dict(units="m"),
    # frame sensors: in the world frame unless a reference frame is given
    "framepos": dict(units="m", names=_XYZ),
    "framequat": dict(rotation="quat_wxyz", names=["qw", "qx", "qy", "qz"], kind="pose"),
    "framelinvel": dict(units="m/s", names=_XYZ),
    "frameangvel": dict(units="rad/s", names=_XYZ),
    "framelinacc": dict(units="m/s^2", names=_XYZ),
    "frameangacc": dict(units="rad/s^2", names=_XYZ),
    "subtreecom": dict(units="m", frame="world", names=_XYZ),
    "subtreelinvel": dict(units="m/s", frame="world", names=_XYZ),
    "clock": dict(units="s"),
}


def _describe_builtin_sensor(env: Any, term_cfg: Any) -> Dict[str, Any]:
    # Builtin sensors may come from a BuiltinSensorCfg or from the robot's XML
    # (no cfg), so read the type and reference frame from the compiled model.
    sensor_name = _param(term_cfg, "sensor_name")
    model_sensor = env.sim.mj_model.sensor(sensor_name)
    sensor_type = mujoco.mjtSensor(int(model_sensor.type[0])).name
    sensor_type = sensor_type.removeprefix("mjSENS_").lower()
    description = dict(_SENSOR_DESCRIPTIONS.get(sensor_type, {}))
    kind = description.pop("kind", "proprio")
    if sensor_type.startswith("frame") and int(model_sensor.refid[0]) < 0:
        description["frame"] = "world"
    return metadata(kind, source=sensor_name, sensor_type=sensor_type, **description)


def _describe_joint_pos_rel(env: Any, term_cfg: Any) -> Dict[str, Any]:
    names, units = _joint_names_and_units(env, _param(term_cfg, "asset_cfg"), velocity=False)
    return proprio_metadata(
        units=units, names=names, label="joint positions relative to the default pose"
    )


def _describe_joint_vel_rel(env: Any, term_cfg: Any) -> Dict[str, Any]:
    names, units = _joint_names_and_units(env, _param(term_cfg, "asset_cfg"), velocity=True)
    return proprio_metadata(
        units=units, names=names, label="joint velocities relative to the default velocities"
    )


def _describe_last_action(env: Any, term_cfg: Any) -> Dict[str, Any]:
    action_name = _param(term_cfg, "action_name")
    return metadata(
        "action",
        names=_action_names(env, action_name),
        action_term=action_name,
        label="last action (raw, before the action terms' scale / offset)",
    )


def _describe_generated_commands(env: Any, term_cfg: Any) -> Dict[str, Any]:
    command_name = _param(term_cfg, "command_name")
    command = env.command_manager.get_term(command_name)
    if type(command).__name__ == "UniformVelocityCommand":
        return metadata(
            "task",
            names=["lin_vel_x", "lin_vel_y", "ang_vel_z"],
            units=["m/s", "m/s", "rad/s"],
            frame="base",
            command=command_name,
            label="commanded base velocity",
        )
    return metadata("task", command=command_name)


def _camera_fields(env: Any, term_cfg: Any) -> Dict[str, Any]:
    return dict(source=_param(term_cfg, "sensor_name"), fps=1.0 / env.step_dt)


def _describe_camera_rgb(env: Any, term_cfg: Any) -> Dict[str, Any]:
    # (B, 3, H, W) float in [0, 1] (uint8 RGB / 255)
    return image_metadata(
        format="rgb", layout="CHW", normalized=True, range=[0, 255], **_camera_fields(env, term_cfg)
    )


def _describe_camera_depth(env: Any, term_cfg: Any) -> Dict[str, Any]:
    # (B, 1, H, W) = clamp(depth, min_depth, cutoff) / cutoff, in [0, 1]
    cutoff = float(_param(term_cfg, "cutoff_distance"))
    min_depth = _param(term_cfg, "min_depth")
    return depth_metadata(
        units="normalized",
        layout="CHW",
        normalized=True,
        range=[0.0, cutoff],
        cutoff_distance=cutoff,
        min_depth=None if min_depth is None else float(min_depth),
        **_camera_fields(env, term_cfg),
    )


def _describe_camera_segmentation(env: Any, term_cfg: Any) -> Dict[str, Any]:
    # (B, 2, H, W): MuJoCo object id and object type per pixel
    return metadata(
        "segmentation",
        layout="CHW",
        channels=["object_id", "object_type"],
        **_camera_fields(env, term_cfg),
    )


def _describe_camera_target_cube_mask(env: Any, term_cfg: Any) -> Dict[str, Any]:
    # (B, 1, H, W) float, 1 where the target cube is visible
    return metadata("mask", layout="CHW", **_camera_fields(env, term_cfg))


def _fixed(**fields: Any) -> Callable[[Any, Any], Dict[str, Any]]:
    kind = fields.pop("kind")

    def describe(env: Any, term_cfg: Any) -> Dict[str, Any]:
        sensor_name = _param(term_cfg, "sensor_name")
        return metadata(kind, source=sensor_name, **fields)

    return describe


_ENVS_MDP = "mjlab.envs.mdp.observations"
_MANIP_MDP = "mjlab.tasks.manipulation.mdp.observations"
_VEL_MDP = "mjlab.tasks.velocity.mdp.observations"

# Term functions by "<module>.<qualname>" (so task modules need not be imported).
_OBSERVATION_DESCRIBERS: Dict[str, Callable[[Any, Any], Dict[str, Any]]] = {
    f"{_ENVS_MDP}.base_lin_vel": _fixed(kind="proprio", units="m/s", frame="base", names=_XYZ),
    f"{_ENVS_MDP}.base_ang_vel": _fixed(kind="proprio", units="rad/s", frame="base", names=_XYZ),
    f"{_ENVS_MDP}.projected_gravity": _fixed(
        kind="proprio", units="normalized", frame="base", names=_XYZ,
        label="gravity direction (unit vector)",
    ),
    f"{_ENVS_MDP}.projected_gravity_from_sensor": _fixed(
        kind="proprio", units="normalized", frame="link_local", names=_XYZ,
        label="gravity direction (unit vector)",
    ),
    f"{_ENVS_MDP}.joint_pos_rel": _describe_joint_pos_rel,
    f"{_ENVS_MDP}.joint_vel_rel": _describe_joint_vel_rel,
    f"{_ENVS_MDP}.last_action": _describe_last_action,
    f"{_ENVS_MDP}.generated_commands": _describe_generated_commands,
    f"{_ENVS_MDP}.builtin_sensor": _describe_builtin_sensor,
    f"{_ENVS_MDP}.height_scan": _fixed(
        kind="height_scan", units="m", label="height of the sensor frame above the terrain"
    ),
    f"{_MANIP_MDP}.camera_rgb": _describe_camera_rgb,
    f"{_MANIP_MDP}.camera_depth": _describe_camera_depth,
    f"{_MANIP_MDP}.camera_segmentation": _describe_camera_segmentation,
    f"{_MANIP_MDP}.camera_target_cube_mask": _describe_camera_target_cube_mask,
    f"{_MANIP_MDP}.ee_to_object_distance": _fixed(
        kind="state", units="m", frame="base", names=_XYZ,
        label="object position relative to the end effector",
    ),
    f"{_MANIP_MDP}.object_to_goal_distance": _fixed(
        kind="task", units="m", frame="base", names=_XYZ,
        label="goal position relative to the object",
    ),
    f"{_MANIP_MDP}.target_position": _fixed(
        kind="task", units="m", frame="link_local", names=_XYZ,
        label="target position in the end-effector frame",
    ),
    f"{_MANIP_MDP}.ee_velocity": _fixed(
        kind="proprio", units="m/s", frame="link_local", names=_XYZ,
        label="end-effector linear velocity in the end-effector frame",
    ),
    f"{_VEL_MDP}.foot_height": _fixed(kind="proprio", units="m", label="foot clearance above the terrain"),
    f"{_VEL_MDP}.foot_air_time": _fixed(kind="proprio", units="s", label="current foot air time"),
    f"{_VEL_MDP}.foot_contact": _fixed(kind="tactile", label="foot contact flags (1 = in contact)"),
    f"{_VEL_MDP}.foot_contact_forces": _fixed(
        kind="force", label="foot contact forces, sign(f) * log(1 + |f|)"
    ),
    "mjlab.tasks.cartpole.cartpole_env_cfg.pole_angle_cos_sin": _fixed(
        kind="proprio", units="normalized", names=["cos", "sin"], label="pole angle (cos, sin)"
    ),
}


def _term_key(func: Any) -> str:
    target = func if inspect.isfunction(func) or inspect.ismethod(func) else type(func)
    return f"{getattr(target, '__module__', '')}.{getattr(target, '__qualname__', '')}"


def _per_element_length(term_dim: Sequence[int]) -> Optional[int]:
    return int(term_dim[-1]) if len(term_dim) > 0 else None


def _describe_observation_term(
    env: Any, term_name: str, term_cfg: Any, term_dim: Sequence[int]
) -> Dict[str, Any]:
    describer = _OBSERVATION_DESCRIBERS.get(_term_key(term_cfg.func))
    if describer is None:
        return {}
    try:
        description = dict(describer(env, term_cfg))
    except Exception as e:  # metadata is best effort; never break env construction
        warnings.warn(f"unienv_mjlab: could not describe observation term {term_name!r}: {e!r}")
        return {}

    if term_cfg.scale is not None:
        # scaled values no longer carry the physical units
        description.pop("units", None)
        description.pop("range", None)
        description["scale"] = _to_json_number_list(term_cfg.scale)
    if term_cfg.clip is not None:
        description["clip"] = [float(term_cfg.clip[0]), float(term_cfg.clip[1])]

    history = int(term_cfg.history_length or 0)
    if history > 0:
        description["history_length"] = history
        if term_cfg.flatten_history_dim:
            # (H * D,): oldest to newest, each a full term vector
            for key in ("names", "units"):
                values = description.get(key)
                if isinstance(values, list):
                    if key == "names":
                        values = [
                            f"{name}[t-{history - 1 - step}]"
                            for step in range(history)
                            for name in values
                        ]
                    else:
                        values = values * history
                    description[key] = values

    # Per-element lists that do not match the term (e.g. ball joints) are dropped.
    length = _per_element_length(term_dim)
    for key in ("names", "units"):
        values = description.get(key)
        if isinstance(values, list) and len(values) != length:
            del description[key]
    return {key: value for key, value in description.items() if value is not None}


def _describe_concatenated_group(
    group_cfg: Any,
    term_names: Sequence[str],
    term_dims: Sequence[Sequence[int]],
    descriptions: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    ndim = len(term_dims[0])
    concat_dim = group_cfg.concatenate_dim
    concat_dim = concat_dim if concat_dim >= 0 else ndim + concat_dim
    # Per-element lists describe the last axis; images (terms with a layout) have none.
    along_last_axis = concat_dim == ndim - 1 and not any("layout" in d for d in descriptions)

    group: Dict[str, Any] = {}
    if len(descriptions) == 1:
        group.update(descriptions[0])
    else:
        for key in _GROUP_COMMON_KEYS:
            values = [d.get(key) for d in descriptions]
            if values[0] is not None and all(v == values[0] for v in values):
                group[key] = values[0]

    if along_last_axis:
        names: List[str] = []
        units: Optional[List[str]] = []
        for term_name, dim, description in zip(term_names, term_dims, descriptions):
            size = int(dim[-1])
            term_element_names = description.get("names")
            if not isinstance(term_element_names, list):
                term_element_names = [str(i) for i in range(size)]
            names.extend(f"{term_name}/{name}" for name in term_element_names)
            term_units = description.get("units")
            if units is not None and isinstance(term_units, str):
                units.extend([term_units] * size)
            elif units is not None and isinstance(term_units, list):
                units.extend(term_units)
            else:
                units = None
        group["names"] = names
        if units is not None:
            group["units"] = units

    terms = []
    start = 0
    for term_name, dim, description in zip(term_names, term_dims, descriptions):
        stop = start + int(dim[concat_dim])
        entry = {"name": term_name, "shape": [int(d) for d in dim], "start": start, "stop": stop}
        entry.update(
            (key, value)
            for key, value in description.items()
            if not (along_last_axis and key == "names")
        )
        terms.append(entry)
        start = stop
    group["terms"] = terms
    return group


def mjlab_observation_metadata(env: Any) -> Dict[Tuple[str, ...], Dict[str, Any]]:
    """Metadata for the observation space of ``env`` (an mjlab ``ManagerBasedRlEnv``).

    Returns a mapping from space paths (``(group,)`` for a concatenated group,
    ``(group, term)`` for a term of a dict group) to metadata, for ``annotate``.
    """
    manager = env.observation_manager
    result: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    for group_name, term_names in manager.active_terms.items():
        term_dims = manager.group_obs_term_dim[group_name]
        term_cfgs = manager._group_obs_term_cfgs[group_name]
        descriptions = [
            _describe_observation_term(env, name, cfg, dim)
            for name, cfg, dim in zip(term_names, term_cfgs, term_dims)
        ]
        if manager.group_obs_concatenate[group_name]:
            result[(group_name,)] = _describe_concatenated_group(
                manager.cfg[group_name], term_names, term_dims, descriptions
            )
        else:
            for term_name, description in zip(term_names, descriptions):
                if description:
                    result[(group_name, term_name)] = description
    return result


# ---------------------------------------------------------------------------
# Action terms
# ---------------------------------------------------------------------------


def _action_term_element_names(term: Any) -> List[str]:
    from mjlab.envs.mdp.actions import DifferentialIKAction

    if isinstance(term, DifferentialIKAction):
        return {
            3: ["x", "y", "z"],
            6: ["x", "y", "z", "rx", "ry", "rz"],
            7: ["x", "y", "z", "qw", "qx", "qy", "qz"],
        }[term.action_dim]
    names = getattr(term, "target_names", None)
    if names is not None and len(names) == term.action_dim:
        return [str(name) for name in names]
    return [str(i) for i in range(term.action_dim)]


def _describe_action_term(term: Any) -> Dict[str, Any]:
    from mjlab.envs.mdp import actions as mjlab_actions
    from mjlab.envs.mdp.actions import (
        DifferentialIKAction,
        JointEffortAction,
        JointPositionAction,
        JointVelocityAction,
    )
    from mjlab.envs.mdp.actions.actions import BaseAction

    # added after mjlab 1.2; ``isinstance(x, ())`` is always False
    RelativeJointPositionAction = getattr(mjlab_actions, "RelativeJointPositionAction", ())

    names = _action_term_element_names(term)
    if isinstance(term, DifferentialIKAction):
        cfg = term.cfg
        dim = term.action_dim
        if cfg.use_relative_mode:
            fields: Dict[str, Any] = dict(
                mode="delta",
                delta_pos_scale=float(cfg.delta_pos_scale),
                label="end-effector pose delta = action * (delta_pos_scale, delta_ori_scale)",
            )
            if dim == 6:
                fields.update(rotation="axis_angle", delta_ori_scale=float(cfg.delta_ori_scale))
        else:
            fields = dict(mode="absolute", label="end-effector pose target")
            if dim == 3:
                fields["units"] = "m"
            elif dim == 7:
                fields.update(
                    rotation="quat_wxyz", units=["m", "m", "m"] + ["normalized"] * 4
                )
        return action_metadata(
            control="eef_pose", frame="world", names=names,
            frame_type=cfg.frame_type, frame_name=cfg.frame_name, **fields,
        )

    if isinstance(term, BaseAction):
        if isinstance(term, RelativeJointPositionAction):
            control, mode, label = (
                "joint_position", "delta", "joint position target = current position + action * scale"
            )
        elif isinstance(term, JointPositionAction):
            control, mode, label = (
                "joint_position", "absolute", "joint position target = action * scale + offset"
            )
        elif isinstance(term, JointVelocityAction):
            control, mode, label = (
                "joint_velocity", "absolute", "joint velocity target = action * scale + offset"
            )
        elif isinstance(term, JointEffortAction):
            control, mode, label = (
                "joint_torque", "absolute", "joint effort target = action * scale + offset"
            )
        else:  # tendon / site actions
            control, mode, label = None, None, f"{type(term).__name__} target = action * scale + offset"
        offset = None if isinstance(term, RelativeJointPositionAction) else _to_json_number_list(term.offset)
        return action_metadata(
            control=control, mode=mode, names=names, label=label,
            scale=_to_json_number_list(term.scale), offset=offset,
        )

    return action_metadata(names=names)


def mjlab_action_metadata(env: Any) -> Dict[str, Any]:
    """Metadata for the (concatenated) action space of ``env`` (an mjlab ``ManagerBasedRlEnv``)."""
    manager = env.action_manager
    term_names = list(manager.active_terms)
    descriptions = []
    for name in term_names:
        term = manager.get_term(name)
        try:
            descriptions.append(_describe_action_term(term))
        except Exception as e:  # metadata is best effort; never break env construction
            warnings.warn(f"unienv_mjlab: could not describe action term {name!r}: {e!r}")
            descriptions.append(action_metadata(names=[str(i) for i in range(term.action_dim)]))

    if len(term_names) == 1:
        result = dict(descriptions[0])
    else:
        result = action_metadata(names=_action_names(env))
        for key in ("control", "mode", "frame", "rotation"):
            values = [d.get(key) for d in descriptions]
            if values[0] is not None and all(v == values[0] for v in values):
                result[key] = values[0]

    terms = []
    start = 0
    for name, description in zip(term_names, descriptions):
        stop = start + manager.get_term(name).action_dim
        entry = {"name": name, "start": start, "stop": stop}
        if len(term_names) > 1:  # a single term is described at the top level
            entry.update((k, v) for k, v in description.items() if k not in ("kind", "names"))
        terms.append(entry)
        start = stop
    result["terms"] = terms
    return result
