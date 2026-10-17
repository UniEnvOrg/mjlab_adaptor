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


def _metadata(space):
    return space.metadata.to_dict()


def test_mjlab_space_metadata():
    from unienv_interface.space.space_utils.metadata_utils import validate_metadata

    env = construct_env(num_envs=2)
    validate_metadata(env.observation_space)
    validate_metadata(env.action_space)

    # Action: one JointPositionAction over all G1 joints.
    joint_names = env.env.action_manager.get_term("joint_pos").target_names
    action_md = _metadata(env.action_space)
    assert action_md["kind"] == "action"
    assert action_md["control"] == "joint_position"
    assert action_md["mode"] == "absolute"
    assert action_md["names"] == list(joint_names)
    assert len(action_md["scale"]) == len(action_md["offset"]) == len(joint_names)

    # Concatenated "actor" group: per-element names and a per-term breakdown.
    actor_md = _metadata(env.observation_space["actor"])
    assert len(actor_md["names"]) == env.observation_space["actor"].shape[-1]
    assert actor_md["names"][:3] == ["base_lin_vel/x", "base_lin_vel/y", "base_lin_vel/z"]
    assert f"joint_pos/{joint_names[0]}" in actor_md["names"]
    terms = {term["name"]: term for term in actor_md["terms"]}
    assert list(terms) == env.env.observation_manager.active_terms["actor"]
    assert terms["base_lin_vel"]["units"] == "m/s"  # IMU velocimeter
    assert terms["base_ang_vel"]["units"] == "rad/s"  # IMU gyro
    assert terms["joint_pos"]["units"] == ["rad"] * len(joint_names)
    assert terms["joint_vel"]["units"] == ["rad/s"] * len(joint_names)
    assert terms["actions"]["kind"] == "action"
    assert terms["command"]["kind"] == "task"
    assert terms["command"]["units"] == ["m/s", "m/s", "rad/s"]
    assert terms["actions"]["start"] == terms["joint_vel"]["stop"]

    env.close()


def test_mjlab_space_metadata_dict_group_and_history():
    from unienv_interface.space.space_utils.metadata_utils import validate_metadata

    cfg = load_env_cfg(TEST_ENV_NAME)
    cfg.scene.num_envs = 2
    cfg.observations["critic"].concatenate_terms = False
    cfg.observations["actor"].terms["projected_gravity"].history_length = 2
    env = FromMJLabEnv(ManagerBasedRlEnv(cfg=cfg, device="cpu"))
    validate_metadata(env.observation_space)

    joint_names = list(env.env.action_manager.get_term("joint_pos").target_names)
    critic = env.observation_space["critic"]
    assert _metadata(critic["joint_pos"]) == {
        "kind": "proprio",
        "names": joint_names,
        "units": ["rad"] * len(joint_names),
        "label": "joint positions relative to the default pose",
    }
    assert _metadata(critic["command"])["names"] == ["lin_vel_x", "lin_vel_y", "ang_vel_z"]

    actor_md = _metadata(env.observation_space["actor"])
    assert len(actor_md["names"]) == env.observation_space["actor"].shape[-1]
    assert "projected_gravity/x[t-1]" in actor_md["names"]
    assert "projected_gravity/z[t-0]" in actor_md["names"]

    _, obs, _ = env.reset(seed=0)
    assert obs in env.observation_space
    env.close()


@pytest.mark.parametrize("task", ["Mjlab-Lift-Cube-Yam-Rgb", "Mjlab-Lift-Cube-Yam-Depth"])
def test_mjlab_camera_metadata(task):
    from unienv_interface.space.space_utils.metadata_utils import validate_metadata

    cfg = load_env_cfg(task)
    cfg.scene.num_envs = 2
    env = FromMJLabEnv(ManagerBasedRlEnv(cfg=cfg, device="cpu"))
    validate_metadata(env.observation_space)

    camera_md = _metadata(env.observation_space["camera"])
    assert camera_md["layout"] == "CHW"
    assert camera_md["source"] == "camera_d405"
    assert camera_md["fps"] == pytest.approx(1.0 / env.env.step_dt)
    assert camera_md["normalized"] is True
    if task.endswith("Rgb"):
        assert camera_md["kind"] == "image" and camera_md["format"] == "rgb"
    else:
        assert camera_md["kind"] == "depth"
        assert camera_md["range"] == [0.0, camera_md["cutoff_distance"]]

    _, obs, _ = env.reset(seed=0)
    assert obs in env.observation_space
    env.close()


def test_mjlab_multi_term_action_metadata():
    from mjlab.envs.mdp.actions import DifferentialIKActionCfg, JointPositionActionCfg
    from unienv_interface.space.space_utils.metadata_utils import validate_metadata

    cfg = load_env_cfg("Mjlab-Lift-Cube-Yam")
    cfg.scene.num_envs = 2
    cfg.actions = {
        "arm": DifferentialIKActionCfg(
            entity_name="robot",
            actuator_names=("joint[1-6]",),
            frame_type="site",
            frame_name="grasp_site",
            delta_pos_scale=0.05,
        ),
        "gripper": JointPositionActionCfg(
            entity_name="robot", actuator_names=("left_finger",), scale=0.01
        ),
    }
    env = FromMJLabEnv(ManagerBasedRlEnv(cfg=cfg, device="cpu"))
    validate_metadata(env.action_space)

    action_md = _metadata(env.action_space)
    assert action_md["names"] == ["x", "y", "z", "rx", "ry", "rz", "left_finger"]
    assert "control" not in action_md  # the terms control different things
    arm, gripper = action_md["terms"]
    assert (arm["name"], arm["start"], arm["stop"]) == ("arm", 0, 6)
    assert arm["control"] == "eef_pose" and arm["mode"] == "delta"
    assert arm["rotation"] == "axis_angle" and arm["delta_pos_scale"] == 0.05
    assert (gripper["start"], gripper["stop"]) == (6, 7)
    assert gripper["control"] == "joint_position" and gripper["scale"] == 0.01
    env.close()
