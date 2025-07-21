from __future__ import annotations

import os
import random
import numpy as np
import torch
from isaacgym import gymapi, gymutil
from typing import Tuple
from gymnasium import VecEnv
from train_heinrich.go2_robot import LeggedRobot
from train_heinrich.heinrich_isaac_gym_config import LeggedRobotCfg, GO2RoughCfg

TASKS = {
    "go2_base": (LeggedRobot, LeggedRobotCfg()),
    "go2_rough": (LeggedRobot, GO2RoughCfg()),
}


def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_sim_params(args, cfg):
    # code from Isaac Gym Preview 2
    # initialize sim params
    sim_params = gymapi.SimParams()

    # set some values from args
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


def make_env(env_config) -> Tuple[VecEnv, LeggedRobotCfg]:
    """Creates an environment either from a registered namme or from the provided config file.

    Args:
        name (string): Name of a registered env.
        args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None.
        env_cfg (Dict, optional): Environment config file used to override the registered config. Defaults to None.

    Raises:
        ValueError: Error if no registered env corresponds to 'name'

    Returns:
        isaacgym.VecTaskPython: The created environment
        Dict: the corresponding config file
    """
    set_seed(env_config.seed)
    task_class, task_config = TASKS.get(env_config.task_name, (None, None))
    sim_params = {"sim": class_to_dict(task_config.sim)}
    sim_params = parse_sim_params(env_config.sim_args, sim_params)
    task_config.env.num_envs = env_config.num_envs
    if task_class is None:
        raise ValueError(f"Task with name: {env_config.task_name} was not registered")
    for task_arg in env_config.task_args:
        if hasattr(task_config, task_arg):
            setattr(task_config, task_arg, env_config.task_args[task_arg])
        else:
            print(
                f"Warning: Task config does not have argument {task_arg}, ignoring it."
            )
    env = task_class(
        cfg=task_config,
        sim_params=sim_params,
        physics_engine=env_config.sim_args.physics_engine,
        sim_device=env_config.sim_device,
        headless=env_config.headless,
    )
    return env
