# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""使用RL-Games播放RL智能体的检查点的脚本。"""

"""首先启动Isaac Sim模拟器。"""

import argparse

from isaaclab.app import AppLauncher

# 添加命令行参数
parser = argparse.ArgumentParser(description="播放RL-Games中RL智能体的检查点。")
parser.add_argument("--video", action="store_true", default=False, help="在训练过程中录制视频。")
parser.add_argument("--video_length", type=int, default=200, help="录制视频的长度（步数）。")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="禁用fabric并使用USD I/O操作。"
)
parser.add_argument("--num_envs", type=int, default=None, help="要模拟的环境数量。")
parser.add_argument("--task", type=str, default=None, help="任务名称。")
parser.add_argument("--checkpoint", type=str, default=None, help="模型检查点路径。")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="使用Nucleus中的预训练检查点。",
)
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="当未提供检查点时，使用最后保存的模型。否则使用最佳保存模型。",
)
parser.add_argument("--real-time", action="store_true", default=False, help="如果可能，以实时方式运行。")
# 添加AppLauncher命令行参数
AppLauncher.add_app_launcher_args(parser)
# 解析参数
args_cli = parser.parse_args()
# 如果录制视频，始终启用相机
if args_cli.video:
    args_cli.enable_cameras = True

# 启动Omniverse应用程序
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""接下来是其余部分。"""


import gymnasium as gym
import math
import os
import time
import torch

from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg

# 占位符：扩展模板（请勿删除此注释）


def main():
    """使用RL-Games智能体进行游戏。"""
    # 解析环境配置
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")

    # 指定实验日志目录
    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[信息] 从目录加载实验：{log_root_path}")
    # 查找检查点
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rl_games", args_cli.task)
        if not resume_path:
            print("[信息] 很遗憾，目前此任务没有可用的预训练检查点。")
            return
    elif args_cli.checkpoint is None:
        # 指定运行日志目录
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        # 指定检查点名称
        if args_cli.use_last_checkpoint:
            checkpoint_file = ".*"
        else:
            # 加载最佳检查点
            checkpoint_file = f"{agent_cfg['params']['config']['name']}.pth"
        # 获取之前检查点的路径
        resume_path = get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
    else:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # 为rl-games包装环境
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    # 创建isaac环境
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # 如果RL算法需要，将多智能体实例转换为单智能体实例
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # 包装视频录制
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[信息] 在训练过程中录制视频。")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # 为rl-games包装环境
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    # 将环境注册到rl-games注册表
    # 注意：在智能体配置中：环境名称必须是"rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # 加载之前训练的模型
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    print(f"[信息]: 从以下位置加载模型检查点：{agent_cfg['params']['load_path']}")

    # 将执行者数量设置到智能体配置中
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    # 从rl-games创建运行器
    runner = Runner()
    runner.load(agent_cfg)
    # 从运行器获取智能体
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    dt = env.unwrapped.step_dt

    # 重置环境
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    timestep = 0
    # 必需：启用批量观察的标志
    _ = agent.get_batch_size(obs, 1)
    # 如果使用RNN，初始化RNN状态
    if agent.is_rnn:
        agent.init_rnn()
    # 模拟环境
    # 注意：我们简化了rl-games player.py中的逻辑（:func:`BasePlayer.run()`函数）
    #   以尝试完全控制环境步进。但是，这会移除其他操作，
    #   例如RL-Games用于多智能体学习的掩码。
    while simulation_app.is_running():
        start_time = time.time()
        # 在推理模式下运行所有内容
        with torch.inference_mode():
            # 将观察转换为智能体格式
            obs = agent.obs_to_torch(obs)
            # 智能体步进
            actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)
            # 环境步进
            obs, _, dones, _ = env.step(actions)

            # 对已终止的情节执行操作
            if len(dones) > 0:
                # 为已终止的情节重置rnn状态
                if agent.is_rnn and agent.states is not None:
                    for s in agent.states:
                        s[:, dones, :] = 0.0
        if args_cli.video:
            timestep += 1
            # 录制一个视频后退出播放循环
            if timestep == args_cli.video_length:
                break

        # 实时评估的时间延迟
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # 关闭模拟器
    env.close()


if __name__ == "__main__":
    # 运行主函数
    main()
    # 关闭模拟应用程序
    simulation_app.close()
