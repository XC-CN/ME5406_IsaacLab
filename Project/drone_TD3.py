import torch
import numpy as np
import argparse
import os
import gym
from gym import spaces
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback
from isaaclab.app import AppLauncher

def rpm_to_force(rpm, k_f=1e-6):
    """将转速(RPM)转换为推力
    
    参数:
        rpm: 电机转速（每分钟转数）
        k_f: 推力系数
    
    返回:
        计算得到的推力值
    """
    rad_s = rpm * 2 * 3.14159 / 60.0  # 转换为弧度/秒
    return k_f * rad_s**2  # 推力与角速度的平方成正比

class HoverEnv:
    """无人机悬停环境类，用于模拟四旋翼无人机的悬停任务"""
    
    def __init__(self, device="cpu"):
        """初始化悬停环境
        
        参数:
            device: 运行设备，可以是'cpu'或'cuda'
        """
        self.device = device
        self.min_rpm = 5000  # 最小电机转速
        self.max_rpm = 5500  # 最大电机转速

        # 初始化Isaac模拟器
        parser = argparse.ArgumentParser()
        AppLauncher.add_app_launcher_args(parser)
        args_cli = parser.parse_args(args=[])
        self.app_launcher = AppLauncher(args_cli)
        self.simulation_app = self.app_launcher.app

        import isaaclab.sim as sim_utils
        from isaaclab.assets import Articulation
        from isaaclab_assets import CRAZYFLIE_CFG

        # 配置模拟环境
        sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=device)  # 设置时间步长和设备
        self.sim = sim_utils.SimulationContext(sim_cfg)
        self.sim.set_camera_view(eye=[1.0, 1.0, 1.0], target=[0.0, 0.0, 0.5])  # 设置相机视角

        # 创建地面
        ground = sim_utils.GroundPlaneCfg()
        ground.func("/World/defaultGroundPlane", ground)

        # 添加光源
        light = sim_utils.DistantLightCfg(intensity=3000.0, color=(1.0, 1.0, 1.0))
        light.func("/World/Light", light)

        # 配置无人机模型
        robot_cfg = CRAZYFLIE_CFG.replace(prim_path="/World/Crazyflie")
        robot_cfg.spawn.func("/World/Crazyflie", robot_cfg.spawn, translation=robot_cfg.init_state.pos)
        self.robot = Articulation(robot_cfg)

        # 重置模拟环境
        self.sim.reset()
        self.prop_ids = self.robot.find_bodies("m.*_prop")[0]  # 获取螺旋桨的ID
        self.dt = self.sim.get_physics_dt()  # 获取物理时间步长

    def reset(self):
        """重置环境到初始状态
        
        返回:
            初始状态
        """
        # 重置关节状态和位置
        self.robot.write_joint_state_to_sim(self.robot.data.default_joint_pos, self.robot.data.default_joint_vel)
        self.robot.write_root_pose_to_sim(self.robot.data.default_root_state[:, :7])
        self.robot.write_root_velocity_to_sim(self.robot.data.default_root_state[:, 7:])
        self.robot.reset()
        self.sim.step()
        self.robot.update(self.dt)
        return self._get_state()

    def _get_state(self):
        """获取当前环境状态
        
        返回:
            包含高度、垂直速度和倾角的状态数组
        """
        z = self.robot.data.root_pos_w[0, 2].item()  # 高度
        vz = self.robot.data.root_vel_w[0, 2].item()  # 垂直速度
        
        # 获取四元数 - 使用正确的root_quat_w变量
        quat = self.robot.data.root_quat_w[0]  # 获取四元数 [qw, qx, qy, qz]
        
        # 计算俯仰角和横滚角
        qw, qx, qy, qz = quat.cpu().numpy()
        # 简化的欧拉角计算（使用四元数估算倾角）
        roll = np.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
        pitch = np.arcsin(2.0 * (qw * qy - qz * qx))
        
        # 计算总倾角（俯仰角和横滚角的平方和的平方根）
        tilt_angle = np.sqrt(roll**2 + pitch**2)
        
        return np.array([z, vz, tilt_angle], dtype=np.float32)

    def step(self, rpms):
        """执行一步模拟
        
        参数:
            rpms: 四个电机的转速
            
        返回:
            新状态、奖励、是否结束、额外信息
        """
        # 限制转速在合理范围内
        rpms = np.clip(rpms, self.min_rpm, self.max_rpm)
        
        # 计算作用于螺旋桨的力和力矩
        forces = torch.zeros(1, 4, 3, device=self.device)
        torques = torch.zeros_like(forces)
        for i in range(4):
            thrust = rpm_to_force(rpms[i])
            forces[0, i, 2] = thrust  # 力作用在z轴方向

        # 施加外力和力矩
        self.robot.set_external_force_and_torque(forces, torques, body_ids=self.prop_ids)
        self.robot.write_data_to_sim()

        # 执行多个物理模拟步骤
        for _ in range(4):
            self.sim.step()
            self.robot.update(self.dt)

        # 获取新状态并计算奖励
        state = self._get_state()
        
        # 使用高度和倾角计算奖励
        height_reward = -abs(state[0] - 1.0)  # 高度奖励
        velocity_reward = -0.1 * abs(state[1])  # 速度奖励
        tilt_penalty = -2.0 * state[2]  # 倾角惩罚，增大惩罚系数
        
        reward = height_reward + velocity_reward + tilt_penalty
        
        # 使用倾角作为主要结束条件
        max_tilt_angle = 0.5  # 大约30度
        done = state[2] > max_tilt_angle or state[0] < 0.1 or state[0] > 2.0
        
        print(f"\r高度: {state[0]:.2f}, 倾角: {state[2]:.2f}, 结束: {done}", end="")
        return state, reward, done, {}

    def close(self):
        """关闭模拟器"""
        self.simulation_app.close()

class IsaacDroneEnv(gym.Env):
    """符合Gym接口的无人机环境包装器，用于与Stable Baselines3兼容"""
    
    def __init__(self, device="cpu"):
        """初始化Gym兼容的环境包装器
        
        参数:
            device: 运行设备，可以是'cpu'或'cuda'
        """
        super(IsaacDroneEnv, self).__init__()
        
        # 创建原始环境
        self.isaac_env = HoverEnv(device=device)
        
        # 定义动作空间 - 四个电机的转速偏移量
        # TD3以[-1,1]范围输出动作，我们将映射到实际的转速范围
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(4,),
            dtype=np.float32
        )
        
        # 定义观察空间 - [高度, 垂直速度, 倾角]
        self.observation_space = spaces.Box(
            low=np.array([0.0, -np.inf, 0.0]),
            high=np.array([2.0, np.inf, np.pi/2]),
            dtype=np.float32
        )
        
        # 保存上一个状态以计算变化率
        self.prev_state = None
        
        # 设置基础转速
        self.base_rpm = (self.isaac_env.min_rpm + self.isaac_env.max_rpm) / 2
        self.rpm_range = (self.isaac_env.max_rpm - self.isaac_env.min_rpm) / 2
    
    def reset(self):
        """重置环境，返回初始观察"""
        state = self.isaac_env.reset()
        self.prev_state = state
        return state
    
    def step(self, action):
        """执行一步，接收TD3算法的动作，映射到实际转速"""
        # 将[-1,1]范围的动作转换为实际的电机转速
        # 动作表示转速的偏移量，加上基础转速得到最终转速
        rpms = self.base_rpm + action * self.rpm_range
        
        # 执行动作
        next_state, reward, done, info = self.isaac_env.step(rpms)
        
        # 计算状态变化率，增强奖励信号
        if self.prev_state is not None:
            # 惩罚状态的快速变化(抖动)
            state_change = np.abs(next_state - self.prev_state)
            # 特别关注倾角变化率
            tilt_change_penalty = -1.0 * state_change[2]
            # 添加到奖励中
            reward += tilt_change_penalty
        
        self.prev_state = next_state
        return next_state, reward, done, info
    
    def close(self):
        """关闭环境"""
        self.isaac_env.close()

def train_drone():
    """训练无人机使用TD3算法"""
    # 创建保存模型的目录
    log_dir = "./logs/td3_drone"
    os.makedirs(log_dir, exist_ok=True)
    model_dir = "./models/td3_drone"
    os.makedirs(model_dir, exist_ok=True)
    
    # 创建环境
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = IsaacDroneEnv(device=device)
    
    # 设置动作噪声以促进探索
    n_actions = env.action_space.shape[0]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions)  # 探索噪声大小
    )
    
    # 创建TD3模型
    model = TD3(
        "MlpPolicy",
        env,
        action_noise=action_noise,
        learning_rate=3e-4,
        buffer_size=10000,  # 回放缓冲区大小
        learning_starts=1000,  # 开始学习前收集的样本数
        batch_size=100,
        gamma=0.99,  # 折扣因子
        tau=0.005,  # 目标网络软更新系数
        policy_delay=2,  # 延迟策略更新
        target_policy_noise=0.2,  # 目标动作噪声
        target_noise_clip=0.5,  # 目标噪声裁剪
        verbose=1,
        tensorboard_log=log_dir
    )
    
    # 设置回调以定期保存模型
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,  # 每5000步保存一次
        save_path=model_dir,
        name_prefix="td3_drone_model"
    )
    
    # 开始训练
    print("开始训练TD3算法，用于无人机悬停...")
    model.learn(
        total_timesteps=100000,  # 总训练步数
        callback=checkpoint_callback
    )
    
    # 保存最终模型
    model.save(f"{model_dir}/td3_drone_final")
    print(f"训练完成，模型已保存到 {model_dir}")

def test_drone(model_path=None):
    """测试训练好的无人机模型"""
    # 创建环境
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = IsaacDroneEnv(device=device)
    
    # 加载模型
    if model_path:
        model = TD3.load(model_path, env=env)
        print(f"已加载模型: {model_path}")
    else:
        model_path = "./models/td3_drone/td3_drone_final"
        model = TD3.load(model_path, env=env)
        print(f"已加载模型: {model_path}")
    
    # 执行测试
    obs = env.reset()
    done = False
    total_reward = 0
    
    print("开始测试无人机悬停控制...")
    while not done:
        # 使用确定性策略进行预测（无探索）
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
    
    print(f"\n测试完成，总奖励: {total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TD3算法训练无人机悬停")
    parser.add_argument("--test", action="store_true", help="测试模式而非训练模式")
    parser.add_argument("--model", type=str, default=None, help="测试时使用的模型路径")
    
    args = parser.parse_args()
    
    if args.test:
        test_drone(args.model)
    else:
        train_drone() 