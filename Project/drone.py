import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import argparse
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
            包含高度和垂直速度的状态数组
        """
        z = self.robot.data.root_pos_w[0, 2].item()  # 高度
        vz = self.robot.data.root_vel_w[0, 2].item()  # 垂直速度
        return np.array([z, vz], dtype=np.float32)

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
        reward = -abs(state[0] - 1.0) - 0.1 * abs(state[1])  # 奖励基于高度误差和速度
        done = state[0] < 0.1 or state[0] > 2.  # 如果高度太低或太高则结束
        print(f"Height: {state[0]}, Done: {done}")
        return state, reward, done, {}

    def sample_action(self):
        """采样一个随机动作
        
        返回:
            随机的四个电机转速
        """
        return np.random.uniform(self.min_rpm, self.max_rpm, size=(4,))

    def balanced_initial_action(self):
        """生成四个电机具有相同转速的初始动作"""
        # 选择一个在有效范围内的随机值
        rpm = np.random.uniform(self.min_rpm, self.max_rpm)
        # 为所有四个电机返回相同的值
        return np.array([rpm, rpm, rpm, rpm])

    def close(self):
        """关闭模拟器"""
        self.simulation_app.close()

class DQN(nn.Module):
    """深度Q网络模型"""
    
    def __init__(self, state_dim, action_dim=4):
        """初始化DQN网络
        
        参数:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),  # 输入层到隐藏层
            nn.ReLU(),                 # 激活函数
            nn.Linear(64, action_dim)  # 隐藏层到输出层
        )

    def forward(self, x):
        """前向传播
        
        参数:
            x: 输入状态
            
        返回:
            Q值
        """
        return self.net(x)

if __name__ == "__main__":
    # 创建环境实例
    env = HoverEnv(device="cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化Q网络和目标网络
    q_net = DQN(2, 4).to(env.device)
    target_net = DQN(2, 4).to(env.device)
    target_net.load_state_dict(q_net.state_dict())
    # optimizer = optim.Adam(q_net.parameters(), lr=1e-3)  # 优化器（已注释）

    # 初始化经验回放缓冲区和训练参数
    buffer = deque(maxlen=10000)  # 经验回放缓冲区
    batch_size = 64               # 批次大小
    gamma = 0.99                  # 折扣因子
    epsilon = 1.0                 # 初始探索率
    epsilon_decay = 0.995         # 探索率衰减
    epsilon_min = 0.1             # 最小探索率

    # 训练循环
    for episode in range(500):
        state = env.reset()
        total_reward = 0
        
        # 单个回合内的步骤
        for step in range(300):
            # 将状态转换为张量
            state_tensor = torch.tensor(state, device=env.device).float()
            
            # 使用Q网络选择动作
            with torch.no_grad():
                action = q_net(state_tensor).cpu().numpy()

            # ε-贪婪策略 - 在前100个回合使用平衡转速，之后使用完全随机动作
            if random.random() < epsilon:
                if episode < 100:  # 训练初期使用平衡转速
                    action = env.balanced_initial_action()
                else:  # 之后使用完全随机动作
                    action = env.sample_action()  # 随机探索

            # 执行动作并获取下一个状态
            next_state, reward, done, _ = env.step(action)
            
            # 存储经验到缓冲区
            buffer.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            # 当缓冲区足够大时进行学习
            if len(buffer) >= batch_size:
                # 从缓冲区采样批次数据
                batch = random.sample(buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                # 转换为张量
                states = torch.tensor(states, device=env.device).float()
                actions = torch.tensor(actions, device=env.device).float()
                rewards = torch.tensor(rewards, device=env.device).float()
                next_states = torch.tensor(next_states, device=env.device).float()
                dones = torch.tensor(dones, device=env.device).float()

                # 计算当前Q值和目标Q值
                q_values = q_net(states)
                q_selected = q_values
                next_q = target_net(next_states)
                target = rewards.unsqueeze(1) + gamma * next_q.mean(1, keepdim=True) * (1 - dones.unsqueeze(1))

                # 计算损失并反向传播（但未进行参数更新，因为优化器被注释了）
                loss = nn.MSELoss()(q_selected.mean(1, keepdim=True), target.detach())
                # optimizer.zero_grad()
                loss.backward()
                # optimizer.step()

            # 如果回合结束则退出
            if done:
                print(f"Episode {episode}, Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")
                break

        # 衰减探索率
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        
        # 定期更新目标网络
        if episode % 10 == 0:
            target_net.load_state_dict(q_net.state_dict())
            print(f"Episode {episode}, Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")

    # 关闭环境
    env.close()