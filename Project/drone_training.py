import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 首先初始化Isaac Sim环境
from isaaclab.app import AppLauncher

# 解析参数
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 启动应用 - 这将初始化Isaac Sim环境
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 在启动应用后，再导入依赖Isaac Sim的模块
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab_assets import CRAZYFLIE_CFG
import isaacsim.core.utils.prims as prim_utils

class DQN(nn.Module):
    """深度Q网络"""
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def design_scene():
    """设计场景"""
    # 地面
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # 灯光
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # 创建无人机组
    prim_utils.create_prim("/World/Origin1", "Xform", translation=(0, 0, 1.0))  # 修改为tuple
    
    # 修复配置类问题 - 假设CRAZYFLIE_CFG是一个配置对象
    # 检查CRAZYFLIE_CFG的类型和结构，根据实际情况调整
    drone_cfg = CRAZYFLIE_CFG.copy()  # 假设有copy方法而不是replace
    drone_cfg.prim_path = "/World/Origin1/Robot"  # 直接设置属性
    drone = Articulation(drone_cfg)
    
    return {"drone": drone}, [[0.0, 0.0, 1.0]]

def run_simulator(sim: sim_utils.SimulationContext, entities: dict, origins: torch.Tensor):
    """运行训练循环"""
    # 初始化 DQN
    state_size = 6  # 位置(3) + 速度(3)
    action_size = 4  # 推力(1) + 力矩(3)
    model = DQN(state_size, action_size).to(sim.device)
    target_model = DQN(state_size, action_size).to(sim.device)
    target_model.load_state_dict(model.state_dict())
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练参数
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    gamma = 0.99
    target_height = 1.0  # 目标悬停高度

    # 获取无人机实例
    drone = entities["drone"]
    
    # 定义仿真步长
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    episode = 0
    step = 0

    # 仿真主循环
    while simulation_app.is_running():
        # 每200步重置一次
        if step % 200 == 0:
            # 重置计数器
            sim_time = 0.0
            step = 0
            episode += 1

            # 重置无人机状态
            root_state = drone.data.default_root_state.clone()
            root_state[:, :3] += origins[0]
            drone.write_root_pose_to_sim(root_state[:, :7])
            drone.write_root_velocity_to_sim(root_state[:, 7:])
            drone.reset()
            print(f"[INFO]: Episode {episode} starting...")

        # 获取当前状态
        pos = drone.data.root_pos_w
        vel = drone.data.root_lin_vel_w
        state = torch.cat([pos, vel], dim=-1).to(sim.device)

        # 选择动作
        if np.random.random() < epsilon:
            action = (torch.rand(4, device=sim.device) * 2 - 1)
        else:
            with torch.no_grad():
                q_values = model(state)
                # 将q_values转换为一维张量
                action = torch.tanh(q_values).squeeze()

        # 应用动作
        thrust = torch.zeros(1, 1, 3, device=sim.device)
        # 修改这行 - 使用索引0获取第一个元素
        thrust[0, 0, 2] = (action[0].item() + 1) * 10.0
        moment = torch.zeros(1, 1, 3, device=sim.device)
        if action.shape[0] >= 4:  # 确保有足够的元素
            moment[0, 0, 0] = action[1] * 0.1
            moment[0, 0, 1] = action[2] * 0.1
            moment[0, 0, 2] = action[3] * 0.1
        
        # 设置推力和力矩
        drone.set_external_force_and_torque(thrust, moment)
        drone.write_data_to_sim()

        # 执行仿真步骤
        sim.step()
        drone.update(sim_dt)

        # 计算奖励
        height_error = abs(pos[0, 2] - target_height)
        velocity_penalty = torch.norm(vel[0])
        reward = -(height_error + 0.1 * velocity_penalty)

        # 获取下一个状态
        next_pos = drone.data.root_pos_w
        next_vel = drone.data.root_lin_vel_w
        next_state = torch.cat([next_pos, next_vel], dim=-1).to(sim.device)

        # 训练 DQN
        if not torch.isnan(reward):  # 确保奖励是有效值
            optimizer.zero_grad()
            current_q = model(state)
            with torch.no_grad():
                next_q = target_model(next_state)
                target_q = reward + gamma * next_q.max()
            
            loss = nn.MSELoss()(current_q, target_q.unsqueeze(0))
            loss.backward()
            optimizer.step()

        # 更新目标网络
        if step % 50 == 0:
            target_model.load_state_dict(model.state_dict())
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # 更新计数器
        sim_time += sim_dt
        step += 1

        # 每100步保存一次模型
        if step % 100 == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'episode': episode,
                'step': step,
            }, f"drone_model_ep{episode}_step{step}.pth")

        # 打印训练信息
        if step % 10 == 0:
            print(f"Episode: {episode}, Step: {step}, Reward: {reward:.2f}, Height: {pos[0, 2]:.2f}, Epsilon: {epsilon:.2f}")

def main():
    

    # 初始化仿真
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01))
    sim.set_camera_view(eye=(2.5, 2.5, 2.5), target=(0.0, 0.0, 0.0))
    
    # 设计场景
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    
    # 开始仿真
    sim.reset()
    print("[INFO]: Setup complete...")
    
    # 运行训练
    run_simulator(sim, scene_entities, scene_origins)

if __name__ == "__main__":
    # 运行主函数
    main()
    # 关闭仿真应用
    simulation_app.close()