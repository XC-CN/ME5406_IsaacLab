import torch
import numpy as np
import argparse
import time
from isaaclab.app import AppLauncher
from drone_TD3 import HoverEnv, rpm_to_force

def find_hover_rpm(start_rpm=5100, end_rpm=5200, step=10, test_duration=300, device="cpu"):
    """
    通过二分搜索找到最佳悬停RPM值
    
    参数:
        start_rpm: 搜索起始RPM值
        end_rpm: 搜索结束RPM值
        step: 搜索步长
        test_duration: 每个RPM值测试的持续时间（步数）
        device: 运行设备，'cpu'或'cuda'
    
    返回:
        最佳悬停RPM值
    """
    print(f"开始寻找最佳悬停RPM，搜索范围: {start_rpm} - {end_rpm}，步长: {step}")
    print(f"注意: 四个电机将保持完全相同的转速")
    
    rpm_results = {}  # 存储每个RPM值的测试结果
    next_rpm_suggestion = None  # 基于失控原因的下一个建议RPM值
    
    # 在给定范围内测试不同的RPM值
    rpm_list = list(range(start_rpm, end_rpm + 1, step))
    i = 0
    while i < len(rpm_list):
        # 如果有建议的RPM值，优先测试它（如果它在我们的范围内且尚未测试）
        if next_rpm_suggestion is not None and start_rpm <= next_rpm_suggestion <= end_rpm and next_rpm_suggestion not in rpm_results:
            rpm = next_rpm_suggestion
            next_rpm_suggestion = None  # 清空建议
        else:
            rpm = rpm_list[i]
            i += 1
            
        print(f"\n测试RPM值: {rpm}")
        
        # 创建环境
        env = HoverEnv(device=device, headless=False, verbose=True)
        env.hover_rpm = rpm
        
        # 重置环境
        state, _ = env.reset()
        
        # 记录测试数据
        heights = []
        tilts = []
        x_positions = []  # 记录X位置
        y_positions = []  # 记录Y位置
        stable_steps = 0
        max_stable_steps = 0
        failure_reason = "完成全部测试步数"  # 默认无失控
        last_state = None
        
        # 运行测试
        for step_idx in range(test_duration):
            # 使用恒定RPM值
            rpms = np.array([rpm, rpm, rpm, rpm])
            
            # 执行一步
            next_state, reward, done, info = env.step(rpms)
            last_state = next_state
            
            # 提取状态
            x, y, z = next_state[0], next_state[1], next_state[2]  # 位置
            vx, vy, vz = next_state[3], next_state[4], next_state[5]  # 速度
            tilt = next_state[6]  # 倾角
            
            # 记录数据
            heights.append(z)
            tilts.append(tilt)
            x_positions.append(x)
            y_positions.append(y)
            
            # 检查稳定性
            if abs(z - 1.0) < 0.15 and tilt < 0.15:
                stable_steps += 1
                max_stable_steps = max(max_stable_steps, stable_steps)
            else:
                stable_steps = 0
            
            # 显示进度，增加水平位置信息
            if step_idx % 20 == 0:
                print(f"\r步数: {step_idx}/{test_duration}, 高度: {z:.2f}, 水平位置: ({x:.2f}, {y:.2f}), "
                      f"倾角: {tilt:.3f}, 当前连续稳定步数: {stable_steps}", end="")
            
            # 如果已经倒下或飞太高，提前结束并分析原因
            if done:
                if z < 0.3:
                    failure_reason = "高度过低（坠毁）"
                    next_rpm_suggestion = rpm + 20  # 建议增加RPM
                elif z > 1.8:
                    failure_reason = "高度过高（飞远）"
                    next_rpm_suggestion = rpm - 20  # 建议减少RPM
                elif tilt > 0.5:
                    failure_reason = "倾角过大（翻转）"
                    # 不改变RPM，因为这可能是初始条件问题而非RPM问题
                elif abs(vz) > 1.5:
                    failure_reason = "垂直速度过大"
                    if vz > 0:  # 上升速度过大
                        next_rpm_suggestion = rpm - 10
                    else:  # 下降速度过大
                        next_rpm_suggestion = rpm + 10
                elif np.sqrt(x**2 + y**2) > 1.5:
                    failure_reason = "水平偏移过大"
                    # 水平偏移通常不是由RPM整体大小引起，而是由转速不平衡引起
                    # 在这个脚本中四个电机转速相同，所以不调整RPM
                
                print(f"\n无人机在{step_idx}步后失控，原因: {failure_reason}")
                break
            
            # 等待一小段时间使显示更流畅
            time.sleep(0.01)
        
        # 关闭环境
        env.close()
        
        # 计算该RPM的综合得分
        if heights:
            height_mean = np.mean(heights)
            height_std = np.std(heights)
            tilt_mean = np.mean(tilts)
            tilt_std = np.std(tilts)
            
            # 计算水平位置漂移
            x_std = np.std(x_positions)
            y_std = np.std(y_positions)
            horizontal_drift = np.sqrt(x_std**2 + y_std**2)
            
            # 计算高度误差
            height_error = abs(height_mean - 1.0)
            
            # 计算得分（越高越好）
            # 稳定性得分 = 最大连续稳定步数
            # 高度得分 = -高度误差
            # 倾角得分 = -平均倾角
            # 抖动得分 = -(高度标准差 + 倾角标准差)
            # 漂移得分 = -水平漂移
            stability_score = max_stable_steps
            height_score = -height_error
            tilt_score = -tilt_mean
            jitter_score = -(height_std + tilt_std)
            drift_score = -horizontal_drift
            
            # 综合得分
            total_score = (
                5.0 * stability_score +  # 稳定性最重要
                3.0 * height_score +     # 高度次之
                2.0 * tilt_score +       # 倾角再次
                1.0 * jitter_score +     # 抖动
                1.0 * drift_score        # 水平漂移
            )
            
            # 保存结果
            rpm_results[rpm] = {
                'total_score': total_score,
                'stability_score': stability_score,
                'height_mean': height_mean,
                'height_std': height_std,
                'tilt_mean': tilt_mean,
                'tilt_std': tilt_std,
                'horizontal_drift': horizontal_drift,
                'max_stable_steps': max_stable_steps,
                'failure_reason': failure_reason
            }
            
            print(f"\nRPM {rpm} 测试结果:")
            print(f"  平均高度: {height_mean:.3f} ± {height_std:.3f}")
            print(f"  平均倾角: {tilt_mean:.3f} ± {tilt_std:.3f}")
            print(f"  水平漂移: {horizontal_drift:.3f}")
            print(f"  最大连续稳定步数: {max_stable_steps}")
            print(f"  终止原因: {failure_reason}")
            print(f"  总得分: {total_score:.2f}")
        
            # 如果有基于失控原因的建议，显示出来
            if next_rpm_suggestion is not None:
                print(f"  基于失控原因的下一个建议RPM值: {next_rpm_suggestion}")
    
    # 找出得分最高的RPM
    if rpm_results:
        best_rpm = max(rpm_results.items(), key=lambda x: x[1]['total_score'])[0]
        best_score = rpm_results[best_rpm]['total_score']
        
        print("\n\n========== 搜索结果 ==========")
        print(f"最佳悬停RPM值: {best_rpm}")
        print(f"最佳得分: {best_score:.2f}")
        print(f"平均高度: {rpm_results[best_rpm]['height_mean']:.3f} ± {rpm_results[best_rpm]['height_std']:.3f}")
        print(f"平均倾角: {rpm_results[best_rpm]['tilt_mean']:.3f} ± {rpm_results[best_rpm]['tilt_std']:.3f}")
        print(f"水平漂移: {rpm_results[best_rpm]['horizontal_drift']:.3f}")
        print(f"最大连续稳定步数: {rpm_results[best_rpm]['max_stable_steps']}")
        print(f"终止原因: {rpm_results[best_rpm]['failure_reason']}")
        
        # 打印所有结果，按得分排序
        print("\n所有测试结果（按得分排序）:")
        sorted_results = sorted(rpm_results.items(), key=lambda x: x[1]['total_score'], reverse=True)
        for rpm, result in sorted_results:
            print(f"RPM {rpm}: 得分={result['total_score']:.2f}, "
                  f"高度={result['height_mean']:.3f}±{result['height_std']:.3f}, "
                  f"倾角={result['tilt_mean']:.3f}±{result['tilt_std']:.3f}, "
                  f"水平漂移={result['horizontal_drift']:.3f}, "
                  f"稳定步数={result['max_stable_steps']}, "
                  f"终止原因={result['failure_reason']}")
        
        # 根据最佳RPM建议微调
        print("\n最佳RPM微调建议:")
        if rpm_results[best_rpm]['height_mean'] > 1.05:
            print(f"  高度偏高，建议尝试稍低的RPM: {best_rpm - 5}")
        elif rpm_results[best_rpm]['height_mean'] < 0.95:
            print(f"  高度偏低，建议尝试稍高的RPM: {best_rpm + 5}")
        else:
            print(f"  高度非常接近目标值，无需调整")
            
        return best_rpm
    else:
        print("没有找到有效的RPM值")
        return None

def binary_search_rpm(min_rpm=4800, max_rpm=5500, precision=5, test_duration=200, device="cpu"):
    """
    使用二分搜索算法更快地找到最佳悬停RPM值
    
    参数:
        min_rpm: 最小RPM值
        max_rpm: 最大RPM值
        precision: 搜索精度
        test_duration: 每个RPM值测试的持续时间（步数）
        device: 运行设备
    
    返回:
        最佳悬停RPM值
    """
    print(f"使用二分搜索寻找最佳悬停RPM，搜索范围: {min_rpm} - {max_rpm}，精度: {precision}")
    
    # 记录所有测试过的RPM值的结果
    all_results = {}
    failure_analysis = {}  # 记录各种失控原因
    
    # 当范围足够小时停止搜索
    while max_rpm - min_rpm > precision:
        mid_rpm = (min_rpm + max_rpm) // 2
        left_rpm = (min_rpm + mid_rpm) // 2
        right_rpm = (mid_rpm + max_rpm) // 2
        
        # 测试三个点
        points = [left_rpm, mid_rpm, right_rpm]
        scores = {}
        
        for rpm in points:
            print(f"\n测试RPM值: {rpm}")
            
            # 创建环境
            env = HoverEnv(device=device, headless=False, verbose=True)
            env.hover_rpm = rpm
            
            # 重置环境
            state, _ = env.reset()
            
            # 记录测试数据
            heights = []
            tilts = []
            x_positions = []
            y_positions = []
            stable_steps = 0
            max_stable_steps = 0
            failure_reason = "完成全部测试步数"
            
            # 运行测试
            for step_idx in range(test_duration):
                # 使用恒定RPM值
                rpms = np.array([rpm, rpm, rpm, rpm])
                
                # 执行一步
                next_state, reward, done, info = env.step(rpms)
                
                # 提取状态
                x, y, z = next_state[0], next_state[1], next_state[2]  # 位置
                vx, vy, vz = next_state[3], next_state[4], next_state[5]  # 速度
                tilt = next_state[6]  # 倾角
                
                # 记录数据
                heights.append(z)
                tilts.append(tilt)
                x_positions.append(x)
                y_positions.append(y)
                
                # 检查稳定性
                if abs(z - 1.0) < 0.15 and tilt < 0.15:
                    stable_steps += 1
                    max_stable_steps = max(max_stable_steps, stable_steps)
                else:
                    stable_steps = 0
                
                # 显示进度
                if step_idx % 20 == 0:
                    print(f"\r步数: {step_idx}/{test_duration}, 高度: {z:.2f}, 水平位置: ({x:.2f}, {y:.2f}), "
                          f"倾角: {tilt:.3f}, 当前连续稳定步数: {stable_steps}", end="")
                
                # 如果已经倒下或飞太高，提前结束并分析原因
                if done:
                    if z < 0.3:
                        failure_reason = "高度过低（坠毁）"
                    elif z > 1.8:
                        failure_reason = "高度过高（飞远）"
                    elif tilt > 0.5:
                        failure_reason = "倾角过大（翻转）"
                    elif abs(vz) > 1.5:
                        failure_reason = "垂直速度过大"
                    elif np.sqrt(x**2 + y**2) > 1.5:
                        failure_reason = "水平偏移过大"
                    
                    print(f"\n无人机在{step_idx}步后失控，原因: {failure_reason}")
                    
                    # 记录失控原因
                    if failure_reason not in failure_analysis:
                        failure_analysis[failure_reason] = []
                    failure_analysis[failure_reason].append(rpm)
                    
                    break
            
            # 关闭环境
            env.close()
            
            # 计算该RPM的综合得分
            if heights:
                height_mean = np.mean(heights)
                height_std = np.std(heights)
                tilt_mean = np.mean(tilts)
                tilt_std = np.std(tilts)
                
                # 计算水平位置漂移
                x_std = np.std(x_positions)
                y_std = np.std(y_positions)
                horizontal_drift = np.sqrt(x_std**2 + y_std**2)
                
                # 计算高度误差
                height_error = abs(height_mean - 1.0)
                
                # 计算得分（越高越好）
                stability_score = max_stable_steps
                height_score = -height_error
                tilt_score = -tilt_mean
                jitter_score = -(height_std + tilt_std)
                drift_score = -horizontal_drift
                
                # 综合得分
                total_score = (
                    5.0 * stability_score +
                    3.0 * height_score +
                    2.0 * tilt_score +
                    1.0 * jitter_score +
                    1.0 * drift_score
                )
                
                # 将结果添加到全局结果字典
                all_results[rpm] = {
                    'total_score': total_score,
                    'height_mean': height_mean,
                    'height_std': height_std,
                    'tilt_mean': tilt_mean,
                    'tilt_std': tilt_std,
                    'horizontal_drift': horizontal_drift,
                    'max_stable_steps': max_stable_steps,
                    'failure_reason': failure_reason
                }
                
                scores[rpm] = total_score
                
                print(f"\nRPM {rpm} 测试结果:")
                print(f"  平均高度: {height_mean:.3f} ± {height_std:.3f}")
                print(f"  平均倾角: {tilt_mean:.3f} ± {tilt_std:.3f}")
                print(f"  水平漂移: {horizontal_drift:.3f}")
                print(f"  最大连续稳定步数: {max_stable_steps}")
                print(f"  终止原因: {failure_reason}")
                print(f"  总得分: {total_score:.2f}")
        
        # 找出得分最高的点并更新搜索范围
        if not scores:
            print("警告：本轮没有有效得分，尝试调整搜索范围")
            # 如果失控分析显示RPM普遍过高，向下调整范围
            if "高度过高（飞远）" in failure_analysis:
                high_rpms = failure_analysis["高度过高（飞远）"]
                if high_rpms and min(high_rpms) < max_rpm:
                    max_rpm = min(high_rpms)
                    print(f"根据失控原因调整最大RPM至: {max_rpm}")
                else:
                    max_rpm = (min_rpm + max_rpm) // 2
            # 如果失控分析显示RPM普遍过低，向上调整范围
            elif "高度过低（坠毁）" in failure_analysis:
                low_rpms = failure_analysis["高度过低（坠毁）"]
                if low_rpms and max(low_rpms) > min_rpm:
                    min_rpm = max(low_rpms)
                    print(f"根据失控原因调整最小RPM至: {min_rpm}")
                else:
                    min_rpm = (min_rpm + max_rpm) // 2
            else:
                # 默认缩小范围
                range_size = max_rpm - min_rpm
                min_rpm += range_size // 4
                max_rpm -= range_size // 4
            continue
            
        best_rpm = max(scores.items(), key=lambda x: x[1])[0]
        print(f"\n本轮最佳RPM: {best_rpm}, 得分: {scores[best_rpm]:.2f}")
        
        if best_rpm == left_rpm:
            max_rpm = mid_rpm
        elif best_rpm == right_rpm:
            min_rpm = mid_rpm
        else:  # best_rpm == mid_rpm
            min_rpm = left_rpm
            max_rpm = right_rpm
        
        print(f"更新搜索范围: {min_rpm} - {max_rpm}")
    
    # 最终结果
    final_rpm = (min_rpm + max_rpm) // 2
    
    # 显示所有测试结果
    print("\n\n========== 搜索结果 ==========")
    print(f"最终确定的最佳悬停RPM值: {final_rpm}")
    
    # 如果有足够的测试数据，找出得分最高的RPM
    if all_results:
        best_rpm = max(all_results.items(), key=lambda x: x[1]['total_score'])[0]
        best_score = all_results[best_rpm]['total_score']
        
        print(f"全局最佳RPM值: {best_rpm}")
        print(f"最佳得分: {best_score:.2f}")
        print(f"平均高度: {all_results[best_rpm]['height_mean']:.3f} ± {all_results[best_rpm]['height_std']:.3f}")
        print(f"平均倾角: {all_results[best_rpm]['tilt_mean']:.3f} ± {all_results[best_rpm]['tilt_std']:.3f}")
        print(f"水平漂移: {all_results[best_rpm]['horizontal_drift']:.3f}")
        print(f"最大连续稳定步数: {all_results[best_rpm]['max_stable_steps']}")
        print(f"终止原因: {all_results[best_rpm]['failure_reason']}")
        
        # 输出失控分析
        if failure_analysis:
            print("\n失控原因分析:")
            for reason, rpms in failure_analysis.items():
                print(f"  {reason}: {len(rpms)}次, RPM范围[{min(rpms)}-{max(rpms)}]")
        
        # 建议使用全局最佳RPM而非二分搜索的最终结果
        return best_rpm
    
    return final_rpm

def continuous_rpm_testing(initial_rpm=5150, test_duration=200, device="cpu"):
    """
    持续测试不同的RPM值，在每次测试后等待用户输入下一个RPM值，直到用户手动中止。
    
    参数:
        initial_rpm: 初始RPM值
        test_duration: 每次测试的持续时间（步数）
        device: 运行设备
    """
    print(f"开始持续RPM测试，初始值: {initial_rpm} RPM")
    print("每次测试后将等待您输入下一次测试的RPM值")
    print("输入 'q' 或 'exit' 可随时退出测试")
    
    # 测试历史记录
    test_history = []
    current_rpm = initial_rpm
    test_count = 0
    
    # 创建一个环境实例，将在整个测试过程中复用
    print("创建模拟环境...")
    env = HoverEnv(device=device, headless=False, verbose=True)
    
    try:
        while True:  # 持续循环，直到用户中断
            test_count += 1
            print(f"\n\n===== 测试 #{test_count} =====")
            print(f"测试RPM值: {current_rpm}")
            
            # 更新当前环境的悬停RPM值
            env.hover_rpm = current_rpm
            
            # 重置环境（而不是重新创建）
            state, _ = env.reset()
            
            # 记录测试数据
            heights = []
            tilts = []
            tilt_rates = []  # 记录倾角变化率
            roll_angles = []  # 记录横滚角
            pitch_angles = []  # 记录俯仰角
            x_positions = []
            y_positions = []
            vz_values = []  # 记录垂直速度
            angular_velocities = []  # 记录角速度
            last_tilt = 0  # 上一步的倾角
            stable_steps = 0
            max_stable_steps = 0
            failure_reason = "完成全部测试步数"
            failure_details = {}  # 详细记录失控时的状态
            step_count = 0
            
            # 设置监控阈值
            tilt_rate_threshold = 0.05  # 倾角变化率阈值
            
            # 运行测试
            for step_idx in range(test_duration):
                step_count = step_idx
                # 使用恒定RPM值
                rpms = np.array([current_rpm, current_rpm, current_rpm, current_rpm])
                
                # 执行一步
                next_state, reward, done, info = env.step(rpms)
                
                # 提取状态，注意可能是扩展的12维状态
                if len(next_state) >= 12:  # 扩展的12维状态
                    x, y, z = next_state[0], next_state[1], next_state[2]  # 位置
                    vx, vy, vz = next_state[3], next_state[4], next_state[5]  # 速度
                    tilt = next_state[6]  # 总倾角
                    roll, pitch = next_state[7], next_state[8]  # 横滚角、俯仰角
                    wx, wy, wz = next_state[9], next_state[10], next_state[11]  # 角速度
                else:  # 旧版的7维状态
                    x, y, z = next_state[0], next_state[1], next_state[2]  # 位置
                    vx, vy, vz = next_state[3], next_state[4], next_state[5]  # 速度
                    tilt = next_state[6]  # 倾角
                    roll, pitch = 0, 0  # 无法获取，设为0
                    wx, wy, wz = 0, 0, 0  # 无法获取，设为0
                
                # 计算倾角变化率
                tilt_rate = tilt - last_tilt
                last_tilt = tilt
                
                # 记录数据
                heights.append(z)
                tilts.append(tilt)
                tilt_rates.append(tilt_rate)
                roll_angles.append(roll)
                pitch_angles.append(pitch)
                x_positions.append(x)
                y_positions.append(y)
                vz_values.append(vz)
                angular_velocities.append([wx, wy, wz])
                
                # 检查稳定性
                if abs(z - 1.0) < 0.15 and tilt < 0.15:
                    stable_steps += 1
                    max_stable_steps = max(max_stable_steps, stable_steps)
                else:
                    stable_steps = 0
                
                # 显示进度，增加更多状态信息
                if step_idx % 10 == 0:
                    angular_velocity_mag = np.sqrt(wx**2 + wy**2 + wz**2) if wx != 0 or wy != 0 or wz != 0 else 0
                    print(f"\r步数: {step_idx}/{test_duration}, 高度: {z:.2f}, 水平位置: ({x:.2f}, {y:.2f}), "
                          f"速度: vz={vz:.2f}, 倾角: {tilt:.3f}[r={roll:.2f},p={pitch:.2f}], "
                          f"角速度: {angular_velocity_mag:.2f}, 当前连续稳定步数: {stable_steps}", end="")
                
                # 如果已经倒下或飞太高，提前结束并分析原因
                if done:
                    # 记录详细失控状态
                    failure_details = {
                        'step': step_idx,
                        'position': [x, y, z],
                        'velocity': [vx, vy, vz],
                        'tilt': tilt,
                        'roll': roll,
                        'pitch': pitch,
                        'tilt_rate': tilt_rate,
                        'angular_velocity': [wx, wy, wz],
                        'rpms': rpms.tolist()
                    }
                    
                    if z < 0.3:
                        failure_reason = "高度过低（坠毁）"
                    elif z > 1.8:
                        failure_reason = "高度过高（飞远）"
                    elif tilt > 0.5:
                        # 区分不同的翻转情况
                        if abs(roll) > abs(pitch):
                            failure_reason = f"翻转 - 主要是横滚方向 (roll={roll:.2f}, pitch={pitch:.2f})"
                        else:
                            failure_reason = f"翻转 - 主要是俯仰方向 (roll={roll:.2f}, pitch={pitch:.2f})"
                            
                        # 检查翻转是否由于角速度过大引起
                        angular_velocity_mag = np.sqrt(wx**2 + wy**2 + wz**2)
                        if angular_velocity_mag > 1.0:
                            failure_reason += f", 角速度过大 ({angular_velocity_mag:.2f})"
                            
                        # 检查翻转是否由于倾角快速变化引起
                        recent_tilt_rates = tilt_rates[-min(10, len(tilt_rates)):]
                        max_tilt_rate = max(abs(rate) for rate in recent_tilt_rates)
                        if max_tilt_rate > tilt_rate_threshold:
                            failure_reason += f", 倾角变化率过大 ({max_tilt_rate:.3f})"
                    elif abs(vz) > 1.5:
                        failure_reason = "垂直速度过大"
                    elif np.sqrt(x**2 + y**2) > 1.5:
                        failure_reason = "水平偏移过大"
                    
                    print(f"\n无人机在{step_idx}步后失控，原因: {failure_reason}")
                    
                    # 打印更详细的失控分析
                    print("详细失控状态:")
                    print(f"  位置: ({x:.2f}, {y:.2f}, {z:.2f})")
                    print(f"  速度: ({vx:.2f}, {vy:.2f}, {vz:.2f})")
                    print(f"  倾角: {tilt:.3f} (横滚={roll:.3f}, 俯仰={pitch:.3f})")
                    print(f"  角速度: ({wx:.2f}, {wy:.2f}, {wz:.2f})")
                    print(f"  倾角变化率: {tilt_rate:.4f}")
                    
                    # 检查过去几步的状态变化
                    if len(tilts) > 10:
                        print("最近10步的状态变化:")
                        for i in range(max(0, len(tilts)-10), len(tilts)):
                            print(f"  步数 {i}: 高度={heights[i]:.2f}, 倾角={tilts[i]:.3f}, "
                                  f"变化率={tilt_rates[i]:.4f}")
                    
                    break
            
            # 计算统计数据
            if heights:
                height_mean = np.mean(heights)
                height_std = np.std(heights)
                tilt_mean = np.mean(tilts)
                tilt_std = np.std(tilts)
                tilt_rate_mean = np.mean([abs(rate) for rate in tilt_rates[1:]])  # 忽略第一个值
                tilt_rate_std = np.std([abs(rate) for rate in tilt_rates[1:]])
                vz_mean = np.mean(vz_values)
                vz_std = np.std(vz_values)
                
                # 计算水平位置漂移
                x_std = np.std(x_positions)
                y_std = np.std(y_positions)
                horizontal_drift = np.sqrt(x_std**2 + y_std**2)
                
                # 创建测试结果记录
                test_result = {
                    'rpm': current_rpm,
                    'step_count': step_count + 1,
                    'height_mean': height_mean,
                    'height_std': height_std,
                    'tilt_mean': tilt_mean,
                    'tilt_std': tilt_std,
                    'tilt_rate_mean': tilt_rate_mean,
                    'tilt_rate_std': tilt_rate_std,
                    'vz_mean': vz_mean,
                    'vz_std': vz_std,
                    'horizontal_drift': horizontal_drift,
                    'max_stable_steps': max_stable_steps,
                    'failure_reason': failure_reason,
                    'failure_details': failure_details
                }
                
                # 添加到历史记录
                test_history.append(test_result)
                
                # 计算测试得分
                stability_score = max_stable_steps
                height_error = abs(height_mean - 1.0)
                height_score = -height_error
                tilt_score = -tilt_mean
                jitter_score = -(height_std + tilt_std)
                drift_score = -horizontal_drift
                
                total_score = (
                    5.0 * stability_score +
                    3.0 * height_score +
                    2.0 * tilt_score +
                    1.0 * jitter_score +
                    1.0 * drift_score
                )
                
                # 打印测试结果，增加倾角变化率信息
                print(f"\n测试 #{test_count} 结果:")
                print(f"  RPM值: {current_rpm}")
                print(f"  测试持续步数: {step_count + 1}")
                print(f"  平均高度: {height_mean:.3f} ± {height_std:.3f}")
                print(f"  平均垂直速度: {vz_mean:.3f} ± {vz_std:.3f}")
                print(f"  平均倾角: {tilt_mean:.3f} ± {tilt_std:.3f}")
                print(f"  平均倾角变化率: {tilt_rate_mean:.4f} ± {tilt_rate_std:.4f}")
                print(f"  水平漂移: {horizontal_drift:.3f}")
                print(f"  最大连续稳定步数: {max_stable_steps}")
                print(f"  终止原因: {failure_reason}")
                print(f"  总得分: {total_score:.2f}")
                
                # 显示测试历史摘要
                print("\n测试历史摘要:")
                for i, hist in enumerate(test_history[-5:], 1):
                    if len(test_history) > 5:
                        idx = len(test_history) - 5 + i
                    else:
                        idx = i
                    print(f"  #{idx}: RPM={hist['rpm']}, 高度={hist['height_mean']:.3f}, "
                          f"稳定步数={hist['max_stable_steps']}, 原因={hist['failure_reason']}")
                
                # 识别表现最好的RPM值
                if len(test_history) > 1:
                    best_test = max(test_history, key=lambda x: x['max_stable_steps'])
                    print(f"\n目前表现最好的RPM值: {best_test['rpm']}")
                    print(f"  持续步数: {best_test['step_count']}")
                    print(f"  稳定悬停步数: {best_test['max_stable_steps']}")
                    print(f"  平均高度: {best_test['height_mean']:.3f}")
                
                # 提供RPM建议
                suggested_rpm = current_rpm
                if failure_reason.startswith("高度过低"):
                    suggested_rpm = current_rpm + 20
                    print(f"\n建议: 由于坠毁，可尝试增加RPM到 {suggested_rpm} (+20)")
                elif failure_reason.startswith("高度过高"):
                    suggested_rpm = current_rpm - 20
                    print(f"\n建议: 由于飞得太高，可尝试减少RPM到 {suggested_rpm} (-20)")
                elif failure_reason.startswith("垂直速度过大"):
                    if vz_mean > 0:  # 上升速度过大
                        suggested_rpm = current_rpm - 10
                        print(f"\n建议: 由于上升速度过大，可尝试减少RPM到 {suggested_rpm} (-10)")
                    else:  # 下降速度过大
                        suggested_rpm = current_rpm + 10
                        print(f"\n建议: 由于下降速度过大，可尝试增加RPM到 {suggested_rpm} (+10)")
                elif failure_reason.startswith("翻转"):
                    # 翻转情况下尝试减小RPM
                    suggested_rpm = current_rpm - 15
                    print(f"\n建议: 由于翻转，可尝试减小RPM到 {suggested_rpm} (-15)")
                    # 也可以考虑略微增大RPM值
                    if len(test_history) > 1 and test_history[-2].get('failure_reason', '').startswith('翻转'):
                        if test_history[-2]['rpm'] < current_rpm:
                            # 如果上次RPM更小也翻转了，建议尝试中间值
                            mid_rpm = (current_rpm + test_history[-2]['rpm']) // 2
                            print(f"  或者: 由于上次较小的RPM值({test_history[-2]['rpm']})也翻转了，可尝试中间值 {mid_rpm}")
                elif failure_reason == "完成全部测试步数":
                    # 基于平均高度进行微调
                    if height_mean > 1.05:
                        suggested_rpm = current_rpm - 5
                        print(f"\n建议: 由于高度偏高 ({height_mean:.3f})，可尝试轻微减少RPM到 {suggested_rpm} (-5)")
                    elif height_mean < 0.95:
                        suggested_rpm = current_rpm + 5
                        print(f"\n建议: 由于高度偏低 ({height_mean:.3f})，可尝试轻微增加RPM到 {suggested_rpm} (+5)")
                    else:
                        print(f"\n建议: 当前RPM值 {current_rpm} 表现良好，可继续使用或尝试微调")
                
                # 等待用户输入下一次测试的RPM值
                while True:
                    next_input = input("\n请输入下一次测试的RPM值 (或输入q退出): ").strip()
                    
                    # 检查是否退出
                    if next_input.lower() in ['q', 'exit', 'quit']:
                        raise KeyboardInterrupt
                    
                    # 检查是否使用建议值
                    if next_input.lower() in ['s', 'suggested']:
                        current_rpm = suggested_rpm
                        print(f"已采用建议值: {current_rpm}")
                        break
                    
                    # 检查是否使用相对调整
                    if next_input.startswith('+') or next_input.startswith('-'):
                        try:
                            adjustment = int(next_input)
                            current_rpm += adjustment
                            print(f"已调整RPM: {current_rpm - adjustment} {next_input} = {current_rpm}")
                            break
                        except ValueError:
                            print("无效输入，请输入一个有效的数字调整量")
                            continue
                    
                    # 尝试解析为直接的RPM值
                    try:
                        rpm_value = int(next_input)
                        if 4000 <= rpm_value <= 6000:  # 合理范围检查
                            current_rpm = rpm_value
                            print(f"下一次测试将使用RPM值: {current_rpm}")
                            break
                        else:
                            print("RPM值应在4000-6000之间，请重新输入")
                    except ValueError:
                        print("无效输入，请输入一个有效的RPM值")
            
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
        
        # 显示总结
        if test_history:
            print("\n===== 测试总结 =====")
            print(f"总共完成 {len(test_history)} 次测试")
            
            # 找出表现最好的RPM值
            best_rpm_by_stability = max(test_history, key=lambda x: x['max_stable_steps'])
            
            # 找出高度最接近目标的RPM值
            best_rpm_by_height = min(test_history, key=lambda x: abs(x['height_mean'] - 1.0))
            
            # 分析翻转情况
            flip_tests = [test for test in test_history if "翻转" in test.get('failure_reason', '')]
            if flip_tests:
                print(f"\n发生翻转的测试: {len(flip_tests)} 次")
                print("翻转RPM分布:")
                flip_rpms = [test['rpm'] for test in flip_tests]
                min_flip_rpm = min(flip_rpms)
                max_flip_rpm = max(flip_rpms)
                print(f"  RPM范围: {min_flip_rpm} - {max_flip_rpm}")
                
                # 查看是否有明显的翻转趋势
                if len(flip_tests) >= 3:
                    flip_tilt_rates = [test.get('tilt_rate_mean', 0) for test in flip_tests]
                    avg_flip_tilt_rate = sum(flip_tilt_rates) / len(flip_tilt_rates)
                    print(f"  翻转测试的平均倾角变化率: {avg_flip_tilt_rate:.4f}")
                    
                    # 检查翻转方向是否一致
                    roll_signs = [test.get('failure_details', {}).get('roll', 0) > 0 for test in flip_tests]
                    pitch_signs = [test.get('failure_details', {}).get('pitch', 0) > 0 for test in flip_tests]
                    
                    roll_consistency = max(sum(roll_signs), len(roll_signs) - sum(roll_signs)) / len(roll_signs)
                    pitch_consistency = max(sum(pitch_signs), len(pitch_signs) - sum(pitch_signs)) / len(pitch_signs)
                    
                    if roll_consistency > 0.7:
                        direction = "右侧" if sum(roll_signs) > len(roll_signs)/2 else "左侧"
                        print(f"  翻转方向趋势: 主要向{direction}翻转 (一致性: {roll_consistency:.2f})")
                    
                    if pitch_consistency > 0.7:
                        direction = "前方" if sum(pitch_signs) > len(pitch_signs)/2 else "后方"
                        print(f"  翻转方向趋势: 主要向{direction}翻转 (一致性: {pitch_consistency:.2f})")
            
            print(f"\n稳定性最好的RPM值: {best_rpm_by_stability['rpm']}")
            print(f"  最大连续稳定步数: {best_rpm_by_stability['max_stable_steps']}")
            print(f"  平均高度: {best_rpm_by_stability['height_mean']:.3f}")
            print(f"  测试持续步数: {best_rpm_by_stability['step_count']}")
            
            print(f"\n高度最接近目标的RPM值: {best_rpm_by_height['rpm']}")
            print(f"  高度误差: {abs(best_rpm_by_height['height_mean'] - 1.0):.3f}")
            print(f"  最大连续稳定步数: {best_rpm_by_height['max_stable_steps']}")
            
            # 找出综合表现最好的RPM值
            def calc_score(test):
                stability = test['max_stable_steps']
                height_error = abs(test['height_mean'] - 1.0)
                tilt = test['tilt_mean']
                jitter = test['height_std'] + test['tilt_std']
                drift = test['horizontal_drift']
                
                return (5.0 * stability - 
                        3.0 * height_error - 
                        2.0 * tilt - 
                        1.0 * jitter - 
                        1.0 * drift)
            
            best_overall = max(test_history, key=calc_score)
            
            print(f"\n综合表现最好的RPM值: {best_overall['rpm']}")
            print(f"  测试持续步数: {best_overall['step_count']}")
            print(f"  最大连续稳定步数: {best_overall['max_stable_steps']}")
            print(f"  平均高度: {best_overall['height_mean']:.3f} ± {best_overall['height_std']:.3f}")
            print(f"  平均倾角: {best_overall['tilt_mean']:.3f} ± {best_overall['tilt_std']:.3f}")
            print(f"  水平漂移: {best_overall['horizontal_drift']:.3f}")
            
            # 建议
            print("\n最终建议:")
            print(f"推荐使用RPM值: {best_overall['rpm']}")
            print(f"该值提供了最好的总体性能，在稳定性、高度控制和姿态保持之间取得了良好平衡。")
            
            print(f"\n使用以下命令开始训练:")
            print(f"python drone_TD3.py --initial-rpm {best_overall['rpm']}")
            
            # 保存测试结果到CSV文件
            try:
                import pandas as pd
                df = pd.DataFrame(test_history)
                # 移除可能不适合CSV格式的复杂数据结构
                if 'failure_details' in df.columns:
                    df = df.drop(columns=['failure_details'])
                filename = f"rpm_test_results_{time.strftime('%Y%m%d_%H%M%S')}.csv"
                df.to_csv(filename, index=False)
                print(f"\n测试结果已保存到: {filename}")
            except Exception as e:
                print(f"\n无法保存测试结果到CSV文件: {str(e)}")
    
    finally:
        # 确保环境被关闭
        try:
            env.close()
        except:
            print("警告: 环境关闭时出现异常")

def manual_rpm_testing(initial_rpm=5150, device="cpu"):
    """
    手动测试无人机RPM值，用户可以实时调整转速值并观察效果
    
    参数:
        initial_rpm: 初始RPM值
        device: 运行设备
    """
    print(f"开始手动RPM测试，初始值: {initial_rpm} RPM")
    print("您可以随时输入新的RPM值来调整四个旋翼的转速")
    print("输入 'q' 或 'exit' 退出测试")
    
    # 创建环境
    print("创建模拟环境...")
    env = HoverEnv(device=device, headless=False, verbose=False)
    
    try:
        # 当前RPM值
        current_rpm = initial_rpm
        rpm_increment = 10  # 增量调整步长
        
        # 重置环境
        env.hover_rpm = current_rpm
        state, _ = env.reset()
        
        # 记录数据
        test_time = 0
        max_stable_time = 0
        stable_time = 0
        
        # 打印操作说明
        print("\n操作说明:")
        print("  输入数字: 直接设置新的RPM值 (例如: 5150)")
        print("  输入 +/- 数字: 按增量调整RPM (例如: +10, -5)")
        print("  输入 'q' 或 'exit': 退出测试")
        print("  输入 'i': 将增量步长调整为更小的值 (当前: 10)")
        print("  输入 'I': 将增量步长调整为更大的值 (当前: 10)")
        print("  输入 'r': 重置无人机位置")
        
        import threading
        import msvcrt
        
        # 用户输入处理线程
        def input_thread():
            nonlocal current_rpm, rpm_increment
            
            while True:
                # 获取用户输入
                user_input = input("\n请输入RPM值或命令: ")
                
                # 检查是否退出
                if user_input.lower() in ['q', 'exit']:
                    break
                
                # 处理增量调整和其他命令
                if user_input.startswith('+'):
                    try:
                        increment = int(user_input[1:])
                        current_rpm += increment
                        print(f"RPM增加{increment}到 {current_rpm}")
                    except ValueError:
                        print("无效输入，请输入有效的增量数值")
                
                elif user_input.startswith('-'):
                    try:
                        decrement = int(user_input[1:])
                        current_rpm -= decrement
                        print(f"RPM减少{decrement}到 {current_rpm}")
                    except ValueError:
                        print("无效输入，请输入有效的减量数值")
                
                elif user_input == 'i':
                    rpm_increment = max(1, rpm_increment // 2)
                    print(f"增量步长减小为 {rpm_increment}")
                
                elif user_input == 'I':
                    rpm_increment = min(100, rpm_increment * 2)
                    print(f"增量步长增大为 {rpm_increment}")
                
                elif user_input == 'r':
                    # 重置无人机位置
                    env.reset()
                    print("无人机位置已重置")
                
                else:
                    # 直接设置RPM值
                    try:
                        new_rpm = int(user_input)
                        if 4000 <= new_rpm <= 6000:  # 合理范围检查
                            current_rpm = new_rpm
                            print(f"RPM设置为 {current_rpm}")
                        else:
                            print("RPM值超出合理范围(4000-6000)")
                    except ValueError:
                        print("无效输入，请输入有效的RPM值或命令")
                
                # 限制RPM在合理范围内
                current_rpm = max(4000, min(6000, current_rpm))
        
        # 启动输入线程
        input_thread = threading.Thread(target=input_thread)
        input_thread.daemon = True  # 设为守护线程，主线程结束时自动结束
        input_thread.start()
        
        # 主循环 - 模拟执行
        step_count = 0
        is_running = True
        
        while is_running:
            step_count += 1
            
            # 使用当前RPM值
            rpms = np.array([current_rpm, current_rpm, current_rpm, current_rpm])
            
            # 执行一步
            next_state, reward, done, info = env.step(rpms)
            
            # 提取状态
            if len(next_state) >= 12:  # 扩展的12维状态
                x, y, z = next_state[0], next_state[1], next_state[2]  # 位置
                vx, vy, vz = next_state[3], next_state[4], next_state[5]  # 速度
                tilt = next_state[6]  # 总倾角
                roll, pitch = next_state[7], next_state[8]  # 横滚角、俯仰角
                wx, wy, wz = next_state[9], next_state[10], next_state[11]  # 角速度
            else:  # 旧版的7维状态
                x, y, z = next_state[0], next_state[1], next_state[2]  # 位置
                vx, vy, vz = next_state[3], next_state[4], next_state[5]  # 速度
                tilt = next_state[6]  # 倾角
                roll, pitch = 0, 0  # 无法获取，设为0
                wx, wy, wz = 0, 0, 0  # 无法获取，设为0
            
            # 计算水平偏移
            horizontal_offset = np.sqrt(x**2 + y**2)
            
            # 检查稳定性
            if abs(z - 1.0) < 0.15 and tilt < 0.15 and horizontal_offset < 0.15:
                stable_time += 1
                max_stable_time = max(max_stable_time, stable_time)
            else:
                stable_time = 0
            
            # 每10步更新一次状态显示
            if step_count % 10 == 0:
                is_stable = "【稳定】" if stable_time > 0 else "【不稳定】"
                angular_velocity = np.sqrt(wx**2 + wy**2 + wz**2) if wx != 0 or wy != 0 or wz != 0 else 0
                
                # 使用转义序列清除当前行并更新状态
                print(f"\r\033[K当前RPM: {current_rpm} | 高度: {z:.2f}m | 水平偏移: {horizontal_offset:.2f}m | " 
                      f"倾角: {tilt:.3f}rad [r={roll:.2f},p={pitch:.2f}] | 角速度: {angular_velocity:.2f} | "
                      f"状态: {is_stable} | 最长稳定: {max_stable_time}步", end="")
            
            # 如果失控，重置环境
            if done:
                print(f"\n无人机失控，自动重置环境. 原因: ", end="")
                if z < 0.3:
                    print("高度过低（坠毁）")
                elif z > 1.8:
                    print("高度过高（飞远）")
                elif tilt > 0.5:
                    if abs(roll) > abs(pitch):
                        print(f"翻转 - 主要是横滚方向 (roll={roll:.2f})")
                    else:
                        print(f"翻转 - 主要是俯仰方向 (pitch={pitch:.2f})")
                elif abs(vz) > 1.5:
                    print("垂直速度过大")
                elif horizontal_offset > 1.5:
                    print("水平偏移过大")
                else:
                    print("未知原因")
                
                # 重置环境但保持当前RPM值
                env.hover_rpm = current_rpm
                env.reset()
                stable_time = 0
            
            # 检查是否需要退出
            if not input_thread.is_alive():
                is_running = False
    
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
    
    finally:
        # 确保环境被关闭
        try:
            env.close()
        except:
            print("警告: 环境关闭时出现异常")
        
        # 输出测试结果摘要
        print("\n==== 手动测试结果摘要 ====")
        print(f"最终RPM值: {current_rpm}")
        print(f"最长稳定持续时间: {max_stable_time}步")
        print("=========================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="寻找最佳无人机悬停RPM值")
    parser.add_argument("--method", type=str, choices=["linear", "binary", "continuous", "manual"], default="continuous",
                        help="搜索方法：linear=线性搜索，binary=二分搜索，continuous=持续测试，manual=手动测试")
    parser.add_argument("--min-rpm", type=int, default=5000, help="最小RPM值")
    parser.add_argument("--max-rpm", type=int, default=5300, help="最大RPM值")
    parser.add_argument("--step", type=int, default=10, help="线性搜索的步长")
    parser.add_argument("--precision", type=int, default=5, help="二分搜索的精度")
    parser.add_argument("--duration", type=int, default=200, help="每个RPM测试的持续步数")
    parser.add_argument("--initial-rpm", type=int, default=5150, help="持续测试或手动测试的初始RPM值")
    
    args = parser.parse_args()
    
    # 检测设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    if args.method == "linear":
        best_rpm = find_hover_rpm(
            start_rpm=args.min_rpm,
            end_rpm=args.max_rpm,
            step=args.step,
            test_duration=args.duration,
            device=device
        )
    elif args.method == "binary":
        best_rpm = binary_search_rpm(
            min_rpm=args.min_rpm,
            max_rpm=args.max_rpm,
            precision=args.precision,
            test_duration=args.duration,
            device=device
        )
    elif args.method == "manual":
        manual_rpm_testing(
            initial_rpm=args.initial_rpm,
            device=device
        )
        best_rpm = None  # 手动模式不返回单一值
    else:  # continuous
        continuous_rpm_testing(
            initial_rpm=args.initial_rpm,
            test_duration=args.duration,
            device=device
        )
        best_rpm = None  # continuous模式不返回单一值
    
    if best_rpm:
        print(f"\n推荐使用以下命令开始训练:")
        print(f"python drone_TD3.py --initial-rpm {best_rpm}")
        print(f"\n或者使用以下命令测试:")
        print(f"python drone_TD3.py --test --initial-rpm {best_rpm}")