"""
CartPole环境的DQN实现示例

这个文件展示了如何使用DQN算法在CartPole-v1环境中训练智能体。
CartPole是一个简单的控制问题，目标是通过左右移动小车来平衡一个倒立的杆子。
"""

import sys
import os
import torch
import gym
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 添加父目录到系统路径，以便导入DQN实现
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from DQN.DQN import DQNAgent, train_dqn, evaluate_agent


def run_cartpole_dqn():
    """
    在CartPole-v1环境中训练和评估DQN智能体
    """
    print("=" * 50)
    print("开始在CartPole-v1环境中训练DQN智能体")
    print("=" * 50)
    
    # 环境参数
    env_name = "CartPole-v1"
    
    # DQN超参数
    hidden_dim = 128       # 隐藏层大小
    lr = 1e-3              # 学习率
    gamma = 0.99           # 折扣因子
    buffer_size = 10000    # 经验回放缓冲区大小
    batch_size = 64        # 训练批大小
    epsilon_start = 1.0    # 初始探索率
    epsilon_end = 0.01     # 最小探索率
    epsilon_decay = 0.995  # 探索率衰减系数
    target_update_freq = 10 # 目标网络更新频率
    
    # 训练参数
    num_episodes = 500     # 训练回合数
    max_steps = 500        # 每回合最大步数
    print_every = 10       # 打印频率
    eval_every = 50        # 评估频率
    
    # 创建保存目录
    save_dir = "./models/dqn"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 训练DQN智能体
    agent, history = train_dqn(
        env_name=env_name,
        num_episodes=num_episodes,
        max_steps=max_steps,
        hidden_dim=hidden_dim,
        lr=lr,
        gamma=gamma,
        buffer_size=buffer_size,
        batch_size=batch_size,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        target_update_freq=target_update_freq,
        print_every=print_every,
        eval_every=eval_every,
        save_dir=save_dir
    )
    
    print("\n训练完成！")
    
    # 对训练好的智能体进行最终评估
    env = gym.make(env_name)
    print("\n最终评估:")
    avg_reward, avg_steps = evaluate_agent(agent, env, num_episodes=10, render=True)
    print(f"平均回报: {avg_reward:.2f}, 平均步数: {avg_steps:.2f}")
    
    # 关闭环境
    env.close()
    
    # 绘制学习曲线
    plt.figure(figsize=(12, 8))
    
    # 回报曲线
    plt.subplot(2, 2, 1)
    plt.plot(history['episode_rewards'], alpha=0.5, label='回合回报')
    plt.plot(np.convolve(history['episode_rewards'], np.ones(10)/10, mode='valid'), label='平滑回报')
    plt.xlabel('回合')
    plt.ylabel('回报')
    plt.title('训练回报')
    plt.legend()
    plt.grid(True)
    
    # 评估回报曲线
    plt.subplot(2, 2, 2)
    eval_episodes = [(i+1) * eval_every for i in range(len(history['eval_rewards']))]
    plt.plot(eval_episodes, history['eval_rewards'], 'o-', label='评估回报')
    plt.xlabel('回合')
    plt.ylabel('评估回报')
    plt.title('评估性能')
    plt.legend()
    plt.grid(True)
    
    # 探索率曲线
    plt.subplot(2, 2, 3)
    plt.plot(history['epsilons'])
    plt.xlabel('回合')
    plt.ylabel('探索率 (ε)')
    plt.title('探索率衰减')
    plt.grid(True)
    
    # 损失曲线
    plt.subplot(2, 2, 4)
    plt.plot(history['losses'])
    plt.xlabel('回合')
    plt.ylabel('损失')
    plt.title('训练损失')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{env_name}_dqn_results.png")
    plt.show()
    
    return agent, history


if __name__ == "__main__":
    agent, history = run_cartpole_dqn()
    print("CartPole DQN示例完成！")
