"""
DQN步骤4: 训练循环

在这个文件中，我们将实现DQN的训练循环，将智能体与环境交互起来
进行交互、收集经验、训练网络，并记录训练过程
"""

import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import os
from tqdm import tqdm
import time

# 导入前面实现的DQN智能体
from dqn.03_dqn_agent import DQNAgent


def train_dqn(env_name, num_episodes=1000, hidden_dim=64, lr=1e-3, 
              gamma=0.99, buffer_size=10000, batch_size=64, 
              epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
              target_update_freq=10, print_every=20, eval_every=100, 
              save_dir='./models'):
    """
    训练DQN智能体
    
    参数:
        env_name (str): Gym环境名称
        num_episodes (int): 训练的总回合数
        hidden_dim (int): 神经网络隐藏层维度
        lr (float): 学习率
        gamma (float): 折扣因子
        buffer_size (int): 经验回放缓冲区大小
        batch_size (int): 训练批大小
        epsilon_start (float): 初始探索率
        epsilon_end (float): 最小探索率
        epsilon_decay (float): 探索率衰减系数
        target_update_freq (int): 目标网络更新频率
        print_every (int): 打印频率（回合数）
        eval_every (int): 评估频率（回合数）
        save_dir (str): 模型保存目录
        
    返回:
        DQNAgent: 训练好的智能体
        dict: 包含训练历史的字典
    """
    # 创建环境
    env = gym.make(env_name)
    eval_env = gym.make(env_name)  # 用于评估的单独环境
    
    # 获取环境信息
    state_dim = env.observation_space.shape[0]  # 状态维度
    if isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = env.action_space.n  # 离散动作数量
    else:
        raise ValueError("仅支持离散动作空间")
    
    print(f"环境: {env_name}")
    print(f"状态维度: {state_dim}")
    print(f"动作数量: {action_dim}")
    
    # 创建DQN智能体
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        learning_rate=lr,
        gamma=gamma,
        buffer_size=buffer_size,
        batch_size=batch_size,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        target_update_freq=target_update_freq
    )
    
    # 创建目录保存模型
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 用于记录训练历史的字典
    history = {
        'episode_rewards': [],  # 每回合的总奖励
        'episode_lengths': [],  # 每回合的步数
        'eval_rewards': [],     # 评估奖励
        'eval_lengths': [],     # 评估步数
        'losses': [],           # 训练损失
        'epsilons': []          # 探索率变化
    }
    
    # 开始训练
    total_steps = 0
    start_time = time.time()
    
    for episode in range(1, num_episodes + 1):
        state = env.reset()  # 重置环境，获取初始状态
        episode_reward = 0
        episode_loss = 0
        episode_steps = 0
        done = False
        
        # 单个回合的交互循环
        while not done:
            # 智能体选择动作
            action = agent.select_action(state)
            
            # 执行动作，获取下一状态、奖励等信息
            next_state, reward, done, _ = env.step(action)
            
            # 存储经验
            agent.add_experience(state, action, reward, next_state, done)
            
            # 训练智能体
            loss = agent.train()
            if loss is not None:
                episode_loss += loss
            
            # 更新状态
            state = next_state
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
        
        # 记录回合信息
        history['episode_rewards'].append(episode_reward)
        history['episode_lengths'].append(episode_steps)
        history['epsilons'].append(agent.epsilon)
        
        # 计算平均损失
        avg_loss = episode_loss / episode_steps if episode_steps > 0 else 0
        history['losses'].append(avg_loss)
        
        # 定期打印训练信息
        if episode % print_every == 0:
            avg_reward = np.mean(history['episode_rewards'][-print_every:])
            avg_length = np.mean(history['episode_lengths'][-print_every:])
            print(f"回合: {episode}/{num_episodes}, 奖励: {episode_reward:.2f}, "
                  f"步数: {episode_steps}, 平均奖励: {avg_reward:.2f}, "
                  f"ε: {agent.epsilon:.4f}, 平均损失: {avg_loss:.4f}")
        
        # 定期评估智能体性能
        if episode % eval_every == 0:
            eval_reward, eval_length = evaluate_agent(agent, eval_env)
            history['eval_rewards'].append(eval_reward)
            history['eval_lengths'].append(eval_length)
            print(f"\n评估 - 回合: {episode}, 奖励: {eval_reward:.2f}, 步数: {eval_length}")
            
            # 保存模型
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(save_dir, f"{env_name}_dqn_{episode}_{timestamp}.pth")
            agent.save(model_path)
    
    # 训练结束
    env.close()
    eval_env.close()
    
    # 计算总训练时间
    total_time = time.time() - start_time
    print(f"\n训练完成！总步数: {total_steps}, 总时间: {total_time:.2f}秒")
    
    # 绘制训练曲线
    plot_training_results(history, env_name)
    
    return agent, history


def evaluate_agent(agent, env, num_episodes=5, render=False):
    """
    评估智能体在环境中的表现
    
    参数:
        agent (DQNAgent): 要评估的智能体
        env (gym.Env): 评估环境
        num_episodes (int): 评估的回合数
        render (bool): 是否渲染环境
        
    返回:
        float: 平均奖励
        float: 平均回合长度
    """
    total_reward = 0
    total_length = 0
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            if render:
                env.render()
                time.sleep(0.01)  # 渲染延迟
            
            # 评估模式下选择最优动作（无探索）
            action = agent.select_action(state, evaluate=True)
            
            # 执行动作
            next_state, reward, done, _ = env.step(action)
            
            # 更新状态和统计信息
            state = next_state
            episode_reward += reward
            episode_length += 1
        
        total_reward += episode_reward
        total_length += episode_length
    
    # 计算平均值
    avg_reward = total_reward / num_episodes
    avg_length = total_length / num_episodes
    
    return avg_reward, avg_length


def plot_training_results(history, env_name):
    """
    绘制训练过程中的各种指标
    
    参数:
        history (dict): 包含训练历史的字典
        env_name (str): 环境名称
    """
    # 创建图形
    plt.figure(figsize=(15, 10))
    
    # 1. 绘制回合奖励
    plt.subplot(2, 2, 1)
    plt.plot(history['episode_rewards'])
    plt.xlabel('回合')
    plt.ylabel('奖励')
    plt.title('回合奖励')
    plt.grid(True)
    
    # 2. 绘制评估奖励
    if history['eval_rewards']:
        plt.subplot(2, 2, 2)
        eval_episodes = [(i+1) * (len(history['episode_rewards']) // len(history['eval_rewards'])) 
                          for i in range(len(history['eval_rewards']))]
        plt.plot(eval_episodes, history['eval_rewards'], marker='o')
        plt.xlabel('回合')
        plt.ylabel('评估奖励')
        plt.title('评估奖励')
        plt.grid(True)
    
    # 3. 绘制探索率
    plt.subplot(2, 2, 3)
    plt.plot(history['epsilons'])
    plt.xlabel('回合')
    plt.ylabel('探索率 (ε)')
    plt.title('探索率衰减')
    plt.grid(True)
    
    # 4. 绘制损失
    plt.subplot(2, 2, 4)
    plt.plot(history['losses'])
    plt.xlabel('回合')
    plt.ylabel('损失')
    plt.title('训练损失')
    plt.grid(True)
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(f"{env_name}_dqn_training.png")
    plt.show()

'''
如果要深入了解，下面是一些可以探索的问题：
  
  1. 如何处理不同类型的奖励（稀疏奖励、延迟奖励等）？
  2. 如何调整超参数以改善训练过程？
  3. 如何处理部分可观察环境？
  4. 如何扩展到高维观察空间（如图像）？
'''

if __name__ == "__main__":
    """
    DQN训练示例
    """
    # 训练DQN智能体在CartPole环境
    env_name = "CartPole-v1"
    agent, history = train_dqn(
        env_name=env_name,
        num_episodes=500,  # 训练回合数
        hidden_dim=128,    # 隐藏层大小
        lr=1e-3,           # 学习率
        gamma=0.99,        # 折扣因子
        buffer_size=10000, # 缓冲区大小
        batch_size=64,     # 批大小
        epsilon_start=1.0, # 初始探索率
        epsilon_end=0.01,  # 最小探索率
        epsilon_decay=0.995, # 探索率衰减
        target_update_freq=10, # 目标网络更新频率
        print_every=10,    # 打印频率
        eval_every=50,     # 评估频率
        save_dir='./models/dqn' # 保存目录
    )
    
    print("DQN训练完成！")
