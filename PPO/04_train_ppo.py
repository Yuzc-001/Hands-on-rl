"""
PPO步骤4: 训练循环

在这个文件中，我们将实现PPO的训练循环，将智能体与环境交互起来
进行交互、收集轨迹、训练网络，并记录训练过程
"""04_train_ppo.py

import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import os
import time
from tqdm import tqdm

# 导入前面实现的PPO智能体
from ppo.03_ppo_agent import PPOAgent


def train_ppo(env_name, num_epochs=100, steps_per_epoch=2048, hidden_dim=64,
             lr=3e-4, gamma=0.99, lam=0.95, clip_ratio=0.2,
             target_kl=0.01, value_coef=0.5, entropy_coef=0.01,
             max_grad_norm=0.5, update_epochs=10, batch_size=64,
             eval_freq=10, save_freq=10, render_final=True, 
             save_dir='./models'):
    """
    训练PPO智能体
    
    参数:
        env_name (str): Gym环境名称
        num_epochs (int): 训练的总轮数
        steps_per_epoch (int): 每轮收集的环境步数
        hidden_dim (int): 神经网络隐藏层维度
        lr (float): 学习率
        gamma (float): 折扣因子
        lam (float): GAE lambda参数
        clip_ratio (float): PPO裁剪比率
        target_kl (float): 目标KL散度（用于早停）
        value_coef (float): 值函数损失系数
        entropy_coef (float): 熵正则化系数
        max_grad_norm (float): 梯度裁剪阈值
        update_epochs (int): 每批数据的更新轮数
        batch_size (int): 训练批大小
        eval_freq (int): 评估频率（轮数）
        save_freq (int): 保存频率（轮数）
        render_final (bool): 是否渲染最终策略
        save_dir (str): 模型保存目录
        
    返回:
        PPOAgent: 训练好的智能体
        dict: 包含训练历史的字典
    """
    # 创建环境
    env = gym.make(env_name)
    eval_env = gym.make(env_name)  # 用于评估的单独环境
    
    # 获取环境信息
    state_dim = env.observation_space.shape[0]  # 状态维度
    
    # 判断动作空间类型
    continuous = isinstance(env.action_space, gym.spaces.Box)
    if continuous:
        action_dim = env.action_space.shape[0]  # 连续动作维度
    else:
        action_dim = env.action_space.n  # 离散动作数量
    
    print(f"环境: {env_name}")
    print(f"状态维度: {state_dim}")
    print(f"动作维度/数量: {action_dim}")
    print(f"连续动作空间: {continuous}")
    
    # 创建PPO智能体
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        continuous=continuous,
        lr=lr,
        gamma=gamma,
        lam=lam,
        clip_ratio=clip_ratio,
        target_kl=target_kl,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        max_grad_norm=max_grad_norm,
        update_epochs=update_epochs,
        buffer_size=steps_per_epoch,
        batch_size=batch_size
    )
    
    # 创建目录保存模型
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 用于记录训练历史的字典
    history = {
        'epoch_returns': [],    # 每轮的平均回报
        'eval_returns': [],     # 评估回报
        'policy_losses': [],    # 策略损失
        'value_losses': [],     # 值函数损失
        'entropies': [],        # 熵
        'kls': [],              # KL散度
        'explained_vars': []    # 解释方差
    }
    
    # 开始训练
    start_time = time.time()
    total_steps = 0
    
    for epoch in range(1, num_epochs + 1):
        # 记录每轮数据
        epoch_returns = []
        episode_lengths = []
        
        # 收集轨迹
        state = env.reset()
        episode_return = 0
        episode_length = 0
        
        # 在一轮内收集固定步数的数据
        for t in range(steps_per_epoch):
            # 选择动作
            action, log_prob, value = agent.select_action(state)
            
            # 如果动作是标量（离散动作空间），转换为int
            if not continuous and isinstance(action, np.ndarray):
                action = action.item()
            
            # 执行动作
            next_state, reward, done, _ = env.step(action)
            
            # 存储转换
            agent.store_transition(state, action, reward, value, log_prob, done)
            
            # 更新状态和统计信息
            state = next_state
            episode_return += reward
            episode_length += 1
            total_steps += 1
            
            # 如果回合结束
            if done:
                # 记录回合数据
                epoch_returns.append(episode_return)
                episode_lengths.append(episode_length)
                
                # 完成轨迹（终止状态的值为0）
                agent.finish_path()
                
                # 重置环境
                state = env.reset()
                episode_return = 0
                episode_length = 0
            
            # 如果到达轮末尾但回合未结束
            elif t == steps_per_epoch - 1:
                # 获取未完成回合的最后状态的值估计
                _, _, last_value = agent.select_action(state)
                
                # 使用这个值完成轨迹
                agent.finish_path(last_value)
        
        # 计算平均回报
        avg_return = np.mean(epoch_returns) if epoch_returns else 0
        avg_length = np.mean(episode_lengths) if episode_lengths else 0
        
        # 记录轮数据
        history['epoch_returns'].append(avg_return)
        
        # 更新策略
        metrics = agent.update()
        
        # 记录更新指标
        history['policy_losses'].append(metrics['policy_loss'])
        history['value_losses'].append(metrics['value_loss'])
        history['entropies'].append(metrics['entropy'])
        history['kls'].append(metrics['kl'])
        history['explained_vars'].append(metrics['explained_variance'])
        
        # 打印训练信息
        print(f"\n轮 {epoch}/{num_epochs}")
        print(f"平均回报: {avg_return:.2f}")
        print(f"平均回合长度: {avg_length:.2f}")
        print(f"策略损失: {metrics['policy_loss']:.4f}")
        print(f"值函数损失: {metrics['value_loss']:.4f}")
        print(f"熵: {metrics['entropy']:.4f}")
        print(f"KL散度: {metrics['kl']:.4f}")
        print(f"解释方差: {metrics['explained_variance']:.4f}")
        
        # 定期评估智能体
        if epoch % eval_freq == 0:
            eval_return, eval_length = evaluate_agent(agent, eval_env)
            history['eval_returns'].append(eval_return)
            
            print(f"\n评估 - 轮 {epoch}, 平均回报: {eval_return:.2f}, 平均长度: {eval_length:.2f}")
        
        # 定期保存模型
        if epoch % save_freq == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(save_dir, f"{env_name}_ppo_{epoch}_{timestamp}.pth")
            agent.save(model_path)
    
    # 训练结束
    total_time = time.time() - start_time
    print(f"\n训练完成！总步数: {total_steps}, 总时间: {total_time:.2f}秒")
    
    # 绘制训练曲线
    plot_training_results(history, env_name)
    
    # 如果要求，渲染最终策略
    if render_final:
        print("\n渲染最终策略...")
        _ = evaluate_agent(agent, env, num_episodes=3, render=True)
    
    # 关闭环境
    env.close()
    eval_env.close()
    
    return agent, history


def evaluate_agent(agent, env, num_episodes=10, render=False):
    """
    评估智能体在环境中的表现
    
    参数:
        agent (PPOAgent): 要评估的智能体
        env (gym.Env): 评估环境
        num_episodes (int): 评估的回合数
        render (bool): 是否渲染环境
        
    返回:
        float: 平均回报
        float: 平均回合长度
    """
    returns = []
    lengths = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_return = 0
        episode_length = 0
        done = False
        
        while not done:
            if render:
                env.render()
                time.sleep(0.01)  # 渲染延迟
            
            # 选择动作
            action, _, _ = agent.select_action(state)
            
            # 如果动作是标量（离散动作空间），转换为int
            if isinstance(action, np.ndarray) and action.size == 1:
                action = action.item()
            
            # 执行动作
            next_state, reward, done, _ = env.step(action)
            
            # 更新状态和统计信息
            state = next_state
            episode_return += reward
            episode_length += 1
        
        returns.append(episode_return)
        lengths.append(episode_length)
    
    # 计算平均值
    avg_return = np.mean(returns)
    avg_length = np.mean(lengths)
    
    return avg_return, avg_length


def plot_training_results(history, env_name):
    """
    绘制训练过程中的各种指标
    
    参数:
        history (dict): 包含训练历史的字典
        env_name (str): 环境名称
    """
    # 创建图形
    plt.figure(figsize=(15, 15))
    
    # 1. 绘制轮回报
    plt.subplot(3, 2, 1)
    plt.plot(history['epoch_returns'])
    plt.xlabel('轮')
    plt.ylabel('平均回报')
    plt.title('轮平均回报')
    plt.grid(True)
    
    # 2. 绘制评估回报
    if history['eval_returns']:
        plt.subplot(3, 2, 2)
        eval_epochs = [(i+1) * (len(history['epoch_returns']) // len(history['eval_returns'])) 
                      for i in range(len(history['eval_returns']))]
        plt.plot(eval_epochs, history['eval_returns'], marker='o')
        plt.xlabel('轮')
        plt.ylabel('评估回报')
        plt.title('评估回报')
        plt.grid(True)
    
    # 3. 绘制策略损失和值函数损失
    plt.subplot(3, 2, 3)
    plt.plot(history['policy_losses'], label='策略损失')
    plt.plot(history['value_losses'], label='值函数损失')
    plt.xlabel('轮')
    plt.ylabel('损失')
    plt.title('损失')
    plt.legend()
    plt.grid(True)
    
    # 4. 绘制熵
    plt.subplot(3, 2, 4)
    plt.plot(history['entropies'])
    plt.xlabel('轮')
    plt.ylabel('熵')
    plt.title('策略熵')
    plt.grid(True)
    
    # 5. 绘制KL散度
    plt.subplot(3, 2, 5)
    plt.plot(history['kls'])
    plt.xlabel('轮')
    plt.ylabel('KL散度')
    plt.title('KL散度')
    plt.grid(True)
    
    # 6. 绘制解释方差
    plt.subplot(3, 2, 6)
    plt.plot(history['explained_vars'])
    plt.xlabel('轮')
    plt.ylabel('解释方差')
    plt.title('值函数解释方差')
    plt.grid(True)
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(f"{env_name}_ppo_training.png")
    plt.show()

'''
如果要深入了解，下面是一些可以探索的问题：

1. 为什么PPO使用"轮"而不是DQN的"回合"作为收集数据的单位？
2. 如何选择steps_per_epoch和batch_size的合适值？
3. 如何调整PPO的超参数以适应不同的环境？
4. 如何处理连续动作空间和离散动作空间的区别？
'''

if __name__ == "__main__":
    """
    PPO训练示例
    """
    # 训练PPO智能体在CartPole环境
    env_name = "CartPole-v1"
    agent, history = train_ppo(
        env_name=env_name,
        num_epochs=50,          # 训练轮数
        steps_per_epoch=2048,   # 每轮步数
        hidden_dim=64,          # 隐藏层大小
        lr=3e-4,                # 学习率
        gamma=0.99,             # 折扣因子
        lam=0.95,               # GAE lambda参数
        clip_ratio=0.2,         # PPO裁剪比率
        target_kl=0.01,         # 目标KL散度
        value_coef=0.5,         # 值函数损失系数
        entropy_coef=0.01,      # 熵正则化系数
        update_epochs=10,       # 每批数据的更新轮数
        batch_size=64,          # 训练批大小
        eval_freq=5,            # 评估频率
        save_freq=10,           # 保存频率
        render_final=True,      # 渲染最终策略
        save_dir='./models/ppo' # 保存目录
    )
    
    print("PPO训练完成！")
