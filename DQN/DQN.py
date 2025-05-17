"""
DQN完整实现

这个文件包含DQN算法的完整实现，结合了前面所有组件：
1. Q-Network: 用于近似Q函数的神经网络
2. 经验回放缓冲区: 存储和采样转换
3. DQN智能体: 执行与环境交互和网络更新
4. 训练循环: 管理整个训练过程

通过这个文件，你可以直接训练DQN算法而不需要单独导入各个组件。
"""

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
from datetime import datetime
import os
import time


class QNetwork(nn.Module):
    """
    Q-Network用于近似Q函数，输入状态，输出每个动作的预期价值。
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        """
        初始化Q-Network
        
        参数:
            state_dim (int): 状态空间的维度
            action_dim (int): 动作空间的维度（可能的动作数量）
            hidden_dim (int): 隐藏层的神经元数量
        """
        super(QNetwork, self).__init__()
        
        # 定义网络层
        self.fc1 = nn.Linear(state_dim, hidden_dim)  # 第一隐藏层
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # 第二隐藏层
        self.fc3 = nn.Linear(hidden_dim, action_dim)  # 输出层
        
        # 初始化权重
        self._initialize_weights()
    
    def forward(self, state):
        """
        前向传播，计算给定状态下每个动作的Q值
        
        参数:
            state (torch.Tensor): 状态输入，形状为(batch_size, state_dim)
            
        返回:
            torch.Tensor: 每个动作的Q值，形状为(batch_size, action_dim)
        """
        x = torch.relu(self.fc1(state))  # 第一层 + ReLU激活
        x = torch.relu(self.fc2(x))      # 第二层 + ReLU激活
        q_values = self.fc3(x)           # 输出层，无激活函数
        
        return q_values
    
    def _initialize_weights(self):
        """
        初始化网络权重，使用He初始化（适用于ReLU激活函数）
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)


class ReplayBuffer:
    """
    经验回放缓冲区，用于存储和采样(state, action, reward, next_state, done)元组。
    """
    
    def __init__(self, capacity):
        """
        初始化经验回放缓冲区
        
        参数:
            capacity (int): 缓冲区的最大容量
        """
        self.buffer = deque(maxlen=capacity)  # 使用双端队列，设置最大长度
    
    def add(self, state, action, reward, next_state, done):
        """
        添加一个经验样本到缓冲区
        
        参数:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 回合是否结束的标志
        """
        # 创建经验元组并添加到缓冲区
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """
        从缓冲区随机采样一批经验
        
        参数:
            batch_size (int): 批大小
            
        返回:
            tuple: 包含批量状态、动作、奖励、下一状态和完成标志的元组
        """
        # 确保缓冲区中有足够的样本
        if len(self.buffer) < batch_size:
            # 如果样本不足，只采样可用的样本
            batch = random.sample(self.buffer, len(self.buffer))
        else:
            # 随机采样指定数量的样本
            batch = random.sample(self.buffer, batch_size)
        
        # 将批样本分解为单独的组件
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 转换为NumPy数组或PyTorch张量，便于神经网络处理
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones, dtype=np.float32))
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """
        返回缓冲区中的经验数量
        """
        return len(self.buffer)


class DQNAgent:
    """
    DQN智能体实现，包含所有DQN算法的核心逻辑。
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=64, learning_rate=1e-3, 
                 gamma=0.99, buffer_size=10000, batch_size=64, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 target_update_freq=10):
        """
        初始化DQN智能体
        
        参数:
            state_dim (int): 状态空间维度
            action_dim (int): 动作空间维度
            hidden_dim (int): 隐藏层维度
            learning_rate (float): 学习率
            gamma (float): 折扣因子
            buffer_size (int): 经验回放缓冲区大小
            batch_size (int): 训练批大小
            epsilon_start (float): 初始探索率
            epsilon_end (float): 最小探索率
            epsilon_decay (float): 探索率衰减系数
            target_update_freq (int): 目标网络更新频率
        """
        # 环境参数
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 学习参数
        self.gamma = gamma  # 折扣因子
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        # 探索参数
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # 网络更新参数
        self.target_update_freq = target_update_freq
        self.update_counter = 0
        
        # 创建主Q网络和目标Q网络
        self.q_network = QNetwork(state_dim, action_dim, hidden_dim)
        self.target_network = QNetwork(state_dim, action_dim, hidden_dim)
        
        # 初始化目标网络权重与主网络相同
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 设置优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # 创建经验回放缓冲区
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # 设置损失函数
        self.criterion = nn.MSELoss()
        
        # 设置设备 (CPU/GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network.to(self.device)
        self.target_network.to(self.device)
        
        print(f"DQNAgent初始化完成，使用设备: {self.device}")
    
    def select_action(self, state, evaluate=False):
        """
        使用epsilon-greedy策略选择动作
        
        参数:
            state: 当前环境状态
            evaluate (bool): 是否处于评估模式（不探索）
            
        返回:
            int: 选择的动作
        """
        # 评估模式下始终选择最优动作
        if evaluate:
            epsilon = 0.0
        else:
            epsilon = self.epsilon
        
        # Epsilon-greedy策略
        if random.random() < epsilon:
            # 随机探索
            return random.randrange(self.action_dim)
        else:
            # 贪婪选择
            with torch.no_grad():  # 不计算梯度
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # 添加批次维度
                q_values = self.q_network(state)
                return q_values.max(1)[1].item()  # 选择Q值最高的动作
    
    def train(self):
        """
        从经验回放缓冲区中采样并训练Q网络
        
        返回:
            float: 训练损失
        """
        # 检查缓冲区是否有足够的样本
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # 从缓冲区采样
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # 将数据移到设备
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # 计算当前Q值: Q(s, a)
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 计算下一个状态的最大Q值: max_a' Q'(s', a')
        with torch.no_grad():  # 不计算梯度
            next_q_values = self.target_network(next_states).max(1)[0]
        
        # 计算目标Q值: r + γ * max_a' Q'(s', a') * (1 - done)
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # 计算损失: (Q(s, a) - target)^2
        loss = self.criterion(current_q_values, target_q_values)
        
        # 优化模型
        self.optimizer.zero_grad()  # 清除梯度
        loss.backward()  # 反向传播计算梯度
        # 梯度裁剪（可选，防止梯度爆炸）
        nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()  # 更新参数
        
        # 更新目标网络
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 衰减探索率
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def add_experience(self, state, action, reward, next_state, done):
        """
        将经验添加到回放缓冲区
        
        参数:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 回合是否结束
        """
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    def save(self, filepath):
        """
        保存模型参数
        
        参数:
            filepath (str): 保存文件路径
        """
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, filepath)
        print(f"模型已保存到 {filepath}")
    
    def load(self, filepath):
        """
        加载模型参数
        
        参数:
            filepath (str): 加载文件路径
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        print(f"模型已从 {filepath} 加载")


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


if __name__ == "__main__":
    """
    运行DQN算法训练
    """
    # 训练参数
    env_name = "CartPole-v1"  # Gym环境名称
    num_episodes = 500        # 训练回合数
    
    # 超参数
    hidden_dim = 128          # 隐藏层大小
    lr = 1e-3                 # 学习率
    gamma = 0.99              # 折扣因子
    buffer_size = 10000       # 缓冲区大小
    batch_size = 64           # 批大小
    
    # 探索参数
    epsilon_start = 1.0       # 初始探索率
    epsilon_end = 0.01        # 最小探索率
    epsilon_decay = 0.995     # 探索率衰减系数
    
    # 网络更新参数
    target_update_freq = 10   # 目标网络更新频率
    
    # 训练参数
    print_every = 10          # 打印频率
    eval_every = 50           # 评估频率
    save_dir = "./models/dqn" # 保存目录
    
    # 开始训练
    agent, history = train_dqn(
        env_name=env_name,
        num_episodes=num_episodes,
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
    
    print("DQN训练完成！")
    
    # 使用训练好的智能体进行演示
    env = gym.make(env_name)
    _ = evaluate_agent(agent, env, num_episodes=3, render=True)
    env.close()
