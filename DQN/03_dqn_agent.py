"""
DQN步骤3: DQN智能体

在这个文件中，我们将实现完整的DQN智能体，整合Q-Network和经验回放缓冲区
DQN智能体负责与环境交互、收集经验，并更新神经网络以学习最优策略
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

# 导入前面实现的组件
from dqn.01_q_network import QNetwork
from dqn.02_replay_buffer import ReplayBuffer


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
        # actions是一个列向量，我们需要计算Q网络对于这些(state, action)对的输出
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # 计算下一个状态的最大Q值: max_a' Q'(s', a')
        # 我们使用目标网络来计算这个值，这样可以提高训练稳定性
        with torch.no_grad():  # 不计算梯度
            next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
        
        # 计算目标Q值: r + γ * max_a' Q'(s', a') * (1 - done)
        # 如果done=True（回合结束），则只考虑即时奖励
        target_q_values = rewards.unsqueeze(1) + self.gamma * next_q_values * (1 - dones.unsqueeze(1))
        
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
            print(f"目标网络已更新，步数: {self.update_counter}")
        
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


'''
如果要深入了解，下面是一些可以探索的问题：
  1. 如何实现双重DQN(Double DQN)来减少Q值过估计？
  2. Epsilon衰减策略如何影响探索-利用权衡？
  3. 目标网络更新频率如何影响训练稳定性？
  4. 如何添加优先经验回放来更有效地学习？
'''

if __name__ == "__main__":
    """
    测试DQN智能体
    """
    # 创建一个简单环境的DQN智能体
    state_dim = 4
    action_dim = 2
    agent = DQNAgent(state_dim, action_dim)
    
    # 测试动作选择
    test_state = np.random.rand(state_dim)
    action = agent.select_action(test_state)
    print(f"对于测试状态，选择的动作是: {action}")
    
    # 添加一些随机经验到缓冲区
    for i in range(100):
        state = np.random.rand(state_dim)
        action = np.random.randint(0, action_dim)
        reward = np.random.rand()
        next_state = np.random.rand(state_dim)
        done = bool(np.random.randint(0, 2))
        
        agent.add_experience(state, action, reward, next_state, done)
    
    # 测试训练
    loss = agent.train()
    print(f"训练损失: {loss}")
    
    # 测试模型保存和加载
    agent.save("test_dqn_model.pth")
    agent.load("test_dqn_model.pth")
    
    print("DQNAgent测试完成！")
