"""
DQN步骤2: 经验回放缓冲区

在这个文件中，我们将实现DQN的另一个核心组件：经验回放缓冲区
经验回放是DQN成功的关键因素之一，它存储智能体与环境交互的经验，并允许随机采样这些经验进行训练
"""

import numpy as np
import random
from collections import deque
import torch


class ReplayBuffer:
    """
    经验回放缓冲区，用于存储和采样(state, action, reward, next_state, done)元组
    
    使用经验回放的主要好处：
    1. 打破样本之间的相关性，使训练更稳定
    2. 提高样本利用率，每个经验可以被多次使用
    3. 允许批量更新，提高训练效率
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
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones, dtype=np.float32)
        
        # 转换为PyTorch张量
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """
        返回缓冲区中的经验数量
        
        返回:
            int: 缓冲区中的样本数
        """
        return len(self.buffer)

'''
如果要深入了解，下面是一些可以探索的问题：
  1. 缓冲区大小如何影响DQN的性能？
  2. 如何实现优先经验回放(Prioritized Experience Replay)？
  3. 为什么随机采样对训练稳定性很重要？
  4. 缓冲区中样本分布不均匀会带来什么问题？
'''

if __name__ == "__main__":
    """
    测试经验回放缓冲区
    """
    # 创建一个小型缓冲区
    buffer_capacity = 1000
    replay_buffer = ReplayBuffer(buffer_capacity)
    
    # 测试添加样本
    state_dim = 4
    for i in range(5):
        state = np.random.rand(state_dim)
        action = np.random.randint(0, 2)
        reward = np.random.rand()
        next_state = np.random.rand(state_dim)
        done = bool(np.random.randint(0, 2))
        
        replay_buffer.add(state, action, reward, next_state, done)
    
    print(f"缓冲区大小: {len(replay_buffer)}")
    
    # 测试采样
    batch_size = 3
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
    
    print(f"\n采样批次大小: {batch_size}")
    print(f"状态形状: {states.shape}")
    print(f"动作形状: {actions.shape}")
    print(f"奖励形状: {rewards.shape}")
    print(f"下一状态形状: {next_states.shape}")
    print(f"完成标志形状: {dones.shape}")
    
    # 验证样本类型
    assert isinstance(states, torch.Tensor), "状态应该是PyTorch张量！"
    assert isinstance(rewards, torch.Tensor), "奖励应该是PyTorch张量！"
    print("\n测试通过！")
