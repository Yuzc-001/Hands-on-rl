"""
PPO步骤2: PPO缓冲区与GAE计算

在这个文件中，我们将实现PPO的两个重要组件：
1. 轨迹缓冲区：收集并存储与环境交互的经验
2. 广义优势估计(GAE)：计算更好的优势函数估计
"""

import numpy as np
import torch


class PPOBuffer:
    """
    PPO缓冲区用于存储轨迹并计算广义优势估计(GAE)。
    
    与DQN的经验回放不同，PPO是一种在线策略算法，它只使用当前策略生成的数据，
    因此我们存储完整的轨迹而不是随机采样的转换。
    """
    
    def __init__(self, state_dim, action_dim, buffer_size, gamma=0.99, lam=0.95):
        """
        初始化PPO缓冲区
        
        参数:
            state_dim (int): 状态空间维度
            action_dim (int): 动作空间维度
            buffer_size (int): 缓冲区大小（最大时间步数）
            gamma (float): 折扣因子
            lam (float): GAE lambda参数
        """
        # 初始化存储缓冲区
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        
        # 如果动作是标量，直接存储；如果是向量，存储为向量
        if action_dim == 1:
            self.actions = np.zeros(buffer_size, dtype=np.float32)
        else:
            self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
            
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.logprobs = np.zeros(buffer_size, dtype=np.float32)
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        
        # 超参数
        self.gamma = gamma
        self.lam = lam
        
        # 缓冲区状态
        self.ptr = 0  # 指向下一个要存储的位置
        self.path_start_idx = 0  # 当前轨迹的起始索引
        self.max_size = buffer_size
    
    def store(self, state, action, reward, value, logprob, done):
        """
        存储一个时间步的转换
        
        参数:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            value: 状态值估计
            logprob: 动作的对数概率
            done: 是否是终止状态
        """
        # 确保缓冲区未满
        assert self.ptr < self.max_size
        
        # 存储转换
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.logprobs[self.ptr] = logprob
        self.dones[self.ptr] = done
        
        # 更新指针
        self.ptr += 1
    
    def finish_path(self, last_value=0):
        """
        完成一条轨迹，计算优势估计和回报
        
        参数:
            last_value (float): 如果轨迹被截断，提供最后状态的值估计
        """
        # 当前轨迹的片段
        path_slice = slice(self.path_start_idx, self.ptr)
        
        # 获取轨迹中的奖励和值
        rewards = np.append(self.rewards[path_slice], last_value)
        values = np.append(self.values[path_slice], last_value)
        dones = np.append(self.dones[path_slice], 0)  # 假设最后一步不是终止状态
        
        # 计算GAE
        deltas = rewards[:-1] + self.gamma * values[1:] * (1 - dones[:-1]) - values[:-1]
        self.advantages[path_slice] = self._discount_cumsum(deltas, self.gamma * self.lam, dones[:-1])
        
        # 计算回报（值函数目标）
        self.returns[path_slice] = self._discount_cumsum(rewards[:-1], self.gamma, dones[:-1])
        
        # 更新轨迹起始索引
        self.path_start_idx = self.ptr
    
    def _discount_cumsum(self, x, discount, dones):
        """
        计算折扣累积和
        
        参数:
            x: 要累积的值（奖励或TD误差）
            discount: 折扣因子（gamma或gamma*lambda）
            dones: 终止标志
            
        返回:
            折扣累积和
        """
        n = len(x)
        y = np.zeros_like(x)
        y[-1] = x[-1]
        for i in reversed(range(n-1)):
            y[i] = x[i] + discount * (1 - dones[i]) * y[i+1]
        return y
    
    def get(self):
        """
        获取所有存储的数据
        
        返回:
            dict: 包含所有需要的数据的字典
        """
        # 断言缓冲区已满或已完成所有轨迹
        assert self.ptr == self.max_size or self.path_start_idx == self.ptr
        
        # 记录当前位置
        self.path_start_idx = 0
        self.ptr = 0
        
        # 计算优势标准化
        adv_mean = np.mean(self.advantages)
        adv_std = np.std(self.advantages) + 1e-8  # 添加小常数防止除零
        self.advantages = (self.advantages - adv_mean) / adv_std
        
        # 转换为PyTorch张量
        data = {
            "states": torch.FloatTensor(self.states),
            "actions": torch.FloatTensor(self.actions),
            "returns": torch.FloatTensor(self.returns),
            "advantages": torch.FloatTensor(self.advantages),
            "logprobs": torch.FloatTensor(self.logprobs),
            "values": torch.FloatTensor(self.values)
        }
        
        return data

'''
如果要深入了解，下面是一些可以探索的问题：

1. 为什么GAE对强化学习算法很重要？
2. lambda参数如何影响偏差-方差权衡？
3. 为什么要对优势进行标准化？
4. 在线策略算法与离线策略算法的缓冲区有何不同？
'''

if __name__ == "__main__":
    """
    测试PPO缓冲区和GAE计算
    """
    # 创建一个简单的PPO缓冲区
    state_dim = 4
    action_dim = 2
    buffer_size = 10
    gamma = 0.99
    lam = 0.95
    
    buffer = PPOBuffer(state_dim, action_dim, buffer_size, gamma, lam)
    
    # 使用一些随机数据填充缓冲区
    for i in range(5):  # 一条轨迹有5步
        state = np.random.rand(state_dim)
        action = np.random.rand(action_dim)
        reward = np.random.rand()
        value = np.random.rand()
        logprob = np.random.rand()
        done = False if i < 4 else True  # 最后一步是终止状态
        
        buffer.store(state, action, reward, value, logprob, done)
    
    # 完成轨迹
    buffer.finish_path(last_value=0.0)
    
    # 添加另一条轨迹
    for i in range(5):  # 另一条轨迹有5步
        state = np.random.rand(state_dim)
        action = np.random.rand(action_dim)
        reward = np.random.rand()
        value = np.random.rand()
        logprob = np.random.rand()
        done = False if i < 4 else True
        
        buffer.store(state, action, reward, value, logprob, done)
    
    # 完成第二条轨迹
    buffer.finish_path(last_value=0.0)
    
    # 获取所有数据
    data = buffer.get()
    
    # 打印数据形状
    for key, value in data.items():
        print(f"{key} 形状: {value.shape}")
    
    # 检查优势是否已标准化
    adv_mean = data["advantages"].mean().item()
    adv_std = data["advantages"].std().item()
    print(f"\n优势均值: {adv_mean:.6f} (应接近0)")
    print(f"优势标准差: {adv_std:.6f} (应接近1)")
    
    # 验证计算
    print("\n验证GAE计算:")
    print(f"优势: {data['advantages']}")
    print(f"回报: {data['returns']}")
    
    print("\n测试通过！")
