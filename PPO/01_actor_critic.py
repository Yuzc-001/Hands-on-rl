"""
PPO步骤1: Actor-Critic网络

在这个文件中，我们将实现PPO的核心组件之一：Actor-Critic网络
Actor-Critic架构结合了策略梯度（Actor）和值函数估计（Critic）的优点
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


class ActorCritic(nn.Module):
    """
    Actor-Critic网络实现，包含一个共享特征提取器，以及策略（Actor）和值函数（Critic）两个头。
    支持离散和连续动作空间。
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=64, continuous=False):
        """
        初始化Actor-Critic网络
        
        参数:
            state_dim (int): 状态空间的维度
            action_dim (int): 动作空间的维度（离散动作空间中是动作数量，连续动作空间中是动作向量维度）
            hidden_dim (int): 隐藏层的神经元数量
            continuous (bool): 是否是连续动作空间
        """
        super(ActorCritic, self).__init__()
        
        # 记录动作空间类型
        self.continuous = continuous
        
        # 共享特征提取层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Actor网络（策略）
        if continuous:
            # 连续动作空间 - 输出动作的均值和标准差
            self.actor_mean = nn.Linear(hidden_dim, action_dim)
            # 对于标准差，使用参数而不是网络层，并初始化为0（对应于初始标准差为1）
            self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            # 离散动作空间 - 输出每个动作的概率
            self.actor = nn.Sequential(
                nn.Linear(hidden_dim, action_dim),
                nn.Softmax(dim=-1)
            )
        
        # Critic网络（值函数）
        self.critic = nn.Linear(hidden_dim, 1)
        
        # 初始化权重
        self._initialize_weights()
    
    def forward(self, state):
        """
        前向传播，计算动作分布和状态价值
        
        参数:
            state (torch.Tensor): 状态输入，形状为(batch_size, state_dim)
            
        返回:
            distribution: 动作概率分布（离散或连续）
            value: 状态价值估计
        """
        # 共享特征提取
        features = self.shared(state)
        
        # 根据动作空间类型创建不同的分布
        if self.continuous:
            # 连续动作空间：输出正态分布
            action_mean = self.actor_mean(features)
            action_std = torch.exp(self.actor_log_std)  # 确保标准差为正
            distribution = Normal(action_mean, action_std)
        else:
            # 离散动作空间：输出分类分布
            action_probs = self.actor(features)
            distribution = Categorical(action_probs)
        
        # 值函数估计
        value = self.critic(features)
        
        return distribution, value
    
    def _initialize_weights(self):
        """
        初始化网络权重
        """
        for module in [self.shared, self.critic]:
            if self.continuous:
                modules = [self.actor_mean, module]
            else:
                modules = [self.actor, module]
            
            for m in modules:
                if isinstance(m, nn.Sequential):
                    for layer in m:
                        if isinstance(layer, nn.Linear):
                            nn.init.orthogonal_(layer.weight, gain=1.0)  # 正交初始化
                            nn.init.constant_(layer.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=1.0)
                    nn.init.constant_(m.bias, 0)
    
    def get_action(self, state, action=None):
        """
        根据状态获取动作、对数概率和熵
        
        参数:
            state (torch.Tensor): 状态输入
            action (torch.Tensor, optional): 如果提供，计算该动作的对数概率
            
        返回:
            torch.Tensor: 动作
            torch.Tensor: 对数概率
            torch.Tensor: 熵
        """
        distribution, _ = self.forward(state)
        
        # 如果没有提供动作，从分布中采样
        if action is None:
            action = distribution.sample()
        
        # 计算对数概率和熵
        log_prob = distribution.log_prob(action)
        entropy = distribution.entropy()
        
        # 对于连续动作空间，需要对对数概率进行求和
        if self.continuous:
            log_prob = log_prob.sum(dim=-1)
        
        return action, log_prob, entropy
    
    def evaluate(self, state, action):
        """
        评估给定状态-动作对的价值、对数概率和熵
        
        参数:
            state (torch.Tensor): 状态输入
            action (torch.Tensor): 动作
            
        返回:
            torch.Tensor: 状态价值
            torch.Tensor: 对数概率
            torch.Tensor: 熵
        """
        distribution, value = self.forward(state)
        
        # 计算对数概率和熵
        log_prob = distribution.log_prob(action)
        entropy = distribution.entropy()
        
        # 对于连续动作空间，需要对对数概率进行求和
        if self.continuous:
            log_prob = log_prob.sum(dim=-1)
        
        return value, log_prob, entropy

'''
如果要深入了解，下面是一些可以探索的问题：

1. 为什么使用共享特征提取器而不是完全分离的Actor和Critic网络？
2. 为什么连续动作空间使用正态分布而不是其他分布？
3. 为什么使用Tanh激活函数而不是ReLU？
4. 正交初始化对于训练稳定性有何影响？

'''

if __name__ == "__main__":
    """
    测试Actor-Critic网络
    """
    # 测试离散动作空间
    state_dim = 4
    action_dim = 2
    hidden_dim = 64
    continuous = False
    
    # 创建网络
    ac_discrete = ActorCritic(state_dim, action_dim, hidden_dim, continuous)
    
    # 打印网络结构
    print("离散动作空间Actor-Critic网络:")
    print(ac_discrete)
    
    # 测试前向传播
    batch_size = 5
    state = torch.randn(batch_size, state_dim)
    
    # 获取动作分布和价值
    dist, value = ac_discrete(state)
    
    print(f"\n状态形状: {state.shape}")
    print(f"价值形状: {value.shape}")
    print(f"价值: {value}")
    
    # 采样动作并计算对数概率
    action, log_prob, entropy = ac_discrete.get_action(state)
    
    print(f"\n动作形状: {action.shape}")
    print(f"动作: {action}")
    print(f"对数概率形状: {log_prob.shape}")
    print(f"对数概率: {log_prob}")
    print(f"熵: {entropy}")
    
    # 测试连续动作空间（可选）
    if True:
        continuous = True
        ac_continuous = ActorCritic(state_dim, action_dim, hidden_dim, continuous)
        
        print("\n\n连续动作空间Actor-Critic网络:")
        print(ac_continuous)
        
        # 获取动作分布和价值
        dist, value = ac_continuous(state)
        
        # 采样动作并计算对数概率
        action, log_prob, entropy = ac_continuous.get_action(state)
        
        print(f"\n动作形状: {action.shape}")
        print(f"动作: {action}")
        print(f"对数概率形状: {log_prob.shape}")
        print(f"对数概率: {log_prob}")
        print(f"熵: {entropy}")
    
    print("\n测试通过！")
