"""
DQN步骤1: Q-Network实现

在这个文件中，我们将实现DQN的核心组件之一：Q-Network
Q-Network是一个神经网络，用于近似Q函数，即在给定状态下每个动作的价值
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """
    Q-Network用于近似Q函数，输入状态，输出每个动作的预期价值。
    
    这是一个简单的前馈神经网络，包含两个隐藏层。
    网络架构可以根据任务的复杂性进行调整。
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
        
        # 初始化权重（可选，但有助于训练）
        self._initialize_weights()
    
    def forward(self, state):
        """
        前向传播，计算给定状态下每个动作的Q值
        
        参数:
            state (torch.Tensor): 状态输入，形状为(batch_size, state_dim)
            
        返回:
            torch.Tensor: 每个动作的Q值，形状为(batch_size, action_dim)
        """
        x = F.relu(self.fc1(state))  # 第一层 + ReLU激活
        x = F.relu(self.fc2(x))      # 第二层 + ReLU激活
        q_values = self.fc3(x)       # 输出层，无激活函数
        
        return q_values
    
    def _initialize_weights(self):
        """
        初始化网络权重，使用He初始化（适用于ReLU激活函数）
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)


# 如果要深入了解，下面是一些可以探索的问题：
#
# 1. 为什么我们使用ReLU而不是其他激活函数？
# 2. 如何选择隐藏层的大小？
# 3. 权重初始化为什么重要，有哪些替代方法？
# 4. 如何修改网络架构以处理图像输入（如Atari游戏）？


if __name__ == "__main__":
    """
    测试Q-Network
    """
    # 创建一个小型环境的Q-Network（如CartPole，状态维度为4，动作维度为2）
    state_dim = 4
    action_dim = 2
    q_network = QNetwork(state_dim, action_dim)
    
    # 打印网络结构
    print(q_network)
    
    # 测试前向传播
    batch_size = 5  # 一批5个状态
    sample_state = torch.randn(batch_size, state_dim)  # 随机生成5个状态
    q_values = q_network(sample_state)  # 计算Q值
    
    print(f"\n输入状态形状: {sample_state.shape}")
    print(f"输出Q值形状: {q_values.shape}")
    print(f"Q值示例:\n{q_values}")
    
    # 验证输出形状
    assert q_values.shape == (batch_size, action_dim), "输出形状不符合预期！"
    print("\n测试通过！")
