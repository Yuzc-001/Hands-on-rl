"""
PPO步骤3: PPO智能体

在这个文件中，我们将实现PPO智能体，它整合Actor-Critic网络和PPO缓冲区
并实现PPO的核心算法逻辑，包括裁剪目标函数和多轮更新
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 导入前面实现的组件
from ppo.01_actor_critic import ActorCritic
from ppo.02_ppo_buffer import PPOBuffer


class PPOAgent:
    """
    PPO智能体实现，包含PPO算法的核心逻辑。
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=64, continuous=False,
                 lr=3e-4, gamma=0.99, lam=0.95, clip_ratio=0.2,
                 target_kl=0.01, value_coef=0.5, entropy_coef=0.01,
                 max_grad_norm=0.5, update_epochs=10, buffer_size=2048, batch_size=64):
        """
        初始化PPO智能体
        
        参数:
            state_dim (int): 状态空间维度
            action_dim (int): 动作空间维度
            hidden_dim (int): 隐藏层维度
            continuous (bool): 是否是连续动作空间
            lr (float): 学习率
            gamma (float): 折扣因子
            lam (float): GAE lambda参数
            clip_ratio (float): PPO裁剪比率
            target_kl (float): 目标KL散度（用于早停）
            value_coef (float): 值函数损失系数
            entropy_coef (float): 熵正则化系数
            max_grad_norm (float): 梯度裁剪阈值
            update_epochs (int): 每批数据的更新轮数
            buffer_size (int): 缓冲区大小
            batch_size (int): 训练批大小
        """
        # 环境参数
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous = continuous
        
        # PPO超参数
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        
        # 创建Actor-Critic网络
        self.actor_critic = ActorCritic(state_dim, action_dim, hidden_dim, continuous)
        
        # 设置优化器
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        
        # 创建PPO缓冲区
        self.buffer = PPOBuffer(state_dim, action_dim, buffer_size, gamma, lam)
        
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor_critic.to(self.device)
        
        print(f"PPOAgent初始化完成，使用设备: {self.device}")
    
    def select_action(self, state):
        """
        根据当前策略选择动作
        
        参数:
            state: 当前环境状态
            
        返回:
            action: 选择的动作
            log_prob: 动作的对数概率
            value: 状态值估计
        """
        # 转换状态为张量
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        
        # 使用Actor选择动作
        with torch.no_grad():
            action, log_prob, _ = self.actor_critic.get_action(state)
            _, value = self.actor_critic(state)
        
        # 转换为CPU上的NumPy数组
        action = action.cpu().numpy()
        log_prob = log_prob.cpu().numpy()
        value = value.cpu().numpy().squeeze()
        
        # 如果是连续动作空间，可能需要裁剪动作到合法范围
        if self.continuous:
            # 这里假设动作范围是[-1, 1]，可以根据实际环境调整
            action = np.clip(action, -1.0, 1.0)
        
        return action, log_prob, value
    
    def store_transition(self, state, action, reward, value, log_prob, done):
        """
        存储一个转换到缓冲区
        
        参数:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            value: 状态值估计
            log_prob: 动作的对数概率
            done: 是否是终止状态
        """
        self.buffer.store(state, action, reward, value, log_prob, done)
    
    def finish_path(self, last_value=0):
        """
        完成当前轨迹并计算优势
        
        参数:
            last_value: 最后状态的值估计（如果轨迹被截断）
        """
        self.buffer.finish_path(last_value)
    
    def update(self):
        """
        使用收集的数据更新策略和值函数
        
        返回:
            metrics: 包含训练指标的字典
        """
        # 获取缓冲区中的所有数据
        data = self.buffer.get()
        
        # 将数据移到设备
        states = data["states"].to(self.device)
        actions = data["actions"].to(self.device)
        old_logprobs = data["logprobs"].to(self.device)
        returns = data["returns"].to(self.device)
        advantages = data["advantages"].to(self.device)
        old_values = data["values"].to(self.device)
        
        # 训练指标
        metrics = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "kl": [],
            "clip_frac": [],
            "explained_variance": []
        }
        
        # 计算值函数的解释方差（衡量值函数预测质量）
        explained_variance = 1 - torch.var(returns - old_values) / torch.var(returns)
        metrics["explained_variance"] = explained_variance.item()
        
        # 多轮更新
        for i in range(self.update_epochs):
            # 生成随机小批量索引
            indices = torch.randperm(len(states))
            
            # 按批次处理数据
            for start in range(0, len(states), self.batch_size):
                # 获取小批量数据
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_logprobs = old_logprobs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # 对当前策略进行评估
                values, logprobs, entropy = self.actor_critic.evaluate(batch_states, batch_actions)
                
                # 计算重要性采样比率
                ratio = torch.exp(logprobs - batch_old_logprobs)
                
                # 裁剪目标函数（PPO的核心）
                clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(ratio * batch_advantages, clip_adv).mean()
                
                # 值函数损失（MSE损失）
                value_loss = 0.5 * ((values.squeeze() - batch_returns) ** 2).mean()
                
                # 熵正则化
                entropy_loss = -entropy.mean()
                
                # 总损失
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # 优化器步骤
                self.optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪（防止梯度爆炸）
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
                
                # 计算KL散度（用于早停）
                approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                
                # 计算裁剪比例
                clip_frac = ((ratio - 1.0).abs() > self.clip_ratio).float().mean().item()
                
                # 记录指标
                metrics["policy_loss"].append(policy_loss.item())
                metrics["value_loss"].append(value_loss.item())
                metrics["entropy"].append(entropy.mean().item())
                metrics["kl"].append(approx_kl)
                metrics["clip_frac"].append(clip_frac)
            
            # 如果达到目标KL散度，提前停止
            if "kl" in metrics and np.mean(metrics["kl"]) > self.target_kl:
                print(f"提前停止更新：达到目标KL散度 {np.mean(metrics['kl']):.4f}")
                break
        
        # 计算平均指标
        for key in metrics.keys():
            if key != "explained_variance":  # 解释方差已经计算过
                metrics[key] = np.mean(metrics[key])
        
        return metrics
    
    def save(self, filepath):
        """
        保存模型参数
        
        参数:
            filepath (str): 保存文件路径
        """
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
        print(f"模型已保存到 {filepath}")
    
    def load(self, filepath):
        """
        加载模型参数
        
        参数:
            filepath (str): 加载文件路径
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"模型已从 {filepath} 加载")


'''
如果要深入了解，下面是一些可以探索的问题：

1. 裁剪目标函数如何防止过大的策略更新？
2. 为什么多轮更新会提高样本效率？
3. KL散度早停的意义是什么？
4. 如何调整超参数来改善PPO性能？
'''

if __name__ == "__main__":
    """
    测试PPO智能体
    """
    # 创建一个小型环境的PPO智能体
    state_dim = 4
    action_dim = 2
    hidden_dim = 64
    continuous = False  # 离散动作空间
    
    # 初始化智能体
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        continuous=continuous,
        buffer_size=128,  # 小型缓冲区用于测试
        batch_size=32
    )
    
    # 测试动作选择
    test_state = np.random.rand(state_dim)
    action, log_prob, value = agent.select_action(test_state)
    
    print(f"测试状态: {test_state}")
    print(f"选择的动作: {action}")
    print(f"动作对数概率: {log_prob}")
    print(f"状态值估计: {value}")
    
    # 存储一些随机转换
    num_steps = 10
    for i in range(num_steps):
        state = np.random.rand(state_dim)
        action = np.random.randint(0, action_dim) if not continuous else np.random.rand(action_dim)
        reward = np.random.rand()
        next_state = np.random.rand(state_dim)
        value = np.random.rand()
        log_prob = np.random.rand()
        done = False if i < num_steps - 1 else True
        
        agent.store_transition(state, action, reward, value, log_prob, done)
    
    # 完成轨迹
    agent.finish_path()
    
    # 添加另一条轨迹
    for i in range(num_steps):
        state = np.random.rand(state_dim)
        action = np.random.randint(0, action_dim) if not continuous else np.random.rand(action_dim)
        reward = np.random.rand()
        next_state = np.random.rand(state_dim)
        value = np.random.rand()
        log_prob = np.random.rand()
        done = False if i < num_steps - 1 else True
        
        agent.store_transition(state, action, reward, value, log_prob, done)
    
    # 完成第二条轨迹
    agent.finish_path()
    
    # 测试更新
    metrics = agent.update()
    
    print("\n更新后的指标:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    # 测试保存和加载
    agent.save("test_ppo_model.pth")
    agent.load("test_ppo_model.pth")
    
    print("\nPPOAgent测试完成！")
