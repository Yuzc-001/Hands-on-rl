"""
PPO完整实现

这个文件包含PPO算法的完整实现，结合了所有核心组件：
1. Actor-Critic网络: 同时输出策略和价值的神经网络
2. PPO缓冲区与GAE: 存储轨迹并计算广义优势估计
3. PPO智能体: 实现裁剪目标函数和多轮更新
4. 训练循环: 管理整个训练过程

通过这个文件，你可以直接训练PPO算法而不需要单独导入各个组件
"""

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
import matplotlib.pyplot as plt
from datetime import datetime
import os
import time


class ActorCritic(nn.Module):
    """
    Actor-Critic网络，包含一个共享特征提取器，以及策略（Actor）和值函数（Critic）两个头。
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=64, continuous=False):
        """
        初始化Actor-Critic网络
        
        参数:
            state_dim (int): 状态空间的维度
            action_dim (int): 动作空间的维度
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
            # 连续动作空间
            self.actor_mean = nn.Linear(hidden_dim, action_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            # 离散动作空间
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
            state (torch.Tensor): 状态输入
            
        返回:
            distribution: 动作概率分布
            value: 状态价值估计
        """
        # 共享特征提取
        features = self.shared(state)
        
        # 根据动作空间类型创建不同的分布
        if self.continuous:
            # 连续动作空间
            action_mean = self.actor_mean(features)
            action_std = torch.exp(self.actor_log_std)
            distribution = Normal(action_mean, action_std)
        else:
            # 离散动作空间
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
                            nn.init.orthogonal_(layer.weight, gain=1.0)
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


class PPOBuffer:
    """
    PPO缓冲区，用于存储轨迹并计算广义优势估计(GAE)。
    """
    
    def __init__(self, state_dim, action_dim, buffer_size, gamma=0.99, lam=0.95):
        """
        初始化PPO缓冲区
        
        参数:
            state_dim (int): 状态空间维度
            action_dim (int): 动作空间维度
            buffer_size (int): 缓冲区大小
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
            x: 要累积的值
            discount: 折扣因子
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
        adv_std = np.std(self.advantages) + 1e-8
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


class PPOAgent:
    """
    PPO智能体，实现PPO算法的核心逻辑。
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
        
        # 计算值函数的解释方差
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
                
                # 值函数损失
                value_loss = 0.5 * ((values.squeeze() - batch_returns) ** 2).mean()
                
                # 熵正则化
                entropy_loss = -entropy.mean()
                
                # 总损失
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # 优化器步骤
                self.optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
                
                # 计算KL散度
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
    
    # 如果要求，渲染最终策略
    if render_final:
        print("\n渲染最终策略...")
        _ = evaluate_agent(agent, env, num_episodes=3, render=True)
    
    # 关闭环境
    env.close()
    eval_env.close()
    
    return agent, history


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
    
    # 绘制训练结果
    plot_training_results(history, env_name)
    
    print("PPO训练完成！")
