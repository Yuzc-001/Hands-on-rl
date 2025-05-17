"""
LunarLander环境的PPO实现示例

这个文件展示了如何使用PPO算法在LunarLander-v2环境中训练智能体
LunarLander是一个控制问题，目标是安全地将着陆器降落在指定区域
"""

import sys
import os
import torch
import gym
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 添加父目录到系统路径，以便导入PPO实现
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PPO.PPO import PPOAgent, train_ppo, evaluate_agent, plot_training_results


def run_lunarlander_ppo():
    """
    在LunarLander-v2环境中训练和评估PPO智能体
    """
    print("=" * 50)
    print("开始在LunarLander-v2环境中训练PPO智能体")
    print("=" * 50)
    
    # 环境参数
    env_name = "LunarLander-v2"
    
    # PPO超参数
    hidden_dim = 128       # 隐藏层大小
    lr = 3e-4              # 学习率
    gamma = 0.99           # 折扣因子
    lam = 0.95             # GAE lambda参数
    clip_ratio = 0.2       # PPO裁剪比率
    target_kl = 0.01       # 目标KL散度
    value_coef = 0.5       # 值函数损失系数
    entropy_coef = 0.01    # 熵正则化系数
    max_grad_norm = 0.5    # 梯度裁剪阈值
    
    # 训练参数
    num_epochs = 100       # 训练轮数
    steps_per_epoch = 4000 # 每轮步数
    update_epochs = 10     # 每批数据的更新轮数
    batch_size = 128       # 小批量大小
    eval_freq = 5          # 评估频率
    save_freq = 20         # 保存频率
    
    # 创建保存目录
    save_dir = "./models/ppo"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 训练PPO智能体
    agent, history = train_ppo(
        env_name=env_name,
        num_epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        hidden_dim=hidden_dim,
        lr=lr,
        gamma=gamma,
        lam=lam,
        clip_ratio=clip_ratio,
        target_kl=target_kl,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        max_grad_norm=max_grad_norm,
        update_epochs=update_epochs,
        batch_size=batch_size,
        eval_freq=eval_freq,
        save_freq=save_freq,
        render_final=True,
        save_dir=save_dir
    )
    
    print("\n训练完成！")
    
    # 绘制训练结果
    plot_training_results(history, env_name)
    
    return agent, history


if __name__ == "__main__":
    agent, history = run_lunarlander_ppo()
    print("LunarLander PPO示例完成！")
