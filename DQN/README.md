# DQN

## 为什么这个部分重要
DQN 不是这个项目里“先讲的一个算法”，而是强化学习第一条核心路线的代表。

如果你刚进入强化学习，DQN 往往是最好的起点，因为它能帮助你先抓住最关键的几个问题：
- agent 到底在学什么？
- Q 值是什么意思？
- 为什么需要 replay buffer？
- 为什么 target network 能让训练更稳定？
- 探索和利用到底怎么平衡？

所以在这个项目里，DQN 的意义不是“一个经典算法案例”，而是：

# **帮助学习者真正进入值函数方法这条主线。**

---

## DQN 在这个项目里教什么

### 1. Value-based learning
理解智能体如何通过学习状态—动作价值来做决策。

### 2. Replay and stability
理解为什么强化学习训练容易不稳定，以及 replay buffer / target network 在稳定训练中的作用。

### 3. Exploration behavior
理解 epsilon-greedy 不只是一个技巧，而是“怎么在不确定中学习”的最初版本。

### 4. Training interpretation
理解 reward、loss、epsilon 这些训练现象分别在告诉你什么。

---

## 为什么先用 CartPole
DQN 最适合先放在像 CartPole 这样的环境里理解，因为它具备：
- 离散动作空间
- 相对清晰的反馈
- 更快看到训练变化
- 更容易把代码和环境现象对应起来

这让学习者可以先把注意力放在：
- 算法骨架
- 数据流
- 更新逻辑
- 训练现象

而不是一上来就被更复杂环境的噪声和不稳定性压住。

---

## 建议学习顺序
1. 先读 `01_q_network.py`
2. 再读 `02_replay_buffer.py`
3. 再读 `03_dqn_agent.py`
4. 再读 `04_train_dqn.py`
5. 最后看 `DQN.py`
6. 配合 `examples/cartpole_dqn.py` 与 `examples/cartpole_dqn.md` 一起理解
7. 再回到 `DQN/learning-map.md` 做整体收束

---

## 学这一部分时要特别关注什么
- Q 值的目标是怎么构成的
- replay buffer 解决了什么问题
- target network 为什么重要
- epsilon 如何影响训练节奏
- reward 曲线、loss 曲线、探索率变化应如何一起看

---

## 常见误区
- 以为 DQN 只是“神经网络版 Q-learning”这么简单
- 只会跑代码，不知道 target / replay 为什么存在
- 一看到 early noise 就觉得训练完全失败
- 一出问题就盲调超参数，而不先检查 target 计算和 buffer 数据流

---

## 这个部分学会了，意味着什么
你不只是会跑 DQN。
你会开始真正理解：
- 为什么值函数方法是强化学习的重要基础
- 为什么稳定训练在 RL 里这么难
- 为什么很多后续算法，本质上都在修正这里暴露出来的问题
