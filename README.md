# Hands-on RL

> **一个帮助你真正理解“算法—环境—训练现象”关系的强化学习实战作品。**

Hands-on RL 不是一个大而全的强化学习框架，也不是只给出几份能跑代码的教程仓库。

它更像一个**强化学习学习系统**：
- 用 **DQN** 和 **PPO** 抓住两条最核心的算法路线
- 用 **CartPole** 和 **LunarLander** 这类环境帮助你理解“为什么这样学、为什么会失败、为什么不同算法适合不同任务”
- 用 **step-by-step 实现 → 完整训练 → 结果观察 → 复盘迭代** 把学习闭环真正跑通

如果你想学的不是“会调库”，而是“真正学会强化学习是怎么工作的”，这个项目就是为此设计的。

---

## 为什么这个项目值得做

很多强化学习仓库有两个问题：
1. 要么太理论，学完还是不会写
2. 要么直接给完整工程，跑起来了却不知道为什么有效

Hands-on RL 想解决的是中间这层空白：

# **从理解算法，到手写实现，到跑通实验，到解释训练现象。**

这里的重点不是“覆盖更多算法”，而是：
- 先把最重要的学透
- 先把核心训练闭环打通
- 先学会怎么观察、调试、复盘

---

## 为什么先学 DQN 和 PPO

DQN 和 PPO 不是随便挑的两个算法。
它们分别代表强化学习最核心的两条路线：

- **DQN**：值函数方法
- **PPO**：策略优化 / actor-critic 方法

学懂这两个，不只是会两个算法名，而是抓住强化学习最重要的两套思维方式。

### 学 DQN，你会理解
- Q 值到底在学什么
- replay buffer 为什么重要
- target network 为什么能提高稳定性
- exploration / exploitation 如何平衡

### 学 PPO，你会理解
- policy 是怎么被直接优化的
- value function 为什么还重要
- advantage 是做什么的
- 为什么策略更新不能太猛
- 为什么 clip 能提高稳定性

所以这个项目不是“教两个热门算法”，而是：

# **用 DQN + PPO 帮你建立强化学习的核心认知框架。**

---

## 为什么环境同样重要

在强化学习里，环境不是 demo 背景板，而是学习发生的地方。

算法决定“怎么学”，环境决定：
- 学什么
- 为什么难
- 奖励如何产生
- 探索为什么会失败
- 训练现象为什么会长成那样

如果不理解环境，你很容易把强化学习学成一堆孤立公式：
- reward 为什么不涨不知道
- loss 异常不知道怎么解释
- 不知道是环境难，还是实现错了
- 也不知道为什么一个算法在某个环境好用，在另一个环境就不行

所以在这个项目里，环境不是附属物，而是理解 RL 的关键入口。

---

## 这个项目现在教什么

Hands-on RL 不是只教“代码怎么写”，而是教四层东西：

### 1. Core Algorithms
- DQN
- PPO

### 2. Learning Workflow
- 先学什么
- 先跑什么
- 先理解什么
- 如何一步步推进

### 3. Failure Boundaries
- 训练异常怎么看
- 常见错误在哪
- 什么时候不要盲调参数
- 什么时候先怀疑环境、接口、buffer、target、advantage

### 4. Experiment Review
- 训练结果怎么理解
- 这次实验学到了什么
- 下次该怎么改

---

## 当前项目结构

```text
Hands-on-rl/
├── README.md
├── README.zh-CN.md
├── PROJECT.md
├── ROADMAP.md
├── CHANGELOG.md
├── VERSION
├── requirements.txt
├── DQN/
│   ├── README.md
│   ├── learning-map.md
│   ├── 01_q_network.py
│   ├── 02_replay_buffer.py
│   ├── 03_dqn_agent.py
│   ├── 04_train_dqn.py
│   └── DQN.py
├── PPO/
│   ├── README.md
│   ├── learning-map.md
│   ├── 01_actor_critic.py
│   ├── 02_ppo_buffer.py
│   ├── 03_ppo_agent.py
│   ├── 04_train_ppo.py
│   └── PPO.py
├── examples/
│   ├── README.md
│   ├── cartpole_dqn.py
│   ├── cartpole_dqn.md
│   ├── lunarlander_ppo.py
│   └── lunarlander_ppo.md
└── docs/
    ├── 2026-03-11_Hands-on-rl重塑原则_v1.md
    ├── project-positioning.md
    ├── learner-entry.md
    ├── learning-path.md
    ├── environment-importance.md
    ├── algorithm-environment-map.md
    ├── debug-boundaries.md
    ├── common-mistakes.md
    ├── expected-behavior.md
    ├── what-good-looks-like.md
    ├── experiment-review.md
    ├── examples-as-learning-interfaces.md
    ├── run-guide.md
    ├── results-plan.md
    ├── track-overview.md
    ├── repo-map.md
    └── restructure-status.md
```

---

## 学习路径

如果你是第一次系统学强化学习，建议按这个顺序走：

### 第一阶段：先理解 DQN
1. 阅读 `DQN/README.md`
2. 按顺序看：
   - `01_q_network.py`
   - `02_replay_buffer.py`
   - `03_dqn_agent.py`
   - `04_train_dqn.py`
3. 再看 `DQN/DQN.py` 的完整实现
4. 运行 `examples/cartpole_dqn.py`

### 第二阶段：再理解 PPO
1. 阅读 PPO 相关文件
2. 按顺序理解 actor-critic、buffer、agent、train loop
3. 再看 `PPO/PPO.py`
4. 运行 `examples/lunarlander_ppo.py`

### 第三阶段：关注训练现象，而不是只看代码
重点问自己：
- reward 曲线正常吗？
- 探索和收敛的节奏合理吗？
- 如果训练失败，是环境难、实现错，还是超参数不合适？

更完整说明见：
- [README.zh-CN.md](README.zh-CN.md)
- [PROJECT.md](PROJECT.md)
- [DQN/README.md](DQN/README.md)
- [DQN/learning-map.md](DQN/learning-map.md)
- [PPO/README.md](PPO/README.md)
- [PPO/learning-map.md](PPO/learning-map.md)
- [docs/project-positioning.md](docs/project-positioning.md)
- [docs/learner-entry.md](docs/learner-entry.md)
- [docs/learning-path.md](docs/learning-path.md)
- [docs/environment-importance.md](docs/environment-importance.md)
- [docs/algorithm-environment-map.md](docs/algorithm-environment-map.md)
- [docs/debug-boundaries.md](docs/debug-boundaries.md)
- [docs/common-mistakes.md](docs/common-mistakes.md)
- [docs/expected-behavior.md](docs/expected-behavior.md)
- [docs/what-good-looks-like.md](docs/what-good-looks-like.md)
- [docs/experiment-review.md](docs/experiment-review.md)
- [docs/track-overview.md](docs/track-overview.md)
- [docs/repo-map.md](docs/repo-map.md)
- [docs/restructure-status.md](docs/restructure-status.md)
- [docs/run-guide.md](docs/run-guide.md)
- [docs/results-plan.md](docs/results-plan.md)
- [docs/examples-as-learning-interfaces.md](docs/examples-as-learning-interfaces.md)
- [examples/README.md](examples/README.md)

---

## 如何运行

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行 DQN on CartPole
```bash
python examples/cartpole_dqn.py
```
参见：[examples/cartpole_dqn.md](examples/cartpole_dqn.md)

### 运行 PPO on LunarLander
```bash
python examples/lunarlander_ppo.py
```
参见：[examples/lunarlander_ppo.md](examples/lunarlander_ppo.md)

更完整运行说明见：[docs/run-guide.md](docs/run-guide.md)

---

## 这个项目不做什么

为了保持作品聚焦，当前阶段**不做功能堆叠**。

这意味着：
- 不急着继续堆 SAC / TD3 / Rainbow / TRPO
- 不急着变成“大而全 RL 算法大全”
- 不优先卷 benchmark 数量

当前边界很明确：

# **只围绕 DQN / PPO，把“理解算法—理解环境—跑通训练—解释现象”这条学习闭环做透。**

---

## 项目元信息

- 当前版本：`v0.2.0`
- 路线图：[ROADMAP.md](ROADMAP.md)
- 变更记录：[CHANGELOG.md](CHANGELOG.md)
- 仓库导览：[docs/repo-map.md](docs/repo-map.md)

## 接下来会继续补什么

### v0.x 重塑方向
- 更清楚的学习路径文档
- 更清楚的环境与任务说明
- 更清楚的训练异常 / 调试边界说明
- 更清楚的实验复盘与结果解释方法
- 更好的 examples 与训练结果展示

---

## 一句话总结

**Hands-on RL 想做的，不是再给你一个 RL 代码仓库，而是给你一条真正能走进去的强化学习学习路径。**

如果你从这里离开时，不只是“看过 DQN / PPO”，而是开始理解：
- 为什么环境重要
- 为什么训练会成功或失败
- 为什么不同算法适合不同任务

那这个作品才算真正成立。
