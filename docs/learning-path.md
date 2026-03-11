# Learning Path

## Goal
This project is designed to help learners move through reinforcement learning in a usable order:
1. understand the task
2. understand the environment
3. understand the algorithm skeleton
4. implement the core modules
5. run training
6. observe the results
7. review what happened

## Recommended order

### Phase 1: DQN
Start with DQN if you are early in reinforcement learning.

Why:
- discrete action space is easier to reason about
- Q-learning concepts are more concrete
- CartPole gives fast feedback

Recommended path:
1. read `DQN/README.md`
2. read `01_q_network.py`
3. read `02_replay_buffer.py`
4. read `03_dqn_agent.py`
5. read `04_train_dqn.py`
6. read `DQN.py`
7. run `examples/cartpole_dqn.py`

### Phase 2: PPO
Move to PPO after DQN.

Why:
- introduces policy optimization
- introduces actor-critic structure
- helps explain advantage, clipping, and update stability

Recommended path:
1. read the PPO step files in order
2. read `PPO.py`
3. run `examples/lunarlander_ppo.py`

## What to focus on while learning
Do not only ask "does the code run?"
Also ask:
- what is the agent learning?
- what feedback does the environment provide?
- what signs show healthy training?
- what signs show a bug or mismatch?

## Completion standard
You have not really learned the algorithm until you can:
- explain why the environment is hard
- explain why the algorithm fits that environment
- identify one likely failure mode
- interpret a simple training curve
