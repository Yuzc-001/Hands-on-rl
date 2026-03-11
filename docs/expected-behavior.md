# Expected Training Behavior

This file explains what learners should roughly expect when training works normally.

## CartPole with DQN
Typical signs of healthy learning:
- episode reward trends upward over time
- epsilon gradually decreases
- performance becomes more stable after early exploration
- evaluation reward is meaningfully better than random behavior

Possible warning signs:
- reward stays flat for too long
- loss is unstable without reward improvement
- epsilon decays too fast and learning stalls
- replay buffer or target update logic may be wrong

## LunarLander with PPO
Typical signs of healthy learning:
- reward improves more slowly than CartPole but shows directionality
- variance is high early on
- policy gradually stabilizes
- evaluation becomes less erratic over time

Possible warning signs:
- reward is highly noisy with no trend at all
- policy collapses to poor repeated behavior
- returns / advantages may be miscomputed
- clipping or update settings may be too aggressive

## Rule
Do not interpret every noisy curve as failure. First ask whether the observed behavior matches the difficulty of the environment.
