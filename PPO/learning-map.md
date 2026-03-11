# PPO Learning Map

## What to learn in order
1. actor-critic network
2. PPO buffer
3. agent update logic
4. training loop
5. full implementation
6. environment behavior through LunarLander

## What PPO helps you understand
- policy optimization
- actor-critic cooperation
- return and advantage estimation
- clipped updates for stability
- why policy learning can be sensitive but powerful

## Main environment fit
PPO becomes more meaningful when the task makes plain value-based intuition less sufficient.

## Watch out for
- incorrect advantage / return computation
- overly aggressive updates
- misreading variance as failure
- forgetting that complex environments naturally look messier than CartPole
