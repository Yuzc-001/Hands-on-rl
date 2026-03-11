# Algorithm × Environment Map

## Why this map exists
This project is not only about algorithms.
It is about how algorithm choice and environment structure interact.

## DQN × CartPole
Why this pairing works well for learning:
- discrete action space
- relatively fast feedback
- easier to observe reward improvement
- easier to understand exploration and value learning

What it teaches:
- Q-value learning
- replay buffer purpose
- target network role
- basic RL debugging habits

## PPO × LunarLander
Why this pairing works well for learning:
- more complex control problem
- more visible instability and variance
- better exposure to policy optimization ideas
- stronger need for stable updates

What it teaches:
- actor-critic structure
- policy update sensitivity
- clipping intuition
- why more complex environments need more careful interpretation

## Core lesson
Do not ask only "which algorithm is better?"
Ask:
- what kind of environment is this?
- what kind of learning signal does it provide?
- what kind of algorithm behavior does it reward or punish?
