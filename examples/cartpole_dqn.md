# CartPole + DQN

## Why this example exists
This is the best first practical entry point for the project.

It helps learners see:
- why DQN is easier to start with than PPO
- how a discrete-action environment fits value-based learning
- how reward improvement can become visible relatively quickly

## What to observe
- does reward trend upward over time?
- does behavior become less random as epsilon decays?
- does evaluation perform better than early episodes?

## Common mistakes
- treating a noisy early curve as total failure
- tuning hyperparameters before checking replay / target logic
- focusing on code only, not on environment feedback

## After running
Use `docs/experiment-review.md` to capture what happened.
