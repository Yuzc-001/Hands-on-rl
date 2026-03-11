# LunarLander + PPO

## Why this example exists
This example is meant to stretch the learner beyond the simplest RL case.

It helps learners see:
- why policy optimization becomes important
- why more complex environments expose stability problems
- why training behavior is often noisier than in CartPole

## What to observe
- does reward show direction over time, even if variance is high?
- does policy behavior become less erratic?
- are updates too aggressive or still learning reasonably?

## Common mistakes
- expecting the same smoothness as CartPole
- reading all variance as failure
- adjusting clipping / learning settings blindly before checking implementation details

## After running
Use `docs/experiment-review.md` to capture what happened.
