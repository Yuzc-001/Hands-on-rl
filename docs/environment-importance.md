# Why Environment Matters

In reinforcement learning, the environment is not a backdrop. It is the problem itself.

## Why this matters
The algorithm decides how learning happens.
The environment decides:
- what the agent observes
- what actions are possible
- how reward is generated
- why exploration is hard
- what successful learning even looks like

## Without environment understanding, learners get stuck
Common beginner failure modes:
- copying code without understanding the task
- treating reward curves as magic
- not knowing whether failure comes from the algorithm, the implementation, or the environment
- not understanding why one algorithm works better than another on a specific task

## What this project wants learners to see
### CartPole
A good first environment because:
- state is manageable
- action space is discrete
- feedback is relatively clear
- training loops are fast enough to inspect

### LunarLander
A better second environment because:
- the control problem is more complex
- exploration is harder
- stability matters more
- the gap between “runs” and “learns well” becomes more visible

## Core takeaway
If you do not understand the environment, you do not really understand the training behavior.

That is why this project treats environments as learning tools, not just demo targets.
