# Debug Boundaries

This document explains what to check before blindly changing hyperparameters.

## Common mistake
When training looks wrong, beginners often start changing:
- learning rate
- batch size
- gamma
- hidden size
- update frequency

Sometimes that helps. Often it hides the real problem.

## Check these boundaries first

### 1. Environment boundary
- Is the environment API behaving as expected?
- Are state, reward, done, and reset values correct?
- Are you mixing gym and gymnasium conventions accidentally?

### 2. Data boundary
- Is replay buffer sampling correct?
- Are rewards, dones, and next states stored correctly?
- Are tensor shapes consistent?

### 3. Target / value boundary
- Is the target being computed correctly?
- Is the bootstrap term masked correctly when done is true?
- Is advantage or return calculation implemented correctly?

### 4. Training signal boundary
- Is loss exploding or flat?
- Is reward random or drifting?
- Is exploration collapsing too early?

## Rule
Do not tune blindly until you have checked:
1. environment interface
2. data flow
3. target computation
4. training signal behavior

A lot of “bad hyperparameters” problems are actually implementation or task-understanding problems.
