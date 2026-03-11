# Examples

These examples are not only runnable scripts. They are the first practical checkpoints in the learning path.

## Why they matter
A learner has not really understood the project if they only read the code.
They should also:
- run the training loop,
- observe the learning behavior,
- compare the result with expectations,
- and reflect on what changed the outcome.

## Included examples

### 1. `cartpole_dqn.py`
Use this to understand:
- why DQN is a good first algorithm
- why CartPole is a good first environment
- how value learning looks when training is relatively stable

### 2. `lunarlander_ppo.py`
Use this to understand:
- why PPO becomes necessary beyond simpler value-based settings
- why more complex environments expose stability challenges
- how policy optimization behaves differently from DQN

## How to use examples well
Do not only ask "does it run?"
Also ask:
- what should improve first?
- what looks normal vs suspicious?
- is the problem algorithmic, environmental, or implementation-level?

## Recommended loop
1. run the example
2. observe the training result
3. compare with `docs/expected-behavior.md`
4. use `docs/experiment-review.md` to write down what happened
