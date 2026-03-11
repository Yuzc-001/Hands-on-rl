# Run Guide

## Install dependencies
```bash
pip install -r requirements.txt
```

## Run DQN on CartPole
```bash
python examples/cartpole_dqn.py
```

## Run PPO on LunarLander
```bash
python examples/lunarlander_ppo.py
```

## What to watch while running
- whether reward trends upward
- whether exploration / policy behavior changes over time
- whether learning appears stable or random
- whether logs and curves roughly match `docs/expected-behavior.md`

## After running
Do not stop at "it finished."
Use `docs/experiment-review.md` to record:
- what happened
- what changed the result
- what should be remembered next time
