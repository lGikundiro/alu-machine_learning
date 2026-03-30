# Q-learning

This directory contains implementations of Q-learning algorithms for reinforcement learning using the FrozenLake environment from OpenAI's Gym.

## Files

| File                  | Description                                |
| --------------------- | ------------------------------------------ |
| `0-load_env.py`       | Loads the FrozenLake environment           |
| `1-q_init.py`         | Initializes the Q-table with zeros         |
| `2-epsilon_greedy.py` | Implements epsilon-greedy action selection |
| `3-q_learning.py`     | Performs Q-learning training               |
| `4-play.py`           | Plays an episode using the trained Q-table |

## Tasks

### 0. Load the Environment

Loads the `FrozenLakeEnv` from OpenAI's gym with optional custom map, pre-made map, or random generation.

### 1. Initialize Q-table

Initializes a Q-table of zeros with shape `(n_states, n_actions)`.

### 2. Epsilon Greedy

Selects the next action using epsilon-greedy policy: explores randomly with probability epsilon, exploits the best known action otherwise.

### 3. Q-learning

Trains the agent using the Q-learning algorithm with the Bellman equation:

```
Q(s,a) = Q(s,a) + alpha * (reward + gamma * max(Q(s',a')) - Q(s,a))
```

### 4. Play

Plays a full episode using the trained Q-table, always exploiting (greedy), and renders each board state to the console.
