#!/usr/bin/env python3
"""Module to play an episode using the trained Q-table."""
import numpy as np


def play(env, Q, max_steps=100):
    """Play an episode using the trained agent.

    Args:
        env: the FrozenLakeEnv instance
        Q: a numpy.ndarray containing the Q-table
        max_steps: maximum number of steps in the episode

    Returns:
        the total rewards for the episode
    """
    action_names = {0: 'Left', 1: 'Down', 2: 'Right', 3: 'Up'}
    state, _ = env.reset()
    total_rewards = 0
    nrow, ncol = env.desc.shape

    def render_state(s):
        """Render the current board state with position marked."""
        row = s // ncol
        col = s % ncol
        for r in range(nrow):
            line = ''
            for c in range(ncol):
                char = env.desc[r, c].decode('utf-8')
                if r == row and c == col:
                    line += '`' + char + '`'
                else:
                    line += char
            print(line)

    render_state(state)

    for _ in range(max_steps):
        action = np.argmax(Q[state, :])
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        print('  ({})'.format(action_names[action]))
        total_rewards += reward
        state = next_state

        render_state(state)

        if done:
            break

    return total_rewards
