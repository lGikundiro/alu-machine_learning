#!/usr/bin/env python3
"""
Module that performs the SARSA(lambda) algorithm for Q-value estimation
"""

import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1,
                  epsilon_decay=0.05):
    """
    Performs the SARSA(lambda) algorithm for Q-value estimation

    Parameters:
    env: openAI environment instance
    Q: numpy.ndarray of shape (s, a) containing the Q table
    lambtha: eligibility trace factor
    episodes: total number of episodes to train over
    max_steps: maximum number of steps per episode
    alpha: learning rate
    gamma: discount rate
    epsilon: initial threshold epsilon greedy
    min_epsilon: minimum value epsilon should decay to
    epsilon_decay: decay rate updating epsilon between episodes

    Returns:
    Q: updated Q table
    """
    n_states, n_actions = Q.shape

    def epsilon_greedy(state, eps):
        """Selects action using epsilon-greedy policy"""
        if np.random.uniform() < eps:
            return np.random.randint(n_actions)
        return np.argmax(Q[state])

    for episode in range(episodes):
        state = env.reset()
        action = epsilon_greedy(state, epsilon)

        # Initialize eligibility traces to zero each episode
        Et = np.zeros((n_states, n_actions))

        for _ in range(max_steps):
            next_state, reward, done, _ = env.step(action)
            next_action = epsilon_greedy(next_state, epsilon)

            # Compute TD error
            delta = reward + gamma * Q[next_state, next_action] - Q[
                state, action
            ]

            # Update eligibility trace
            Et[state, action] += 1

            # Update Q and eligibility traces
            Q = Q + alpha * delta * Et
            Et = gamma * lambtha * Et

            state = next_state
            action = next_action
            if done:
                break

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon - epsilon_decay)

    return Q
