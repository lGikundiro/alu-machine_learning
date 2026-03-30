#!/usr/bin/env python3
"""
Module implementing the full training loop using Monte-Carlo policy gradients
"""

import numpy as np
from policy_gradient import policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """
    Implements a full training using Monte-Carlo policy gradients

    Parameters:
    env: initial environment
    nb_episodes: number of episodes used training
    alpha: learning rate
    gamma: discount factor
    show_result: if True, render environment every 1000 episodes

    Returns:
    scores: list of all values of score (sum of rewards each episode)
    """
    # Initialize weight matrix randomly
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    weight = np.random.rand(state_size, action_size)

    scores = []

    for episode in range(nb_episodes):
        state = env.reset()[None, :]
        episode_states = []
        episode_gradients = []
        episode_rewards = []

        done = False
        while not done:
            # Render every 1000 episodes if show_result is True
            if show_result and episode % 1000 == 0:
                env.render()

            action, gradient = policy_gradient(state, weight)
            next_state, reward, done, _ = env.step(action)

            episode_states.append(state)
            episode_gradients.append(gradient)
            episode_rewards.append(reward)

            state = next_state[None, :]

        # Compute discounted returns
        T = len(episode_rewards)
        returns = np.zeros(T)
        G = 0
        for t in reversed(range(T)):
            G = episode_rewards[t] + gamma * G
            returns[t] = G

        # Update weights using policy gradient
        for t in range(T):
            weight += alpha * episode_gradients[t] * returns[t]

        score = sum(episode_rewards)
        scores.append(score)

        print("Episode: {}, Score: {}".format(episode + 1, score),
              end="\r", flush=False)

    return scores
