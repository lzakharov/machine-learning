import random

import gym
import numpy as np


class Agent:
    def __init__(self, observation_space_size, action_space_size):
        self.observation_space_size = observation_space_size
        self.action_space_size = action_space_size
        self.q_table = np.zeros((observation_space_size, action_space_size))
        self.exploration_rate = 1
        self.exploration_rate_min = 0.01
        self.exploration_rate_decay = 0.95
        self.learning_rate = 0.1
        self.discount_rate = 0.6

    def action(self, state):
        if random.uniform(0, 1) < self.exploration_rate:
            return random.randrange(self.action_space_size)
        return np.argmax(self.q_table[state])

    def update(self, state, action, next_state, reward):
        old_value = self.q_table[state, action]
        next_action_quality = np.max(self.q_table[next_state])
        new_value = ((1 - self.learning_rate) * old_value +
                     self.learning_rate * (reward + self.discount_rate * next_action_quality))
        self.q_table[state, action] = new_value

        if self.exploration_rate > self.exploration_rate_min:
            self.exploration_rate *= self.exploration_rate_decay

    def save(self, filename):
        np.save(filename, self.q_table)

    def load(self, filename):
        self.q_table = np.load(filename)


if __name__ == '__main__':
    env = gym.make('Taxi-v2')
    agent = Agent(env.observation_space.n, env.action_space.n)
    episodes = 100000

    # train
    for episode in range(episodes):
        state = env.reset()
        done = False

        while not done:
            action = agent.action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, next_state, reward)
            state = next_state

    # evaluate
    state = env.reset()
    done = False

    while not done:
        action = np.argmax(agent.q_table[state])
        state, _, done, _ = env.step(action)
        env.render()
