import random
from collections import deque

import gym
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


EPISODES = 100
EPISODE_DURATION = 500
BATCH_SIZE = 32


class DQAgent:
    def __init__(self, observation_space_size, action_space_size):
        self.observation_space_size = observation_space_size
        self.action_space_size = action_space_size
        self.memory = deque(maxlen=2048)
        self.exploration_rate = 1
        self.exploration_rate_min = 0.01
        self.exploration_rate_decay = 0.95
        self.learning_rate = 0.001
        self.discount_rate = 0.9
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(16, input_dim=self.observation_space_size, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(self.action_space_size, activation='linear'))
        model.compile(optimizer=Adam(lr=self.learning_rate), loss='mse')
        return model

    def action(self, state):
        if random.uniform(0, 1) < self.exploration_rate:
            return random.randrange(self.action_space_size)
        return np.argmax(self.predict_qualities(state))

    def predict_qualities(self, state):
        return self.model.predict(state)[0]

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        mini_batch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in mini_batch:
            new_quality = reward
            if not done:
                next_action_quality = np.max(self.predict_qualities(next_state))
                new_quality = (reward + self.discount_rate * next_action_quality)

            new_qualities = self.predict_qualities(state)
            new_qualities[action] = new_quality
            new_qualities = new_qualities.reshape(1, -1)
            self.model.fit(state, new_qualities, verbose=0)

        if self.exploration_rate > self.exploration_rate_min:
            self.exploration_rate *= self.exploration_rate_decay

    def save(self, filename):
        self.model.save_weights(filename)

    def load(self, filename):
        self.model.load_weights(filename)


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    agent = DQAgent(env.observation_space.shape[0], env.action_space.n)

    for e in range(EPISODES):
        state = env.reset().reshape(1, -1)

        for t in range(EPISODE_DURATION):
            action = agent.action(state)
            next_state, reward, done, info = env.step(action)
            next_state = next_state.reshape(1, -1)
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                print(f'Episode {e}: {t}')
                break

            if len(agent.memory) >= BATCH_SIZE:
                agent.replay(BATCH_SIZE)
