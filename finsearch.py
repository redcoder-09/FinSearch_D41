import random
import gym
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from collections import deque
import matplotlib.pyplot as plt

# Defining the Agent
class DQNAgent:
    def _init_(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)    # to store the passes
        self.gamma = 0.99   # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.model = self.build_model()

    def build_model(self):
        # Create Model
        model = Sequential()
        model.add(Dense(18, input_dim=self.state_size, activation='relu'))
        model.add(Dense(16, activation='sigmoid'))
        model.add(Dense(self.action_size, activation='linear'))
        # Compile Model
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)  # exploration
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])   # 0 defines axis for max
    
    def replay(self, batch_size):
        mini_batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Create the environment
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Initialize the Agent
agent = DQNAgent(state_size, action_size)

# Training Loop
batch_size = 32
num_episodes = 100  # Adjusted for a more substantial training session
episode_timesteps = []  # To store timesteps per episode

for episode in range(num_episodes):
    state = env.reset().reshape(1, state_size)
    total_timesteps = 0
    for t in range(500):
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        next_state = next_state.reshape(1, state_size)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_timesteps += 1
        if done:
            print(f"Episode {episode+1} finished after {t+1} timesteps")
            episode_timesteps.append(total_timesteps)
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

# Plotting
plt.plot(np.arange(len(episode_timesteps)), episode_timesteps)
plt.xlabel('Episodes')
plt.ylabel('Timesteps')
plt.title('CartPole Training Progress')
plt.grid(True)
plt.show()