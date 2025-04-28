# === train.py ===
import gym
import numpy as np
from agent import Agent
from utils import epsilon_by_frame

env = gym.make("CartPole-v1")
agent = Agent(env.observation_space.shape[0], env.action_space.n)
num_episodes = 500

for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        epsilon = epsilon_by_frame(episode)
        action = agent.select_action(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.memory.push(state, action, reward, next_state, done)
        state = next_state
        agent.update()
        total_reward += reward

    if episode % 10 == 0:
        agent.update_target()
        print(f"Episode {episode}, Total Reward: {total_reward}")

env.close()