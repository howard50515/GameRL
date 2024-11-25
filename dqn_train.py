import pygame
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from envs import FlappyBirdEnv
from agents import DQNAgent

NUM_EPISODES = 100
STEP_PER_EPISODE = 1000

# initialize Environment and Agent
env = FlappyBirdEnv(0, False)

agent = DQNAgent(env.get_observation_shape(), 2)
agent.eval_net.train()
agent.target_net.train()

epsilon = 1.
decrease_batch = NUM_EPISODES * 0.6
decrease_epsilon = 0.98 / decrease_batch

q_values_log = []
epsilon_log = []

for i_epoch in range(NUM_EPISODES):
    state, _ = env.reset() # reset environment to initial state for each episode

    q_values = agent.eval_net(torch.tensor(state, dtype=torch.float32)).detach().numpy()
    q_values_log.append(np.max(q_values))  # Record the max Q-value
    epsilon_log.append(epsilon)

    epoch_total_reward = 0
    for i_step in range(STEP_PER_EPISODE):
        action = agent.sample(state, epsilon=epsilon)

        next_state, reward, truncated, info = env.step(action)

        agent.store_transition(state, action, reward, next_state)

        agent.learn()
        
        state = next_state
        if truncated:
            state, _ = env.reset()
            epoch_total_reward += info['total_reward']

    pipes_cleared = info['pipes_cleared']
    if pipes_cleared > 0:
        print(f"Episode {i_epoch + 1}/{NUM_EPISODES}, Pipes Cleared: {pipes_cleared}")
        
    if (i_epoch + 1) % 5 == 0:
        print(f"{i_epoch + 1}/{NUM_EPISODES}, Lr: {agent.optimizer.param_groups[0]['lr']: .6f}, Epsilon: {epsilon: 4.4f}, Final Reward: {epoch_total_reward: 4.2f}")
    
    # graually descrease epsilon
    if i_epoch < decrease_batch:
        epsilon -= decrease_epsilon

# Plot Q-values and Epsilon
plt.figure(figsize=(12, 6))

# Plot Q-values
plt.subplot(1, 2, 1)
plt.plot(q_values_log, label="Max Q-value", color="blue")
plt.xlabel("Episode")
plt.ylabel("Max Q-value")
plt.title("Q-values over Episodes")
plt.legend()

# Plot Epsilon
plt.subplot(1, 2, 2)
plt.plot(epsilon_log, label="Epsilon", color="red")
plt.xlabel("Episode")
plt.ylabel("Epsilon")
plt.title("Epsilon over Episodes")
plt.legend()

plt.tight_layout()
plt.show()

# 儲存模型
output_dir = './dqn_model'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

seed_code = random.randint(100, 999)
file_name = f'q_learning_{NUM_EPISODES}_{seed_code}.ckpt'
save_path = os.path.join(output_dir, file_name)
agent.save(save_path)

print(f"Model saved to: {save_path}")
