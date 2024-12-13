import torch
import pygame
import os
import random

from envs import FlappyBirdEnv
from agents import DQNAgent

# 確定是否有 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

NUM_EPISODES = 50
STEP_PER_EPISODE = 200

# initialize Environment and Agent
env = FlappyBirdEnv('numeric', False)

agent = DQNAgent(env.get_observation_shape(), 2, lr=0.001, device=device) # 將 Agent 放到 GPU 上
print(env.get_observation_shape())
agent.eval_net.train()
agent.target_net.train()

epsilon = 1.0
min_epsilon = 0.1
decrease_batch = NUM_EPISODES * 0.8  
decrease_epsilon = 0.98 / decrease_batch

for i_epoch in range(NUM_EPISODES):
    state, _ = env.reset()  # reset environment to initial state for each episode
    state = torch.tensor(state, device=device)  # 將 state 放到 GPU 上
    epoch_total_reward = 0

    for i_step in range(STEP_PER_EPISODE):
        action = agent.sample(state, epsilon=epsilon)

        next_state, reward, truncated, info = env.step(action)
        next_state = torch.tensor(next_state, device=device)  # 放到 GPU 上

        agent.store_transition(state.cpu().numpy(), action, reward, next_state.cpu().numpy())

        agent.learn()  # 確保你的 learn 方法也處理 GPU 資料
        
        state = next_state
        if truncated:
            state, _ = env.reset()
            state = torch.tensor(state, device=device)
            epoch_total_reward += info['total_reward']
    
    if (i_epoch + 1) % 10 == 0:
        print(f"Episode {i_epoch + 1}/{NUM_EPISODES}, "
              f"Lr: {agent.optimizer.param_groups[0]['lr']: .6f}, "
              f"Epsilon: {epsilon: .4f}, "
              f"Final Reward: {epoch_total_reward: .2f}")
    
    if i_epoch < decrease_batch:
        epsilon = max(min_epsilon, epsilon - decrease_epsilon)

# 儲存模型
output_dir = './dqn_model'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

seed_code = random.randint(100, 999)
file_name = f'q_learning_{NUM_EPISODES}_{seed_code}.ckpt'
save_path = os.path.join(output_dir, file_name)
agent.save(save_path)

print(f"Model saved to: {save_path}")
