import pygame
import os
import random

from envs import FlappyBirdEnv
from agents import DQNAgent

NUM_EPISODES = 600
STEP_PER_EPISODE = 600

# initialize Environment and Agent
#env = FlappyBirdEnv(1, True)
env = FlappyBirdEnv('numeric', True)

agent = DQNAgent(env.get_observation_shape(), 2, lr=0.001)
print(env.get_observation_shape())
agent.eval_net.train()
agent.target_net.train()

epsilon = 0.5
min_epsilon = 0.0
decrease_batch = NUM_EPISODES * 0.5 
#decrease_epsilon = (epsilon - min_epsilon) / decrease_batch
decrease_epsilon = 0.5 / decrease_batch

for i_epoch in range(NUM_EPISODES):
    state, _ = env.reset() # reset environment to initial state for each episode
    epoch_total_reward = 0
    count = 0
    for i_step in range(STEP_PER_EPISODE):
        action = agent.sample(state, epsilon=epsilon)

        next_state, reward, truncated, info = env.step(action)

        agent.store_transition(state, action, reward, next_state)

        agent.learn()
        for event in pygame.event.get():
            pass
        state = next_state
        if truncated:
            state, _ = env.reset()
            epoch_total_reward += info['total_reward']
    
    #if (i_epoch + 1) % 5 == 0:
        #print(f"{i_epoch + 1}/{NUM_EPISODES}, Lr: {agent.optimizer.param_groups[0]['lr']: .6f}, Epsilon: {epsilon: 4.4f}, Final Reward: {epoch_total_reward: 4.2f}")
    if (i_epoch + 1) % 10 == 0:
        print(f"Episode {i_epoch + 1}/{NUM_EPISODES}, "
              f"Lr: {agent.optimizer.param_groups[0]['lr']: .6f}, "
              f"Epsilon: {epsilon: .4f}, "
              f"Final Reward: {epoch_total_reward: .2f}")
    
    # graually descrease epsilon
    if i_epoch < decrease_batch:
        epsilon = max(min_epsilon, epsilon - decrease_epsilon)
        #epsilon -= decrease_epsilon

# 儲存模型
output_dir = './dqn_model'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

seed_code = random.randint(100, 999)
file_name = f'q_learning_{NUM_EPISODES}_{seed_code}.ckpt'
save_path = os.path.join(output_dir, file_name)
agent.save(save_path)

print(f"Model saved to: {save_path}")