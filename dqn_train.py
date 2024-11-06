import pygame

from envs import FlappyBirdEnv
from agents import DQNAgent

NUM_EPISODES = 200
STEP_PER_EPISODE = 1000

# initialize Environment and Agent
env = FlappyBirdEnv(1, False)

agent = DQNAgent(env.get_observation_shape()[0], 2)
agent.eval_net.train()
agent.target_net.train()

epsilon = 1.
decrease_batch = NUM_EPISODES * 0.6
decrease_epsilon = 0.98 / decrease_batch

for i_epoch in range(NUM_EPISODES):
    state, _ = env.reset() # reset environment to initial state for each episode

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
    
    if (i_epoch + 1) % 5 == 0:
        print(f"{i_epoch + 1}/{NUM_EPISODES}, Lr: {agent.optimizer.param_groups[0]['lr']: .6f}, Epsilon: {epsilon: 4.4f}, Final Reward: {epoch_total_reward: 4.2f}")
    
    # graually descrease epsilon
    if i_epoch < decrease_batch:
        epsilon -= decrease_epsilon

agent.save('./q learning.cpkt')