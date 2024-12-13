import pygame

from typing import Literal

from envs import FlappyBirdEnv
from agents import PolicyGradientAgent

NUM_BATCHES = 5000
EPISODE_PER_BATCH = 2
def my_calculate_rewards(self: FlappyBirdEnv, action: Literal[0, 1]) -> float:
    return 0.1

class MyFlappyBirdEnv(FlappyBirdEnv):
    _calculate_reward = my_calculate_rewards
# initialize Environment and Agent
env = FlappyBirdEnv('numeric', True)

agent = PolicyGradientAgent(env.get_observation_shape(), 2, lr=0.001)
agent.network.train()

temperature = 1
decrease_batch = NUM_BATCHES * 0.6
decrease_temperature = (temperature - 1.0) / decrease_batch

clock = pygame.time.Clock()
for i_bacth in range(NUM_BATCHES):
    state, _ = env.reset() # reset environment to initial state for each episode

    episode_total_reward = 0
    episode_total_score = 0
    for i_episode in range(EPISODE_PER_BATCH):
        while True:
            action, log_prob = agent.sample(state, temperature)

            state, reward, truncated, info = env.step(action)

            agent.store_transition(reward, log_prob)
            for event in pygame.event.get():
                pass
            if truncated:
                state, _ = env.reset()
                episode_total_reward += info['total_reward']
                episode_total_score += info['score']
                break
        
        agent.store_episode()
    agent.learn()

    if (i_bacth + 1) % 50 == 0:
        agent.scheduler.step()
        print(f"{i_bacth + 1}/{NUM_BATCHES}, Lr: {agent.optimizer.param_groups[0]['lr']: .6f}, Temperature: {temperature: 4.2f}, Final Reward: {episode_total_reward: 4.2f}, Final Score: {episode_total_score}")

    # gradually decrease temperture
    if i_bacth < decrease_batch:
        temperature -= decrease_temperature

agent.save('./policy gradient.cpkt')