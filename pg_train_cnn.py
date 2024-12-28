import cv2
import pygame
import matplotlib.pyplot as plt
import numpy as np
import time

from typing import Literal

from envs import FlappyBirdEnv
from agents import PolicyGradientAgent

NUM_BATCHES = 10000
EPISODE_PER_BATCH = 1

# initialize Environment and Agent
env = FlappyBirdEnv('rgb', True)

agent = PolicyGradientAgent(env.get_observation_shape(), 2, lr=0.0001)
agent.network.train()
agent.load('rgb_pg/new_pg_3600.ckpt')
print(env.get_observation_shape())

temperature = 1
decrease_batch = NUM_BATCHES * 0.6
decrease_temperature = (temperature - 1.0) / decrease_batch

batch_rewards = []
batch_scores = []

clock = pygame.time.Clock()
for i_batch in range(NUM_BATCHES):
    state, _ = env.reset() # reset environment to initial state for each episode

    episode_total_reward = 0
    episode_total_score = 0

    for i_episode in range(EPISODE_PER_BATCH):
        while True:
            action, log_prob = agent.sample(state)

            state, reward, truncated, info = env.step(action)

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        agent.save(f'./rgb_pg/new_pg_{3600 + i_batch + 1}.ckpt')

                        plt.figure(figsize=(15, 10))
                        plt.subplot(2, 1, 1)
                        plt.title('Train Reward')
                        plt.plot(batch_rewards)

                        plt.subplot(2, 1, 2)
                        plt.title('Train Score')
                        plt.plot(batch_scores)

                        plt.savefig(f'./rgb_pg/new_train_reward_p{3600 + i_batch + 1}.png')

            agent.store_transition(reward, log_prob)

            if truncated:
                state, _ = env.reset()
                episode_total_reward += info['total_reward']
                episode_total_score += info['score']
                # print(info['history'])
                break
            
            # clock.tick(120)
        agent.store_episode()

    agent.learn()

    batch_rewards.append(episode_total_reward)
    batch_scores.append(episode_total_score)

    if (i_batch + 1) % 20 == 0:
        # agent.scheduler.step()
        print(f"{i_batch + 1}/{NUM_BATCHES}, Lr: {agent.optimizer.param_groups[0]['lr']: .6f}, Temperature: {temperature: 4.2f}, Final Reward: {episode_total_reward: 4.2f}, Final Score: {episode_total_score}")

    if (i_batch + 1) % 200 == 0:
        agent.save(f'./rgb_pg/new_pg_{3600 + i_batch + 1}.ckpt')

        plt.figure(figsize=(15, 10))
        plt.subplot(2, 1, 1)
        plt.title('Train Reward')
        plt.plot(batch_rewards)

        plt.subplot(2, 1, 2)
        plt.title('Train Score')
        plt.plot(batch_scores)

        plt.savefig(f'./rgb_pg/new_train_reward_p{3600 + i_batch + 1}.png')

    # gradually decrease temperture
    if i_batch < decrease_batch:
        temperature -= decrease_temperature

agent.save('./policy gradient.cpkt')
env.close()