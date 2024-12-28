import numpy as np
import pygame

from typing import Literal

from envs import FlappyBirdEnv
from agents import PolicyGradientAgent

CHECKPOINT_PATH = './new_pg_3762.ckpt'

# initialize Environment and Agent
env = FlappyBirdEnv('rgb', True)

agent = PolicyGradientAgent(env.get_observation_shape(), 2)
agent.load(CHECKPOINT_PATH)
agent.network.eval()

total_score = 0
num_test = 20
scores = []
for i in range(num_test):
    state, _ = env.reset()
    clock = pygame.time.Clock()
    while True:
        action, log_prob = agent.sample(state)
        state, reward, truncated, info = env.step(action)
        # print('velocity', env.player.y_velocity, 'reward', reward)

        for event in pygame.event.get():
            pass

        if truncated:
            print(f"Game Over. Score: {info['score']}")
            scores.append(info['score'])
            break

        # 使主迴圈以 120FPS 運行，以便於觀看遊戲過程
        # clock.tick(120)
    
    if (i + 1) % 10 == 0:
        print('mean', np.mean(scores))
        print('median', np.median(scores))
        print('std', np.std(scores))
        print('max', np.max(scores))
        print('min', np.min(scores))

    # env.save_game_video(f'./video{i+1}.mp4', 60)
env.close()

