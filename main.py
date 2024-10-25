import pygame

from envs import FlappyBirdEnv
from agents import DQNAgent

env = FlappyBirdEnv(800, 600, 1, 240, 5, 200, 0)
agent = DQNAgent()

state, info = env.reset()

running = True
clock = pygame.time.Clock()
while running:
    # 事件處理
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        # elif event.type == pygame.KEYDOWN:
        #     if event.key == pygame.K_SPACE:
        #         action = 1 

    if not running:
        break
    
    action = agent.sample(state, epsilon=0.1)

    next_state, reward, truncated, info = env.step(action)

    agent.store_transition(state, action, reward, next_state)

    agent.learn()
    state = next_state

    running = not truncated
    clock.tick(60)

env.quit()