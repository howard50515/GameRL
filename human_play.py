import pygame

from envs import FlappyBirdEnv

env = FlappyBirdEnv(0, True)
state, info = env.reset()

running = True
clock = pygame.time.Clock()
while running:
    # 事件處理
    action = 0
    for event in pygame.event.get():
        if event.type == pygame.QUIT: 
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                action = 1 

    if not running:
        break

    next_state, reward, truncated, info = env.step(action)
    state = next_state  
 
    # print("state", state, "reward", reward, "truncated", truncated, "info", info)
 
    running = not truncated
    clock.tick(60)
