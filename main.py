import pygame

from envs import FlappyBirdEnv

env = FlappyBirdEnv(800, 600, 1, 240, 5, 200, 0)

running = True
clock = pygame.time.Clock()
while running:
    action = 0
    # 事件處理
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                action = 1 

    if not running:
        break

    observation, reward, truncated, info = env.step(action)
    print("observation", observation)
    print("reward", reward)
    print("info", info)


    running = not truncated
    clock.tick(60)

env.quit()