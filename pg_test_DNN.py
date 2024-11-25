import pygame

from envs import FlappyBirdEnv
from agents import PolicyGradientAgent

EPOCHS = 200
STEP_PER_EPOCH = 1000
CHECKPOINT_PATH = './policy gradient.cpkt'

# initialize Environment and Agent
env = FlappyBirdEnv(0, True)

agent = PolicyGradientAgent(env.get_observation_shape(), 2)
agent.load(CHECKPOINT_PATH)
agent.network.eval()

for i in range(10):
    state, _ = env.reset()
    clock = pygame.time.Clock()
    while True:
        action, log_prob = agent.sample(state)
        state, reward, truncated, info = env.step(action)
        # for s in state:
        #     print(f'{s: .4f}', end=' ')
        print(f'reward: {reward}')

        if truncated:
            print(f"Game Over. Score: {info['score']}")
            break

        # event = pygame.event.wait()
        # if event.type == pygame.QUIT:
        #     running = False
        # elif event.type == pygame.KEYDOWN:
        #     if event.key == pygame.K_SPACE:
        #         pass

        # 使主迴圈以 60FPS 運行，以便於觀看遊戲過程
        clock.tick(60)

env.close()

