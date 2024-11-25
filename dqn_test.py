import pygame

from envs import FlappyBirdEnv
from agents import DQNAgent

EPOCHS = 200
STEP_PER_EPOCH = 1000
CHECKPOINT_PATH = './dqn_model/q_learning_100_258.ckpt'

# initialize Environment and Agent
env = FlappyBirdEnv(0, True)

agent = DQNAgent(env.get_observation_shape(), 2)
# agent.load(CHECKPOINT_PATH)
agent.eval_net.eval()
agent.target_net.eval()

state, _ = env.reset()
clock = pygame.time.Clock()
while True:
    action = agent.sample(state, 0)
    state, reward, truncated, info = env.step(action)

    if truncated:
        print(f"Game Over. Score: {info['score']}")
        break

    # 使迴圈以 60FPS 運行，以便於觀看遊戲過程
    clock.tick(60)

