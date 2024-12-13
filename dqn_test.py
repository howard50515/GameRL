import pygame

from envs import FlappyBirdEnv
from agents import DQNAgent

EPOCHS = 1000
STEP_PER_EPOCH = 1000
CHECKPOINT_PATH = './dqn_model\q_learning_600_225.ckpt'

# initialize Environment and Agent
env = FlappyBirdEnv('numeric', True)
'''The FlappyBird Environment.

        mode         : observation 模式，0 回傳純數字表示角色和管道位置、速度， 1 回傳畫面截圖
        screen_debug : 是否顯示遊戲畫面，當 mode 為 1 時，強制顯示遊戲畫面。'''
#env = FlappyBirdEnv(0, False)

agent = DQNAgent(env.get_observation_shape(), 2, lr=0.001)
print(env.get_observation_shape())
agent.load(CHECKPOINT_PATH)
agent.eval_net.eval()
agent.target_net.eval()

state, _ = env.reset()
clock = pygame.time.Clock()

while True:
    action = agent.sample(state, 0)
    state, reward, truncated, info = env.step(action)
    for event in pygame.event.get():
            pass
    if truncated:
        print(f"Game Over. Score: {info['score']}")
        break

    # 使迴圈以 60FPS 運行，以便於觀看遊戲過程
    # clock.tick(60)