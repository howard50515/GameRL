import pygame
import matplotlib.pyplot as plt
from envs import FlappyBirdEnv
from agents import DQNAgent

NUM_EPISODES = 1000
STEP_PER_EPISODE = 600

# initialize Environment and Agent
#env = FlappyBirdEnv(1, True)
env = FlappyBirdEnv('numeric', False)

agent = DQNAgent(env.get_observation_shape(), 2, lr=0.001)
print(env.get_observation_shape())
agent.eval_net.train()
agent.target_net.train()

epsilon = 0.5
min_epsilon = 0.0
decrease_batch = NUM_EPISODES * 0.5 
#decrease_epsilon = (epsilon - min_epsilon) / decrease_batch
decrease_epsilon = 0.5 / decrease_batch
rewards = []
scores = []
for i_epoch in range(NUM_EPISODES):
    state, _ = env.reset() # reset environment to initial state for each episode
    epoch_total_reward = 0
    print(i_epoch)
    for i_step in range(STEP_PER_EPISODE):
        for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        agent.save(f'./dqn{i_epoch + 1}.ckpt')

                        plt.figure(figsize=(15, 10))
                        plt.subplot(2, 1, 1)
                        plt.title('Train Reward')
                        plt.plot(rewards)

                        plt.subplot(2, 1, 2)
                        plt.title('Train Score')
                        plt.plot(scores)

                        plt.savefig(f'./dqn{i_epoch + 1}.png')
        action = agent.sample(state, epsilon=epsilon)

        next_state, reward, truncated, info = env.step(action)

        agent.store_transition(state, action, reward, next_state)

        agent.learn()
        
        state = next_state
        if truncated:
            state, _ = env.reset()
            rewards.append(info['total_reward'])
            scores.append(info['score'])
            epoch_total_reward += info['total_reward']
    
    #if (i_epoch + 1) % 5 == 0:
        #print(f"{i_epoch + 1}/{NUM_EPISODES}, Lr: {agent.optimizer.param_groups[0]['lr']: .6f}, Epsilon: {epsilon: 4.4f}, Final Reward: {epoch_total_reward: 4.2f}")
    if (i_epoch + 1) % 200 == 0:
        agent.save(f'./dqn{i_epoch + 1}.ckpt')

        plt.figure(figsize=(15, 10))
        plt.subplot(2, 1, 1)
        plt.title('Train Reward')
        plt.plot(rewards)

        plt.subplot(2, 1, 2)
        plt.title('Train Score')
        plt.plot(scores)

        plt.savefig(f'./dqn{i_epoch + 1}.png')
    
    # graually descrease epsilon
    if i_epoch < decrease_batch:
        epsilon = max(min_epsilon, epsilon - decrease_epsilon)
        #epsilon -= decrease_epsilon

agent.save('./q learning.cpkt')