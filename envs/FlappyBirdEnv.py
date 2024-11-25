import cv2
import gymnasium as gym
import numpy as np
import pygame
import random

from typing import Any, Literal, Callable

from .FlappyBirdEnvConstants import *

class Player(pygame.sprite.Sprite):
    def __init__(self, x, y) -> None:
        super().__init__()
        self.image = pygame.Surface((PLAYER_WIDTH, PLAYER_HEIGHT))
        self.image.fill(WHITE)
        self.rect = self.image.get_rect()
        self.rect.center = (x, y)
        self.y_velocity = PLAYER_INIT_VELOCITY

    def update(self) -> None:
        self._apply_gravity()
        self.rect.y -= self.y_velocity

    def draw(self, screen: pygame.Surface) -> None:
        screen.blit(self.image, self.rect)
    
    def jump(self) -> None:
        self.y_velocity = PLAYER_JUMP_VELOCITY

    def _apply_gravity(self) -> None:
        self.y_velocity -= GRAVITY
        self.y_velocity = min(PLAYER_MAX_VELOCITY, self.y_velocity)
        self.y_velocity = max(PLAYER_MIN_VELOCITY, self.y_velocity)


class Pipe(pygame.sprite.Sprite):
    def __init__(self, width, height, x, y, spacing_y, color) -> None:
        super().__init__()
        self.image = pygame.Surface((width, height))
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.spacing_y = spacing_y
        self.image.fill(color)

    def update(self) -> None:
        pass


class FlappyBirdEnv(gym.Env):
    """
        The FlappyBird Environment.

        mode         : observation 模式，0 回傳純數字表示角色和管道位置、速度， 1 回傳畫面截圖
        screen_debug : 是否顯示遊戲畫面，當 mode 為 1 時，強制顯示遊戲畫面。
    """
    def __init__(self, 
                 mode: Literal[0, 1], 
                 screen_debug: bool,) -> None:
        if not isinstance(mode, int):
            raise ValueError('mode should be a integer.')
        
        if mode != 0 and mode != 1:
            raise ValueError('mode should be 0 or 1.')
        
        if not isinstance(screen_debug, bool):
            raise ValueError('screen_debug should be boolean.')
        
        self.mode = mode
        self.screen_debug = screen_debug

        pygame.init()
        self.screen = None
        if mode == 1 or screen_debug:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Jumping Game")
            self.scoreFont = pygame.font.Font(None, 36)

        self._init_sprites()

        self.survived_tick = 0
        self.total_distance = 0
        self.total_reward = 0
        self.game_over = False
        self.over_pipe = False
        self.score = 0
        self.pipes_cleared = 0  # 每輪成功跳過的管道數


    def reset(self) -> tuple[np.ndarray, dict[str, Any]]:
        self._init_sprites()

        self.survived_tick = 0
        self.total_distance = 0
        self.total_reward = 0
        self.game_over = False
        self.over_pipe = False
        self.pipe_collide = False
        self.score = 0
        self.pipes_cleared = 0  # 重置成功跳過管道的次數

        observation = self._get_observation()
        info = self._get_info()

        return observation, info
    
    def step(self, 
             action: Literal[0, 1]) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        if self.game_over:
            raise ValueError('game is over. use \'reset()\' to restart the game.')
        
        if not isinstance(action, int):
            raise ValueError('action should be a integer.')
        
        if action != 0 and action != 1:
            raise ValueError('action should be 0 or 1.')

        self.over_pipe = False
        self._game_step(action)
        observation = self._get_observation()
        reward = self._calculate_reward(action)
        truncated = self.game_over

        self.survived_tick += 1
        self.total_reward += reward
        info = self._get_info()
        
        return observation, reward, truncated, info
    
    def get_observation_shape(self) -> tuple[int, ...]:
        return self._get_observation().shape
    
    def close(self) -> None:
        pygame.quit()
        super().close()
    
    def _get_observation(self) -> np.ndarray:
        if self.mode == 0:
            observation = [self.player.rect.center[1] / SCREEN_HEIGHT, self.player.y_velocity]
            pipe_observation = [x for i, pipe in enumerate(self.pipes) 
                                    if i % 2 == 0 and self._pipe_in_front(pipe)
                                    for x in ((pipe.rect.x - self.player.rect.center[0]) / SCREEN_WIDTH, pipe.spacing_y / SCREEN_HEIGHT)]
            # pipe_observation.extend([-1, -1] * ((6 - len(pipe_observation)) // 2))
            observation.extend(pipe_observation)
            return np.array(observation)
        
        # return pygame.surfarray.array3d(self.screen)
        return self._to_gray_scale(pygame.surfarray.array3d(self.screen)).reshape((1, SCREEN_WIDTH, SCREEN_HEIGHT)).astype('float32') / 255.0
    
    def _calculate_reward(self, action: Literal[0, 1]) -> float:
        # 如果遊戲結束，根據碰撞情況給予懲罰
        if self.game_over:
            return -5.0 if self.pipe_collide else -10.0

        # 通過管道，給予較大的正向獎勵
        if self.over_pipe:
            return 5.0

        # 獲取角色與洞口中心的垂直距離
        pipe_center = self.next_pipe.spacing_y
        bird_center = self.player.rect.center[1]
        distance_to_center = abs(pipe_center - bird_center)

        # 根據距離給予獎勵或懲罰
        if distance_to_center < (PIPE_VERTICAL_SPACING // 4):
            reward = 0.5  # 靠近中心，獎勵
        else:
            reward = -0.1  # 偏離中心，懲罰

        # 根據動作與垂直位置判斷
        if action == 1:  # 跳躍
            if bird_center < pipe_center + (PIPE_VERTICAL_SPACING // 2):  # 高於洞口下緣
                reward -= 0.5  # 懲罰無效跳躍
            else:
                reward += 0.1  # 合理跳躍，微小獎勵
        else:  # 不跳躍
            if bird_center > pipe_center - (PIPE_VERTICAL_SPACING // 2):  # 低於洞口上緣
                reward += 0.1  # 合理下降，獎勵
            else:
                reward -= 0.1  # 不合理不跳，懲罰

        # 每個時間步生存，增加固定獎勵
        reward += 0.1

        return reward

    
    def _get_info(self) -> dict[str, Any]:
        return {
            'score': self.score,
            'total_reward': self.total_reward,
            'total_distance': self.total_distance,
            'pipes_cleared': self.pipes_cleared  # 新增管道成功次數
        }

    
    def _game_step(self, 
                   action: Literal[0, 1]) -> None:
        if action == 1:
            self.player.jump()
        
        self.player.update()
        self._update_pipes()
        self.game_over = self._check_player_game_over()
        self.total_distance += SCREEN_SPEED

        # 如果角色越過前方管道則分數加一，並尋找下一個管道
        if not self._pipe_in_front(self.next_pipe):
            self.next_pipe = self._next_pipe()
            # 創建新管道，以保持角色面前永遠有三個管道
            self._new_pipes(self.pipes.sprites()[-1].rect.x + PIPE_HORIZONTAL_SPACING)
            self.over_pipe = True
            self.score += 1
            self.pipes_cleared += 1  # 記錄成功跳過的管道數

        if self.mode == 1 or self.screen_debug:
            # 繪製畫面
            self.screen.fill(BLACK)
            self.player.draw(self.screen)
            self.pipes.draw(self.screen)
            score_text = self.scoreFont.render("Score: " + str(self.score), True, WHITE)
            self.screen.blit(score_text, (10, 10))  # 在左上角位置繪製分數
            # pygame.surfarray.blit_array(self.screen, self._to_gray_scale(pygame.surfarray.array3d(self.screen)))
            
            # 更新畫面
            pygame.display.flip()

    def _init_sprites(self) -> None:
        # 創建角色
        self.player = Player(PLAYER_INIT_X, PLAYER_INIT_Y)

        # 創建管道群組，並生成管道
        self.pipes = pygame.sprite.Group()
        for i in range(3):
            self._new_pipes((SCREEN_WIDTH // 2) + i * PIPE_HORIZONTAL_SPACING)
        self.next_pipe = self._next_pipe()

    def _pipe_in_front(self, 
                       pipe: Pipe) -> bool:
        return pipe.rect.x >= self.player.rect.center[0]

    def _next_pipe(self) -> Pipe:
        """
            尋找離角色最近且位於角色前方的管道
        """
        return next(pipe for pipe in self.pipes if self._pipe_in_front(pipe))
        
    def _check_player_game_over(self) -> bool:
        # 角色如果接觸障礙物，遊戲結束回傳 True
        if self._check_pipe_collision():
            self.pipe_collide = True
            return True
        
        # 角色如果接觸窗口頂部或底部，遊戲結束回傳 True
        if self.player.rect.bottom >= SCREEN_HEIGHT or self.player.rect.top <= 0:  
            return True
    
        return False

    def _check_pipe_collision(self) -> bool:
        return len(pygame.sprite.spritecollide(self.player, self.pipes, False)) > 0

    def _new_pipes(self, 
                   x: int) -> None:
        """
            創建新管道

            x : 新管道的 x 座標
        """
        half_extents = PIPE_VERTICAL_SPACING // 2

        y = random.randint(half_extents, SCREEN_HEIGHT - half_extents)
        bottom_width = y - half_extents
        top_width = SCREEN_HEIGHT - y - half_extents
        pipe_top = Pipe(PIPE_WIDTH, bottom_width, x, 0, y, RED)
        pipe_bottom = Pipe(PIPE_WIDTH, top_width, x, SCREEN_HEIGHT - top_width, y, RED)

        self.pipes.add(pipe_top)
        self.pipes.add(pipe_bottom)

    def _update_pipes(self) -> None:
        # 將障礙物往前移動
        for pipe in self.pipes:
            pipe.rect.x -= SCREEN_SPEED
            
        # 删除超出窗口的障礙物
        self.pipes.remove(*[pipe for pipe in self.pipes if pipe.rect.x < 0])

    def _to_gray_scale(self, arr) -> np.ndarray:
        # 0.2989 0.587 0.114
        # gray_scale = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        # np.round(np.dot(arr[...,:], [0.2989, 0.5870, 0.1140]))
        return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        # return np.dstack((gary_scale, gary_scale, gary_scale))
