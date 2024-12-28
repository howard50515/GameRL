import cv2
import gymnasium as gym
import numpy as np
import pygame
import random
import time

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
        self.spacing_image = pygame.Surface((width, PIPE_VERTICAL_SPACING))
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.spacing_y = spacing_y
        self.image.fill(color)
        self.spacing_image.fill(BLUE)

    def update(self) -> None:
        pass

    def blit_spacing(self, screen: pygame.Surface):
        screen.blit(self.spacing_image, (self.rect.x, self.spacing_y - (PIPE_VERTICAL_SPACING // 2)))


class FlappyBirdEnv(gym.Env):
    """
        The FlappyBird Environment.

        mode         : observation 模式，0 回傳純數字表示角色和管道位置、速度， 1 回傳畫面截圖
        screen_debug : 是否顯示遊戲畫面，當 mode 為 1 時，強制顯示遊戲畫面。
    """
    def __init__(self, 
                 mode: Literal['numeric', 'rgb', 'gray'], 
                 screen_debug: bool) -> None:
        if not isinstance(mode, str):
            raise ValueError('mode should be str, with one of [\'numeric\', \'rgb\', \'gray\'].')
        
        if isinstance(mode, str) and mode != 'numeric' and mode != 'rgb' and mode != 'gray':
            raise ValueError(f'invalid mode \'{mode}\'. mode should be one of [\'numeric\', \'rgb\', \'gray\'].')
        
        if not isinstance(screen_debug, bool):
            raise ValueError('screen_debug should be boolean.')
        
        self.mode = mode
        self.screen_debug = screen_debug

        pygame.init()
        self.screen = None
        if self._is_img_mode() or screen_debug:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Flappy Bird Game")
            self.scoreFont = pygame.font.Font(None, 36)

        self.reset()

    def reset(self) -> tuple[np.ndarray, dict[str, Any]]:
        self._init_sprites()

        self.history = []
        self.survived_tick = 0
        self.total_distance = 0
        self.total_reward = 0
        self.game_over = False
        self.over_pipe = False
        self.pipe_collide = False
        self.score = 0
        self.keep = 0
        
        self._screen_image = None
        self._image_tick = 0

        if self._is_img_mode():
            self.img_list = []

        observation = self._get_observation()
        info = self._get_info()

        return observation, info
    
    def step(self, 
             action: Literal[0, 1]) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        if self.game_over:
            raise ValueError('game is over. use \'reset()\' to restart the game.')
        
        if not isinstance(action, int):
            raise ValueError(f'action is expected a integer, get {type(action)} instead.')
        
        if action != 0 and action != 1:
            raise ValueError('action should be 0 or 1.')

        self.over_pipe = False
        self.last_y_velocity = self.player.y_velocity

        self._game_step(action)
        observation = self._get_observation()
        reward = self._calculate_reward(action)
        truncated = self.game_over

        if action == 1:
            self.history.append(f'jump at {self.survived_tick}')
        self.total_reward += reward
        info = self._get_info()

        # if self.mode == 'rgb':
        #     self.img_list.append(self._get_rgb_screen_image().transpose([1, 0, 2]))
        # elif self.mode == 'gray':
        #     self.img_list.append(self._get_gray_scale_screen_image().transpose([1, 0]))
            
        self.survived_tick += 1

        return observation, reward, truncated, info
    
    def get_observation_shape(self) -> tuple[int, ...]:
        return self._get_observation().shape
    
    def save_game_video(self, 
                       file_name: str, 
                       fps: float) -> None:
        if not self._is_img_mode():
            print('Warning: save_game_video is only avaiable in mode with one of [\'rgb\', \'gray\']')
            print('Can not generate game video.')
            return
        
        if self.mode == 'rgb':
            out = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (SCREEN_WIDTH, SCREEN_HEIGHT))
        else:
            out = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (SCREEN_WIDTH, SCREEN_HEIGHT), False)

        for img in self.img_list:
            out.write(img)
        out.release()
    
    def close(self) -> None:
        pygame.quit()
        super().close()
    
    def _get_observation(self) -> np.ndarray:
        if self.mode == 'numeric':
            observation = [self.player.rect.center[1] / SCREEN_HEIGHT, self.player.y_velocity]
            pipe_observation = [x for i, pipe in enumerate(self.pipes) 
                                    if i % 2 == 0 and self._pipe_in_front(pipe)
                                    for x in ((pipe.rect.x - self.player.rect.center[0]) / SCREEN_WIDTH, pipe.spacing_y / SCREEN_HEIGHT)]
            observation.extend(pipe_observation)
            return np.array(observation)
        
        if self.mode == 'rgb':
            resized_image = cv2.resize(self._get_rgb_screen_image(), 
                (OBSERVATION_HEIGHT, OBSERVATION_WIDTH), interpolation=cv2.INTER_AREA)
        else:
            resized_image = cv2.resize(self._get_gray_scale_screen_image(), 
                (OBSERVATION_HEIGHT, OBSERVATION_WIDTH), interpolation=cv2.INTER_AREA)[:, :, np.newaxis]
            
        return resized_image.transpose([2, 0, 1]).astype('float32') / 255.0
    
    def _calculate_reward(self, 
                          action: Literal[0, 1]) -> float:
        if self.game_over:
            return -3.0
        
        if self.over_pipe:
            return 1.0
        
        half_extents = PIPE_VERTICAL_SPACING // 2

        pipe_hole_y = self.next_pipe.spacing_y
        player_y = self.player.rect.center[1]

        hole_distance_ratio = (abs(pipe_hole_y - player_y) / half_extents) - 1
        
        base_reward = 0.1
        if hole_distance_ratio <= -0.75:
            return base_reward
        elif hole_distance_ratio <= 0.25:
            return -hole_distance_ratio * base_reward
        elif hole_distance_ratio <= 0.0:
            return 0
        else:
            return -min(hole_distance_ratio, 1) * base_reward

        # return -min(1, (abs(pipe_hole_y - player_y) / half_extents) - 1) * 0.1
    
    def _get_info(self) -> dict[str, Any]:
        return {
            'score': self.score,
            'total_reward': self.total_reward,
            'total_distance': self.total_distance,
            'history' : self.history
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

        if self._is_img_mode() or self.screen_debug:
            # 繪製畫面
            self.screen.fill(BLACK)

            self.player.draw(self.screen)
            self.pipes.draw(self.screen)
            score_text = self.scoreFont.render("Score: " + str(self.score), True, WHITE)
            self.screen.blit(score_text, (10, 10))  # 在左上角位置繪製分數

            if self.mode == 'gray':
                gray_scale_screen =  np.repeat(
                    self._get_gray_scale_screen_image()[:, :, np.newaxis], 3, axis=2)
                pygame.surfarray.blit_array(self.screen, gray_scale_screen)

            # 更新畫面
            pygame.display.flip()

    def _init_sprites(self) -> None:
        # 創建角色
        self.player = Player(PLAYER_INIT_X, PLAYER_INIT_Y)

        # 創建管道群組，並生成管道
        self.pipes: pygame.sprite.Group[Pipe] = pygame.sprite.Group()
        for i in range(3):
            self._new_pipes((SCREEN_WIDTH // 2) + i * PIPE_HORIZONTAL_SPACING)
        self.next_pipe = self._next_pipe()

    def _pipe_in_front(self, 
                       pipe: Pipe) -> bool:
        return (pipe.rect.x + PIPE_WIDTH) >= self.player.rect.x

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

    def _get_gray_scale_screen_image(self) -> np.ndarray:
        if self.mode == 'numeric':
            return None
    
        return cv2.cvtColor(self._get_rgb_screen_image(), cv2.COLOR_RGB2GRAY)

    def _get_rgb_screen_image(self):
        if self.mode == 'numeric':
            return None

        # 快取截圖，如果此 tick 已計算過截圖，則直接回傳快取
        # 在 _game_step() 前調用，可能會有問題
        if self._image_tick == self.survived_tick and self._screen_image is not None:
            return self._screen_image
        
        image = pygame.surfarray.array3d(self.screen)
        self._image_tick = self.survived_tick
        self._screen_image = image

        return image
    
    def _is_img_mode(self) -> bool:
        return self.mode == 'rgb' or self.mode == 'gray'
