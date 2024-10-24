import gymnasium as gym
import numpy as np
import pygame
import random

from typing import Any

BLACK = (0, 0, 0)
RED = (225, 0, 0)
WHITE = (255, 255, 255)

class Player(pygame.sprite.Sprite):
    def __init__(self, x, y) -> None:
        super().__init__()
        self.image = pygame.Surface((40, 40))
        self.image.fill(WHITE)
        self.rect = self.image.get_rect()
        self.rect.center = (x, y)
        self.y_velocity = -10

    def update(self, gravity) -> None:
        self._apply_gravity(gravity)
        self.rect.y += self.y_velocity
    
    def jump(self) -> None:
        self.y_velocity = -15

    def _apply_gravity(self, gravity) -> None:
        self.y_velocity += gravity


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

        width  : 畫面寬度
        height : 畫面高度
        gravity: 重力，每一個 step 與玩家的速度相減
        pipe_hole_size: 上下管道之間洞口的大小
        roll_speed: 畫面整體的移動速度
        obstacles_spacing: 兩個管道之間的水平間距
        mode   : 模式，0 為純數字表示玩家和管道位置、速度， 1 為畫面截圖
    """
    def __init__(self, width, height, gravity, pipe_hole_size, roll_speed, obstacles_spacing, mode) -> None:
        self.game_settings = {
            'width': width,
            'height': height,
            'gravity': gravity,
            'pipe_hole_size': pipe_hole_size,
            'obstacle_spacing': obstacles_spacing,
            'roll_speed': roll_speed,
            'mode': mode
        }

        pygame.init()
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Jumping Game")

        # 創建角色
        player = Player(self.game_settings['width'] // 2 - 200, self.game_settings['height'] // 2)

        # 創建障礙物群組
        self.pipes = pygame.sprite.Group()

        # 生成障礙物
        self._new_pipes(self.game_settings['width'])

        # 創建所有遊戲物件的群組
        player_sprite = pygame.sprite.Group() 
        player_sprite.add(player)

        self.screen = screen
        self.player = player
        self.player_sprite = player_sprite
        self.next_pipe = self._next_pipe()

        self.scoreFont = pygame.font.Font(None, 36)
        self.total_distance = 0
        self.move_distance = 0
        self.game_over = False
        self.score = 0

    def reset(self) -> tuple[np.ndarray, dict[str, Any]]:
        self.game_over = False
        self.score = 0
        
        observation = self._get_observation()
        info = self._get_info()

        return observation, info
    
    def step(self, action: Any) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
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
        info = self._get_info()

        return observation, reward, truncated, info
    
    def quit(self) -> None:
        pygame.quit()
    
    def _get_observation(self) -> np.ndarray:
        observation = [self.player.rect.center[1], self.player.y_velocity]
        pipe_observation = [x for i, pipe in enumerate(self.pipes) 
                                if i % 2 == 0 and pipe.rect.x > self.player.rect.center[0] 
                                for x in (pipe.rect.x, pipe.spacing_y)]
        pipe_observation.extend([-1, -1] * ((6 - len(pipe_observation)) // 2))
        observation.extend(pipe_observation)

        return np.array(observation)
    
    def _calculate_reward(self, action) -> float:
        if self.game_over:
            return -10.0
        
        if self.over_pipe:
            return 1.0
  
        return 0.1
    
    def _get_info(self) -> dict[str, Any]:
        return {
            'score': self.score
        }
    
    def _game_step(self, action):
        if action == 1:
            self.player.jump()
        
        self.player.update(self.game_settings['gravity'])
        self.total_distance += self.game_settings['roll_speed']
        self.move_distance += self.game_settings['roll_speed']

        self._update_pipes()
        self.game_over = self._check_player_game_over()

        # 如果玩家越過前方管道則分數加一，並尋找下一個管道
        if self.next_pipe.rect.x < self.player.rect.center[0]:
            self.next_pipe = self._next_pipe()
            self.over_pipe = True
            self.score += 1

        # 繪製畫面
        self.screen.fill(BLACK)
        self.player_sprite.draw(self.screen)
        self.pipes.draw(self.screen)
        score_text = self.scoreFont.render("Score: " + str(self.score), True, WHITE)
        self.screen.blit(score_text, (10, 10))  # 在左上角位置繪製分數

        # 更新畫面
        pygame.display.flip()

    def _next_pipe(self) -> Pipe:
        return next(pipe for pipe in self.pipes if pipe.rect.x > self.player.rect.center[0])
        
    def _check_player_game_over(self) -> bool:
        # 角色如果接觸障礙物，遊戲結束回傳 True
        if self._check_pipe_collision():
            return True
        
        # 角色如果接觸窗口頂部或底部，遊戲結束回傳 True
        if self.player.rect.bottom >= self.game_settings['height'] or self.player.rect.top <= 0:  
            return True
    
        return False

    def _check_pipe_collision(self) -> bool:
        return len(pygame.sprite.spritecollide(self.player, self.pipes, False)) > 0

    def _new_pipes(self, x) -> None:
        half_extents = self.game_settings['pipe_hole_size'] // 2
        height = self.game_settings['height']

        y = random.randint(half_extents, height - half_extents)
        top_width = height - y - half_extents
        bottom_width = y - half_extents
        pipe_top = Pipe(10, top_width, x, height - top_width, y, RED)
        pipe_bottom = Pipe(10, bottom_width, x, 0, y, WHITE)

        self.pipes.add(pipe_top)
        self.pipes.add(pipe_bottom)

    def _update_pipes(self) -> None:
        # 玩家超出螢幕過多，生成新障礙物
        if self.move_distance >= self.game_settings['obstacle_spacing']:
            self._new_pipes(self.game_settings['width'])
            self.move_distance = 0
        
        # 將障礙物往前移動
        for pipe in self.pipes:
            pipe.rect.x -= self.game_settings['roll_speed']
            
        # 删除超出窗口的障礙物
        self.pipes.remove(*[pipe for pipe in self.pipes if pipe.rect.x < 0])
