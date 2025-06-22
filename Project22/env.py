import numpy as np
import pygame
import gym
from gym import spaces
import math

class FootballEnv(gym.Env):
    def __init__(self):
        super(FootballEnv, self).__init__()
        
        # Пространство действий (4 направления)
        self.action_space = spaces.Discrete(4)  # 0:вверх, 1:вправо, 2:вниз, 3:влево
        
        # Пространство наблюдений (нормализованные координаты)
        self.observation_space = spaces.Box(
            low=np.array([-1]*18, dtype=np.float32),
            high=np.array([1]*18, dtype=np.float32),
            dtype=np.float32
        )
        
        
        pygame.init()
        self.screen = pygame.display.set_mode((600, 400))
        pygame.display.set_caption('Football RL - 6 Players')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 15)
        
        
        self.player_ids = {
            'a': 1,  
            'b': 0,  
            'c': 2,  
            'd': 4,  
            'e': 5,  
            'f': 3   
        }
        
        
        self.team1_pos = {
            'a': [550, 200],  #
            'c': [400, 150],  
            'f': [400, 250]   
        }
        self.team2_pos = {
            'b': [50, 200],   
            'd': [200, 150],  
            'e': [200, 250]   
        }
        
        
        self.ball_pos = [300, 200]
        self.DIRECT = {'0':[0,-5], '1':[5,0], '2':[0,5], '3':[-5,0]}
        self.player_radius = 15
        self.ball_radius = 11
        self.score = 0
        self.max_score = 400
        self.border = pygame.Rect(29, 29, 546, 346)  

    def reset(self):
        
        self.team1_pos = {'a':[550,200], 'c':[400,150], 'f':[400,250]}
        self.team2_pos = {'b':[50,200], 'd':[200,150], 'e':[200,250]}
        self.ball_pos = [300, 200]
        self.score = 0
        return self._get_obs('a')  

    def _get_obs(self, player_name):
        
        obs = []
        
        obs.extend([self.ball_pos[0]/600, self.ball_pos[1]/400])
        
        if player_name in ['a', 'c', 'f']:  # Команда 1
           
            for name, pos in self.team1_pos.items():
                if name != player_name:
                    obs.extend([pos[0]/600, pos[1]/400])
           
            for pos in self.team2_pos.values():
                obs.extend([pos[0]/600, pos[1]/400])
        else: 
            
            for name, pos in self.team2_pos.items():
                if name != player_name:
                    obs.extend([pos[0]/600, pos[1]/400])
            
            for pos in self.team1_pos.values():
                obs.extend([pos[0]/600, pos[1]/400])
                
        return np.array(obs, dtype=np.float32)

    def step(self, action, player_name):
        
        done = False
        reward = 0.2
        ball_point = pygame.Vector2(self.ball_pos[0], self.ball_pos[1])
        
        
        if player_name in ['a', 'c', 'f']:  
            team_pos = self.team1_pos
            opponent_pos = self.team2_pos
            goal_x = 25  
            forward_dir = -1  
        else:  
            team_pos = self.team2_pos
            opponent_pos = self.team1_pos
            goal_x = 575
            forward_dir = 1
        
        
        new_pos = [
            team_pos[player_name][0] + self.DIRECT[str(action)][0],
            team_pos[player_name][1] + self.DIRECT[str(action)][1]
        ]
        
        
        collision = False
        
        
        for name, pos in team_pos.items():
            if name != player_name and self._circles_collide(new_pos, self.player_radius, pos, self.player_radius):
                collision = True
                break
        
        
        if not collision:
            for pos in opponent_pos.values():
                if self._circles_collide(new_pos, self.player_radius, pos, self.player_radius):
                    collision = True
                    break
        
        
        if not collision:
            team_pos[player_name] = new_pos
        
        
        if not self.border.collidepoint(team_pos[player_name]):
            done = True
            reward -= 30
        
        
        if self._circles_collide(team_pos[player_name], self.player_radius, self.ball_pos, self.ball_radius):
            reward += 1
            self.ball_pos[0] += self.DIRECT[str(action)][0]
            self.ball_pos[1] += self.DIRECT[str(action)][1]
            
            
            if (forward_dir == -1 and action == 3) or (forward_dir == 1 and action == 1):
                reward += 0.4
            
            
            if (forward_dir == -1 and self.ball_pos[0] < 30) or (forward_dir == 1 and self.ball_pos[0] > 570):
                if 120 < self.ball_pos[1] < 280: 
                    reward += 20
                    done = True
        
        
        dist_to_ball = ball_point.distance_to(pygame.Vector2(team_pos[player_name][0], team_pos[player_name][1]))
        if dist_to_ball < 50:
            reward += 0.5
        
        
        if (forward_dir == -1 and team_pos[player_name][0] < 300) or (forward_dir == 1 and team_pos[player_name][0] > 300):
            reward += 0.3
        
        self.score += 1
        if self.score > self.max_score:
            done = True
        
        return self._get_obs(player_name), reward, done, {"score": self.score}

    def render(self, mode='human'):
        
        pygame.event.pump()
        self.screen.fill((50, 200, 50))  
        
        
        field_width, field_height = 550, 350
        field_x, field_y = 25, 25
        
        
        pygame.draw.rect(self.screen, (255,255,255), (field_x, field_y, field_width, field_height), 3)
        pygame.draw.line(self.screen, (255,255,255), (field_x+field_width//2, field_y), (field_x+field_width//2, field_y+field_height), 2)
        pygame.draw.circle(self.screen, (255,255,255), (field_x+field_width//2, field_y+field_height//2), 50, 2)
        
        
        gate_h = 160
        gate_w = 20
        pygame.draw.rect(self.screen, (255,255,255), (field_x-gate_w, field_y+field_height//2-gate_h//2, gate_w, gate_h), 2)
        pygame.draw.rect(self.screen, (255,255,255), (field_x+field_width, field_y+field_height//2-gate_h//2, gate_w, gate_h), 2)
        
        
        for name, pos in self.team1_pos.items():
            color = (255,0,0) if name == 'a' else (200,0,0) if name == 'c' else (150,0,0)
            pygame.draw.circle(self.screen, color, pos, self.player_radius)
            self.screen.blit(self.font.render(name.upper(), True, (0,0,0)), (pos[0]-10, pos[1]-10))
        
        
        for name, pos in self.team2_pos.items():
            color = (0,0,255) if name == 'b' else (0,0,200) if name == 'd' else (0,0,150)
            pygame.draw.circle(self.screen, color, pos, self.player_radius)
            self.screen.blit(self.font.render(name.upper(), True, (0,0,0)), (pos[0]-10, pos[1]-10))
        
        
        pygame.draw.circle(self.screen, (255,255,255), self.ball_pos, self.ball_radius)
        
        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        pygame.quit()

    def _circles_collide(self, pos1, r1, pos2, r2):
        
        distance = math.hypot(pos2[0]-pos1[0], pos2[1]-pos1[1])
        return distance <= (r1 + r2)