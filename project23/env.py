import numpy as np
import pygame
import gym
from gym import spaces
import math




class FootballEnv(gym.Env):
    def __init__(self):
        super(FootballEnv, self).__init__()
        
        # Определение пространства действий
        self.action_space = spaces.Discrete(4)  # 0 = не прыгать, 1 = прыгать
        
        # Исправлено: теперь observation_space соответствует возвращаемому наблюдению
        self.observation_space = spaces.Box(
            low=np.array([-1, -1,-1,-1,-1,-1,-1], dtype=np.float32),  # Только x_dist и y_diff
            high=np.array([1, 1,1,1,1,1,1], dtype=np.float32),
            dtype=np.float32
        )
        
        # Инициализация игры
        pygame.init()
        self.screen = pygame.display.set_mode((600, 400))
        pygame.display.set_caption('Football RL')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 15)
        self.player_pos_a = [470,200]
        self.player_pos_b = [130,200]
        self.ball_pos = [300,200]
        self.DIRECT = {'0':[0,5],'1':[5,0],'2':[0,-5],'3':[-5,0]}
        self.score = 0
        self.border = pygame.Rect(29, 29, 546, 346)
        self.max_score = 400
        
        
    def reset(self):
        self.player_pos_a = [470,200]
        self.player_pos_b = [130,200]
        self.ball_pos = [300,200]
        self.score=0
        
        return self._get_obs(player=0)
    
    def _get_obs(self,player):
        
        ballpoint = pygame.Vector2(self.ball_pos[0],self.ball_pos[1])
        player_a = pygame.Vector2(self.player_pos_a[0],self.player_pos_a[1])
        player_b = pygame.Vector2(self.player_pos_b[0],self.player_pos_b[1])

        if player==1:
            return np.array([self.ball_pos[0]/600, self.ball_pos[1]/400,self.player_pos_a[0]/600
                             ,self.player_pos_a[1]/400,self.player_pos_b[0]/600,self.player_pos_b[1]/400,ballpoint.distance_to(player_a)/730,player_a.distance_to(player_b)/730], dtype=np.float32)
        else:
            return np.array([self.ball_pos[0]/600, self.ball_pos[1]/400,self.player_pos_b[0]/600
                             ,self.player_pos_b[1]/400,self.player_pos_a[0]/600,self.player_pos_a[1]/400,ballpoint.distance_to(player_b)/730,player_b.distance_to(player_a)/730], dtype=np.float32)
    def step(self, action,player):
        
        done = False
        reward = 0
        winer = '0'

        ballpoint = pygame.Vector2(self.ball_pos[0],self.ball_pos[1])
        
        if player:
            
            self.player_pos_a[0]+=self.DIRECT[str(action)][0]
            self.player_pos_a[1]+=self.DIRECT[str(action)][1]
            if circles_collide(self.player_pos_a,15,self.player_pos_b,15) or circles_collide(self.player_pos_a,15,self.ball_pos,11):
                if circles_collide(self.player_pos_a,15,self.ball_pos,11):
                    reward+=0.5
                    self.ball_pos[0]+=self.DIRECT[str(action)][0]
                    self.ball_pos[1]+=self.DIRECT[str(action)][1]
                    if action==3:
                        reward+=0.4

                    if self.ball_pos[0]<290:
                        reward+=0.2 

                        if 195<self.ball_pos[1]<355 :
                            reward+=0.3

                if circles_collide(self.player_pos_a,15,self.player_pos_b,15):
                    reward-=0.7
                self.player_pos_a[0]-=self.DIRECT[str(action)][0]
                self.player_pos_a[1]-=self.DIRECT[str(action)][1]
            if action==3:
                reward+=0.3

            if ballpoint.distance_to(pygame.Vector2(self.player_pos_a[0],self.player_pos_a[1])) < 50:
                reward+=0.5

            if self.player_pos_a[0]< 300:
                reward+=0.3

            if (not self.border.collidepoint((int(self.player_pos_a[0]), int(self.player_pos_a[1]))) or
                not self.border.collidepoint((int(self.ball_pos[0]), int(self.ball_pos[1])))):
                done = True
                reward -= 30

            

            if self.ball_pos[0]<290:
                reward+=0.2 

                if 195<self.ball_pos[1]<355:
                    reward+=0.3  

             
            if ballpoint.distance_to(pygame.Vector2(self.player_pos_a[0],self.player_pos_a[1]))>pygame.Vector2(self.ball_pos[0],self.ball_pos[1]).distance_to(pygame.Vector2(self.player_pos_a[0],self.player_pos_a[1])):
                reward+=1
            else:
                reward-=0.7

            if pygame.Rect(25 - 20, 25 + 160//2 - 160//2, 
                        20, 160).collidepoint(int(self.ball_pos[0]),int(self.ball_pos[1])):
                reward+=50
                done=True
                winer = 'a'
        else:
            self.player_pos_b[0]+=self.DIRECT[str(action)][0]
            self.player_pos_b[1]+=self.DIRECT[str(action)][1]

            if circles_collide(self.player_pos_a,15,self.player_pos_b,15)or circles_collide(self.player_pos_b,15,self.ball_pos,11):
                if circles_collide(self.player_pos_b,15,self.ball_pos,11):
                    self.ball_pos[0]+=self.DIRECT[str(action)][0]
                    self.ball_pos[1]+=self.DIRECT[str(action)][1]
                    reward+=0.5
                    if action ==1:
                        reward+=0.4

                    if self.ball_pos[0]>310:
                        reward+=0.2 

                        if 195<self.ball_pos[1]<355 :
                            reward+=0.3

                if circles_collide(self.player_pos_a,15,self.player_pos_b,15):
                    reward-=0.7
                self.player_pos_b[0]-=self.DIRECT[str(action)][0]
                self.player_pos_b[1]-=self.DIRECT[str(action)][1]

            if action==1:
                reward+=0.3
            
            if self.player_pos_b[0]>300:
                reward+=0.3

            if ballpoint.distance_to(pygame.Vector2(self.player_pos_b[0],self.player_pos_b[1])) < 50:
                reward+=0.5

            if (not self.border.collidepoint((int(self.player_pos_b[0]), int(self.player_pos_b[1]))) or
                not self.border.collidepoint((int(self.ball_pos[0]), int(self.ball_pos[1])))):
                done = True
                reward -= 30

            if circles_collide(self.player_pos_b,15,self.ball_pos,11):
                self.ball_pos[0]+=self.DIRECT[str(action)][0]
                self.ball_pos[1]+=self.DIRECT[str(action)][1]
                reward+=1
                if action ==1:
                    reward+=0.4

                if self.ball_pos[0]>310:
                    reward+=0.2 

                    if 195<self.ball_pos[1]<355 :
                        reward+=0.3


            if self.ball_pos[0]>310:
                reward+=0.2 

                if 195<self.ball_pos[1]<355 :
                    reward+=0.3  

            if ballpoint.distance_to(pygame.Vector2(self.player_pos_b[0],self.player_pos_b[1]))>pygame.Vector2(self.ball_pos[0],self.ball_pos[1]).distance_to(pygame.Vector2(self.player_pos_b[0],self.player_pos_b[1])):
                reward+=1
            else:
                reward-=0.7

            if pygame.Rect(25 + 20, 25 + 160//2 - 160//2, 
                        20, 160).collidepoint(int(self.ball_pos[0]),int(self.ball_pos[1])):
                reward+=50
                done=True
                winer = 'b'
            

        self.score+=1
 
        
        
        
        

        

        


        if self.score>self.max_score:
            done = True 
        
        return self._get_obs(player=player), reward, done, {"win": winer}
    
    def render(self, mode='human'):
        pygame.event.pump()
        self.screen.fill((50, 200, 50))  
        
        
        field_width, field_height = 550, 350
        field_x, field_y = 25, 25
        
        
        pygame.draw.rect(self.screen, (50, 200, 50), (field_x, field_y, field_width, field_height))
        
        
        pygame.draw.rect(self.screen, (255, 255, 255), (field_x, field_y, field_width, field_height), 3)
        
        
        pygame.draw.line(self.screen, (255, 255, 255), 
                        (field_x + field_width//2, field_y), 
                        (field_x + field_width//2, field_y + field_height), 2)
        
        
        pygame.draw.circle(self.screen, (255, 255, 255), 
                        (field_x + field_width//2, field_y + field_height//2), 
                        50, 2)
        
        
        gate_height = 160
        gate_width = 20  
        
        
        pygame.draw.rect(self.screen, (255, 255, 255), 
                    (field_x - gate_width, field_y + field_height//2 - gate_height//2, 
                        gate_width, gate_height), 2)
        
        
        pygame.draw.rect(self.screen, (255, 255, 255), 
                    (field_x + field_width, field_y + field_height//2 - gate_height//2, 
                        gate_width, gate_height), 2)
        
        
        penalty_area_width = 100
        penalty_area_height = 140
        
        pygame.draw.rect(self.screen, (255, 255, 255), 
                    (field_x, field_y + field_height//2 - penalty_area_height//2,
                        penalty_area_width, penalty_area_height), 2)
        
        pygame.draw.rect(self.screen, (255, 255, 255), 
                    (field_x + field_width - penalty_area_width, 
                        field_y + field_height//2 - penalty_area_height//2,
                        penalty_area_width, penalty_area_height), 2)
        
        
        pygame.draw.circle(self.screen, [255,0,0], self.player_pos_a, 15, width=0)
        self.screen.blit(self.font.render('A',True,(0,0,0)), (self.player_pos_a[0]-10,self.player_pos_a[1]-10))  
        pygame.draw.circle(self.screen, [0,0,255], self.player_pos_b, 15, width=0)  
        self.screen.blit(self.font.render('B',True,(0,0,0)), (self.player_pos_b[0]-10,self.player_pos_b[1]-10))  
        pygame.draw.circle(self.screen, [255,255,255], self.ball_pos, 11, width=0)  
        
        pygame.display.flip()
        self.clock.tick(180)



    def close(self):
        pygame.quit()



#[ball_x,ball_y,he_x,he_y,enemy_x,enemy_y]




def circles_collide(circle1_pos, circle1_radius, circle2_pos, circle2_radius):
    
    distance = math.hypot(circle2_pos[0] - circle1_pos[0], 
                          circle2_pos[1] - circle1_pos[1])
    
    return distance <= (circle1_radius + circle2_radius)