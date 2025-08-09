import numpy as np
import pygame
import gym
from gym import spaces
import math
import pymunk
import pymunk.pygame_util

class FootballEnv(gym.Env):
    def __init__(self):
        super(FootballEnv, self).__init__()
        
        
        self.action_space = spaces.Discrete(4)  
        self.observation_space = spaces.Box(
            low=np.array([-1]*14, dtype=np.float32),
            high=np.array([1]*14, dtype=np.float32),
            dtype=np.float32
        )
        
        
        pygame.init()
        self.screen = pygame.display.set_mode((600, 400))
        pygame.display.set_caption('Football RL with Physics')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 15)


        
        
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        
        
        self.max_score = 800
        self.score = 0
        self.DIRECT = {'0': (0, -1), '1': (1, 0), '2': (0, 1), '3': (-1, 0)}  
        self.FORCE_MULTIPLIER = 50  
        self.MAX_SPEED = 300  
        
        
        self._create_field()
        self._create_players()
        self._create_ball()


    def _create_field(self):
        
        static_body = self.space.static_body
        
        
        walls = [

            pymunk.Segment(static_body, (575, 120), (575, 280), 2),
            pymunk.Segment(static_body, (25, 120), (25, 280), 2)

        ]
        
        for wall in walls:
            wall.elasticity = 0.8
            wall.friction = 0.5
            wall.filter = pymunk.ShapeFilter(categories=0b1, mask=0b1011)
            self.space.add(wall)

        walls = [
            pymunk.Segment(static_body, (25, 25), (25, 120), 2),
            pymunk.Segment(static_body, (25,120), (0, 120), 2),
            pymunk.Segment(static_body, (0, 120), (0, 280), 2),
            pymunk.Segment(static_body, (0, 280), (25, 280), 2),
            pymunk.Segment(static_body, (25, 280), (25, 375), 2),
            pymunk.Segment(static_body, (25, 375), (575, 375), 2),
            pymunk.Segment(static_body, (575, 375), (575, 280), 2),
            pymunk.Segment(static_body, (575, 280), (600, 280), 2),
            pymunk.Segment(static_body, (600, 280), (600, 120), 2),
            pymunk.Segment(static_body, (600, 120), (575, 120), 2),
            pymunk.Segment(static_body, (575, 120), (575, 25), 2),
            pymunk.Segment(static_body, (575, 25), (25, 25), 2),

        ]

        for wall in walls:
            wall.elasticity = 0.8
            wall.friction = 0.5
            wall.filter = pymunk.ShapeFilter(categories=0b1000, mask=0b1111)
            self.space.add(wall)
        
        

        
        
        self.field_lines = [
            ((300, 25), (300, 375)),  
            ((300, 200), 70, 1),        
            ((100, 125), (100, 275)),  
            ((500, 125), (500, 275))   
        ]
    
    def _create_players(self):
        
        self.player_a = pymunk.Body(10, float('inf'))
        self.player_a.position = (470, 200)
        self.player_a_shape = pymunk.Circle(self.player_a, 15)
        self.player_a_shape.elasticity = 0.2
        self.player_a_shape.friction = 1.5
        self.player_a.damping = 0.8
        self.player_a_shape.color = (255, 0, 0, 255)
        self.player_a_shape.filter = pymunk.ShapeFilter(categories=0b10, mask=0b1111)
        self.space.add(self.player_a, self.player_a_shape)
        
        
        self.player_b = pymunk.Body(10, float('inf'))
        self.player_b.position = (130, 200)
        self.player_b_shape = pymunk.Circle(self.player_b, 15)
        self.player_b_shape.elasticity = 0.2
        self.player_b_shape.friction = 1.5
        self.player_b.damping = 0.8
        self.player_b_shape.color = (0, 0, 255, 255)
        self.player_b_shape.filter = pymunk.ShapeFilter(categories=0b10, mask=0b1111)
        self.space.add(self.player_b, self.player_b_shape)
    
    def _create_ball(self):
        self.ball = pymunk.Body(5, float('inf'))
        self.ball.position = (300, 200)
        self.ball_shape = pymunk.Circle(self.ball, 11)
        self.ball_shape.elasticity = 0.9
        self.ball_shape.friction = 0.1
        self.ball_shape.color = (255, 255, 255, 255)
        self.ball_shape.filter = pymunk.ShapeFilter(categories=0b100, mask=0b1111)
        self.space.add(self.ball, self.ball_shape)
    
    def reset(self):
       
        self.player_a.position = (470, 200)
        self.player_b.position = (130, 200)
        self.ball.position = (300, 200)
        
        
        self.player_a.velocity = (0, 0)
        self.player_b.velocity = (0, 0)
        self.ball.velocity = (0, 0)
        
        self.score = 0
        return self._get_obs(player=0)
    
    def _get_obs(self, player: int) -> np.ndarray:
        ball_pos = self.ball.position
        player_a_pos = self.player_a.position
        player_b_pos = self.player_b.position
        
        if player == 1: 
            return np.array([
                ball_pos.x / 600, ball_pos.y / 400,
                player_a_pos.x / 600, player_a_pos.y / 400,
                player_b_pos.x / 600, player_b_pos.y / 400,
                math.dist(ball_pos, player_a_pos) / 730,
                math.dist(player_b_pos, player_a_pos) / 730,
                self.player_a.velocity.x / 700,
                self.player_a.velocity.y / 600,
                self.player_b.velocity.x / 700,
                self.player_b.velocity.y / 600,
                self.ball.velocity.x / 700,
                self.ball.velocity.y / 600
            ], dtype=np.float32)
        else:  
            return np.array([
                ball_pos.x / 600, ball_pos.y / 400,
                player_b_pos.x / 600, player_b_pos.y / 400,
                player_a_pos.x / 600, player_a_pos.y / 400,
                math.dist(ball_pos, player_b_pos) / 730,
                math.dist(player_b_pos, player_a_pos) / 730,
                self.player_b.velocity.x / 700,
                self.player_b.velocity.y / 600,
                self.player_a.velocity.x / 700,
                self.player_a.velocity.y / 600,
                self.ball.velocity.x / 700,
                self.ball.velocity.y / 600
            ], dtype=np.float32)
    
    def step(self, action: int, player: int):
        done = False
        reward = 0
        winner = '0'

        keys = pygame.key.get_pressed()

        if keys[pygame.K_t]:
            done = True
        
      
        direction = self.DIRECT[str(action)]
        force = (direction[0] * self.FORCE_MULTIPLIER, 
                 direction[1] * self.FORCE_MULTIPLIER)
        
        if player == 1:
            
            self.player_a.apply_impulse_at_local_point(force, (0, 0))
            
           
        else:
           
            self.player_b.apply_impulse_at_local_point(force, (0, 0))
            
            
        
    
        self.space.step(1/60.0)
        
      
        ball_pos = self.ball.position
        player_a_pos = self.player_a.position
        player_b_pos = self.player_b.position
        
      
        if ball_pos.x < 20 and 120 < ball_pos.y < 280:
            reward = 50 if player == 1 else -50
            done = True
            winner = 'a'
        elif ball_pos.x > 580 and 120 < ball_pos.y < 280:
            reward = 50 if player == 0 else -50
            done = True
            winner = 'b'
        
       
        if player == 1:

            if math.dist(ball_pos, player_a_pos) < 27:
                reward += 7

            if self.player_a.velocity.x < -50:
                reward += 0.2

            if abs(self.player_a.velocity.x) > 50 or abs(self.player_a.velocity.y):
                reward += 0.2
            else:
                reward -= 0.2

            if self.player_a.position.x < 53 or self.player_a.position.x > 547 or self.player_a.position.y < 53 or self.player_a.position.y > 347:
                reward-=0.7





        else:

            if math.dist(ball_pos, player_b_pos) < 27:
                reward += 7

            if self.player_b.velocity.x > 50:
                reward += 0.2

            if abs(self.player_b.velocity.x) > 50 or abs(self.player_b.velocity.y):
                reward += 0.2
            else:
                reward -= 0.2

            if self.player_b.position.x < 53 or self.player_b.position.x > 547 or self.player_b.position.y < 53 or self.player_b.position.y > 347:
                reward-=0.7
            
            
        
      
        if not (0 <= ball_pos.x <= 600) or not (0 <= ball_pos.y <= 400):
            
            done = True
        
        self.score += 1
        if self.score >= self.max_score:
            done = True

        
        
        return self._get_obs(player), reward, done, {"win": winner}
    
    def render(self):
        pygame.event.pump()
        self.screen.fill((50, 200, 50))



        field_width, field_height = 550, 350
        field_x, field_y = 25, 25




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







        self.space.debug_draw(self.draw_options)

        pygame.draw.line(self.screen, (200, 200, 200), (25, 120), (25, 280), 4)
        pygame.draw.line(self.screen, (200, 200, 200), (575, 120), (575, 280), 4)




        self.screen.blit(self.font.render('A', True, (0, 0, 0)),
                         (self.player_a.position.x - 5, self.player_a.position.y - 8))
        self.screen.blit(self.font.render('B', True, (0, 0, 0)),
                         (self.player_b.position.x - 5, self.player_b.position.y - 8))


        score_text = self.font.render(f"Time: {self.score}/{self.max_score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))


        dist_a = round(math.dist(self.ball.position, self.player_a.position))
        dist_b = round(math.dist(self.ball.position, self.player_b.position))
        color_a = (255,255,255)
        color_b = (255,255,255)

        if dist_a < 27:
            color_a = (255,0,0)

        if dist_b < 27:
            color_b = (0,0,255)


        dist_a_text = self.font.render('dist a: '+str(dist_a), True,
                                       color_a)
        self.screen.blit(dist_a_text, (10, 30))

        dist_b_text = self.font.render('dist b: '+str(dist_b), True,
                                       color_b)
        self.screen.blit(dist_b_text, (10, 50))

        velos_a_text = self.font.render(f'veloc a: {str(round(self.player_a.velocity.x))}/{str(round(self.player_a.velocity.y))}',
                                        True,(255,255,255))
        self.screen.blit(velos_a_text, (10, 70))

        velos_b_text = self.font.render(f'veloc b: {str(round(self.player_b.velocity.x))}/{str(round(self.player_b.velocity.y))}',
                                        True,(255,255,255))
        self.screen.blit(velos_b_text, (10, 90))
        
        pygame.display.flip()
        self.clock.tick(30)
    
    def close(self):
        pygame.quit()


def circles_collide(circle1_pos, circle1_radius, circle2_pos, circle2_radius):
    distance = math.dist(circle1_pos, circle2_pos)
    return distance <= (circle1_radius + circle2_radius)