import pygame
import sys
import random

def game_floor():
  screen.blit(ground, (floor_x_pos,400))
  screen.blit(ground, (floor_x_pos + 576, 400))

def check_collision():
  if bird_rect.top <= -100 or bird_rect.bottom >= 400:
    return False
  return True

pygame.init()
clock = pygame.time.Clock()

#SCREEN SIZE
screen  = pygame.display.set_mode((576,512))

#LOAD THE IMAGES

#load bg
bg_day = pygame.image.load('assets/background-day.png').convert()
bg_day = pygame.transform.scale2x(bg_day)
ground = pygame.image.load('assets/base.png').convert()
ground = pygame.transform.scale2x(ground)

#load bird
bird = pygame.image.load('assets/bluebird-midflap.png').convert_alpha()
bird = pygame.transform.scale2x(bird)

#GAME_PHYSICS

gravity = 0.8
bird_movement = 0

#COLLISION DETECT
bird_rect = bird.get_rect(center = (100,256))

floor_x_pos= 0

while True:
  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      pygame.quit()
      sys.exit()
    if event.type == pygame.KEYDOWN:
      if event.key == pygame.K_SPACE and game_active:
        bird_movement -= 40
        bird_rect.center = (100,256)

    #building the images
    screen.blit(bg_day,(0,0))

  #controls
  bird_movement += gravity
  bird_rect.centery += bird_movement
  screen.blit(bird, (100,bird_movement))


  check_collision()

  floor_x_pos -=1
  game_floor()
  if floor_x_pos <= -576:
    floor_x_pos = 0
  pygame.display.update()
  clock.tick(120)