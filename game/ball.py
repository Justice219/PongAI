import pygame
import random

class Ball:
    def __init__(self, settings):
        self.update_settings(settings)
        self.reset()

    def update_settings(self, settings):
        self.settings = settings
        self.size = settings.ball_size
        self.speed = settings.ball_speed

    def reset(self):
        self.x = self.settings.width // 2
        self.y = self.settings.height // 2
        self.dx = random.choice([-1, 1]) * self.speed
        self.dy = random.uniform(-0.5, 0.5) * self.speed
        self.rect = pygame.Rect(self.x, self.y, self.size, self.size)

    def move(self):
        self.x += self.dx
        self.y += self.dy

        if self.y <= 0 or self.y >= self.settings.height - self.size:
            self.dy *= -1

        self.rect.x = int(self.x)
        self.rect.y = int(self.y)

    def bounce(self):
        self.dx *= -1.1  # Increase speed slightly on bounce
        self.dy = random.uniform(-0.5, 0.5) * self.speed

    def is_out(self):
        return self.x < 0 or self.x > self.settings.width
