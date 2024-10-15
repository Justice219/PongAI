import pygame

class Paddle:
    def __init__(self, settings, side):
        self.update_settings(settings, side)

    def update_settings(self, settings, side=None):
        self.settings = settings
        self.width = settings.paddle_width
        self.height = settings.paddle_height
        self.speed = settings.paddle_speed
        
        if side:
            if side == "left":
                self.x = int(settings.width * 0.05)
            else:
                self.x = int(settings.width * 0.95 - self.width)
        
        self.y = (settings.height - self.height) // 2
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)

    def move(self, action):
        if action == 1:  # Move up
            self.y = max(0, self.y - self.speed)
        elif action == 2:  # Move down
            self.y = min(self.settings.height - self.height, self.y + self.speed)
        
        self.rect.y = self.y

    def collides_with(self, ball):
        return self.rect.colliderect(ball.rect)

    def draw(self, screen):
        pygame.draw.rect(screen, (255, 255, 255), self.rect)
