import pygame

class Settings:
    def __init__(self, screen_width, screen_height):
        self.width = screen_width
        self.height = screen_height
        self.paddle_width = int(screen_width * 0.015)
        self.paddle_height = int(screen_height * 0.15)
        self.ball_size = int(min(screen_width, screen_height) * 0.02)
        self.ball_speed = int(min(screen_width, screen_height) * 0.01)
        self.paddle_speed = int(screen_height * 0.01)

    def show_settings_menu(self, screen, font):
        settings = [
            ("Ball Speed", "ball_speed", 1, int(min(self.width, self.height) * 0.02)),
            ("Paddle Speed", "paddle_speed", 1, int(self.height * 0.02)),
        ]

        running = True
        selected = 0

        while running:
            screen.fill((0, 0, 0))

            for i, (name, attr, min_val, max_val) in enumerate(settings):
                color = (255, 255, 255) if i == selected else (150, 150, 150)
                text = font.render(f"{name}: {getattr(self, attr)}", True, color)
                screen.blit(text, (int(self.width * 0.1), int(self.height * 0.1 + i * self.height * 0.1)))

            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_UP:
                        selected = (selected - 1) % len(settings)
                    elif event.key == pygame.K_DOWN:
                        selected = (selected + 1) % len(settings)
                    elif event.key == pygame.K_LEFT:
                        name, attr, min_val, max_val = settings[selected]
                        setattr(self, attr, max(min_val, getattr(self, attr) - 1))
                    elif event.key == pygame.K_RIGHT:
                        name, attr, min_val, max_val = settings[selected]
                        setattr(self, attr, min(max_val, getattr(self, attr) + 1))

        return self
