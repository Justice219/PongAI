import pygame

class MainMenu:
    def __init__(self, screen, font):
        self.screen = screen
        self.font = font
        self.update_layout()

    def update_layout(self):
        self.buttons = [
            ("New Game", (self.screen.get_width() // 2, int(self.screen.get_height() * 0.4))),
            ("Load Game", (self.screen.get_width() // 2, int(self.screen.get_height() * 0.5))),
            ("Settings", (self.screen.get_width() // 2, int(self.screen.get_height() * 0.6)))
        ]

    def update(self):
        self.update_layout()
        self.screen.fill((0, 0, 0))
        mouse_pos = pygame.mouse.get_pos()
        click = pygame.mouse.get_pressed()[0]

        for text, pos in self.buttons:
            button_rect = pygame.Rect(pos[0] - int(self.screen.get_width() * 0.1), pos[1] - int(self.screen.get_height() * 0.025), 
                                      int(self.screen.get_width() * 0.2), int(self.screen.get_height() * 0.05))
            if button_rect.collidepoint(mouse_pos):
                pygame.draw.rect(self.screen, (100, 100, 100), button_rect)
                if click:
                    return text.lower().replace(" ", "_")
            else:
                pygame.draw.rect(self.screen, (50, 50, 50), button_rect)

            text_surface = self.font.render(text, True, (255, 255, 255))
            text_rect = text_surface.get_rect(center=pos)
            self.screen.blit(text_surface, text_rect)

        return None
