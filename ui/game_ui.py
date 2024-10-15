import pygame
from ui.network_visualizer import NetworkVisualizer
from ai.agent import Agent
from collections import deque

class GameUI:
    def __init__(self, screen, font):
        self.screen = screen
        self.font = font
        self.network_visualizer = NetworkVisualizer(screen, font)
        self.show_network_agent1 = False
        self.show_network_agent2 = False
        self.update_layout()
        self.console_output = deque(maxlen=20)  # Store last 20 console messages
        self.button_height = int(self.screen.get_height() * 0.05)
        self.button_spacing = int(self.screen.get_height() * 0.02)

    def update_layout(self):
        self.left_sidebar_width = int(self.screen.get_width() * 0.2)
        self.right_sidebar_width = int(self.screen.get_width() * 0.2)
        self.game_area_width = self.screen.get_width() - self.left_sidebar_width - self.right_sidebar_width
        self.game_area_height = self.screen.get_height()

    def draw(self, game_instance, paused, training_mode):
        self.update_layout()
        self.screen.fill((0, 0, 0))

        # Draw game area
        game_area = pygame.Rect(self.left_sidebar_width, 0, self.game_area_width, self.game_area_height)
        pygame.draw.rect(self.screen, (50, 50, 50), game_area)

        # Calculate scale factors
        scale_x = self.game_area_width / game_instance.settings.width
        scale_y = self.game_area_height / game_instance.settings.height

        # Draw paddles
        paddle1_rect = pygame.Rect(
            self.left_sidebar_width + int(game_instance.paddle1.x * scale_x),
            int(game_instance.paddle1.y * scale_y),
            int(game_instance.paddle1.width * scale_x),
            int(game_instance.paddle1.height * scale_y)
        )
        paddle2_rect = pygame.Rect(
            self.left_sidebar_width + int(game_instance.paddle2.x * scale_x),
            int(game_instance.paddle2.y * scale_y),
            int(game_instance.paddle2.width * scale_x),
            int(game_instance.paddle2.height * scale_y)
        )
        pygame.draw.rect(self.screen, (255, 255, 255), paddle1_rect)
        pygame.draw.rect(self.screen, (255, 255, 255), paddle2_rect)

        # Draw ball
        ball_rect = pygame.Rect(
            self.left_sidebar_width + int(game_instance.ball.x * scale_x),
            int(game_instance.ball.y * scale_y),
            int(game_instance.ball.size * scale_x),
            int(game_instance.ball.size * scale_y)
        )
        pygame.draw.rect(self.screen, (255, 255, 255), ball_rect)

        # Draw scores
        score1_text = self.font.render(str(game_instance.score1), True, (255, 255, 255))
        score2_text = self.font.render(str(game_instance.score2), True, (255, 255, 255))
        self.screen.blit(score1_text, (int(self.screen.get_width() * 0.4), int(self.screen.get_height() * 0.05)))
        self.screen.blit(score2_text, (int(self.screen.get_width() * 0.6), int(self.screen.get_height() * 0.05)))

        # Draw left sidebar
        self.draw_left_sidebar(game_instance)

        # Draw right sidebar (console output)
        self.draw_right_sidebar()

        # Draw pause indicator
        if paused:
            pause_text = self.font.render("PAUSED", True, (255, 0, 0))
            pause_rect = pause_text.get_rect(center=(self.screen.get_width() // 2, self.screen.get_height() // 2))
            self.screen.blit(pause_text, pause_rect)

        # Draw training mode indicator
        if training_mode:
            training_text = self.font.render("TRAINING MODE", True, (0, 255, 0))
            training_rect = training_text.get_rect(center=(self.screen.get_width() // 2, int(self.screen.get_height() * 0.05)))
            self.screen.blit(training_text, training_rect)

        # Draw neural network visualizations
        if self.show_network_agent1 and isinstance(game_instance.agent1, Agent):
            state1 = game_instance.get_state(game_instance.paddle1, game_instance.paddle2)
            activations1 = game_instance.agent1.get_network_activations(state1)
            self.network_visualizer.draw_network(game_instance.agent1.policy_net, activations1, game_area)

        if self.show_network_agent2 and isinstance(game_instance.agent2, Agent):
            state2 = game_instance.get_state(game_instance.paddle2, game_instance.paddle1)
            activations2 = game_instance.agent2.get_network_activations(state2)
            self.network_visualizer.draw_network(game_instance.agent2.policy_net, activations2, game_area)

        # Draw confidence meters
        confidence1, confidence2 = game_instance.get_confidence()
        self.draw_confidence_meter(confidence1, "Agent 1", int(self.screen.get_width() * 0.25), int(self.screen.get_height() * 0.95))
        self.draw_confidence_meter(confidence2, "Agent 2", int(self.screen.get_width() * 0.55), int(self.screen.get_height() * 0.95))

        # Calculate button positions
        button_y = int(self.screen.get_height() * 0.7)
        
        # Draw neural network toggle buttons
        self.draw_toggle_button("Show Agent 1 Network", self.show_network_agent1, int(self.screen.get_width() * 0.01), button_y)
        button_y += self.button_height + self.button_spacing
        self.draw_toggle_button("Show Agent 2 Network", self.show_network_agent2, int(self.screen.get_width() * 0.01), button_y)
        button_y += self.button_height + self.button_spacing

        # Draw save game button
        self.draw_button("Save Game", int(self.screen.get_width() * 0.01), button_y)
        button_y += self.button_height + self.button_spacing

        # Draw main menu button
        self.draw_button("Main Menu", int(self.screen.get_width() * 0.01), button_y)

        # Draw performance bar
        self.draw_performance_bar(game_instance)

    def draw_left_sidebar(self, game_instance):
        pygame.draw.rect(self.screen, (30, 30, 30), (0, 0, self.left_sidebar_width, self.screen.get_height()))
        
        # Agent 1 data
        self.draw_agent_data(game_instance.agent1, "Agent 1", int(self.screen.get_width() * 0.01), int(self.screen.get_height() * 0.05))
        
        # Agent 2 data
        self.draw_agent_data(game_instance.agent2, "Agent 2", int(self.screen.get_width() * 0.01), int(self.screen.get_height() * 0.5))

    def draw_right_sidebar(self):
        pygame.draw.rect(self.screen, (30, 30, 30), (self.screen.get_width() - self.right_sidebar_width, 0, self.right_sidebar_width, self.screen.get_height()))
        
        title = self.font.render("Console Output", True, (255, 255, 255))
        self.screen.blit(title, (self.screen.get_width() - self.right_sidebar_width + 10, 10))

        for i, message in enumerate(self.console_output):
            text_surface = self.font.render(message, True, (200, 200, 200))
            self.screen.blit(text_surface, (self.screen.get_width() - self.right_sidebar_width + 10, 50 + i * int(self.screen.get_height() * 0.03)))

    def add_console_message(self, message):
        self.console_output.append(message)

    def draw_agent_data(self, agent, name, x, y):
        title = self.font.render(name, True, (255, 255, 255))
        self.screen.blit(title, (x, y))
        
        data = [
            f"Type: {agent.__class__.__name__}",
            f"Epsilon: {agent.epsilon:.2f}",
            f"Memory: {len(agent.memory) if hasattr(agent.memory, '__len__') else 'N/A'}",
            f"Last Reward: {agent.last_reward if hasattr(agent, 'last_reward') else 'N/A'}"
        ]

        for i, text in enumerate(data):
            text_surface = self.font.render(text, True, (200, 200, 200))
            self.screen.blit(text_surface, (x, y + int(self.screen.get_height() * 0.05) + i * int(self.screen.get_height() * 0.04)))

    def draw_toggle_button(self, text, is_on, x, y):
        button_width = self.left_sidebar_width - int(self.screen.get_width() * 0.02)
        button_rect = pygame.Rect(x, y, button_width, self.button_height)
        color = (0, 255, 0) if is_on else (255, 0, 0)
        pygame.draw.rect(self.screen, color, button_rect)
        text_surface = self.font.render(text, True, (0, 0, 0))
        text_rect = text_surface.get_rect(center=button_rect.center)
        self.screen.blit(text_surface, text_rect)

    def toggle_network_view(self, mouse_pos):
        if self.check_button_click(mouse_pos, "Show Agent 1 Network"):
            self.show_network_agent1 = not self.show_network_agent1
            return True
        elif self.check_button_click(mouse_pos, "Show Agent 2 Network"):
            self.show_network_agent2 = not self.show_network_agent2
            return True
        return False

    def draw_learning_progress(self, progress):
        bar_width = int(self.screen.get_width() * 0.2)
        bar_height = int(self.screen.get_height() * 0.02)
        bar_x = int(self.screen.get_width() * 0.4)
        bar_y = int(self.screen.get_height() * 0.95)

        # Draw background
        pygame.draw.rect(self.screen, (100, 100, 100), (bar_x, bar_y, bar_width, bar_height))
        
        # Draw progress
        progress_width = int(bar_width * progress)
        pygame.draw.rect(self.screen, (0, 255, 0), (bar_x, bar_y, progress_width, bar_height))

        # Draw text
        text = self.font.render(f"Learning Progress: {progress:.0%}", True, (255, 255, 255))
        text_rect = text.get_rect(center=(bar_x + bar_width // 2, bar_y - bar_height))
        self.screen.blit(text, text_rect)

    def draw_confidence_meter(self, confidence, agent_name, x, y):
        bar_width = int(self.screen.get_width() * 0.2)
        bar_height = int(self.screen.get_height() * 0.02)

        # Draw background
        pygame.draw.rect(self.screen, (100, 100, 100), (x, y, bar_width, bar_height))
        
        # Draw confidence
        confidence_width = int(bar_width * confidence)
        pygame.draw.rect(self.screen, (0, 255, 0), (x, y, confidence_width, bar_height))

        # Draw text
        text = self.font.render(f"{agent_name} Confidence: {confidence:.2f}", True, (255, 255, 255))
        text_rect = text.get_rect(center=(x + bar_width // 2, y - bar_height))
        self.screen.blit(text, text_rect)

    def draw_button(self, text, x, y):
        button_width = self.left_sidebar_width - int(self.screen.get_width() * 0.02)
        button_rect = pygame.Rect(x, y, button_width, self.button_height)
        pygame.draw.rect(self.screen, (100, 100, 100), button_rect)
        text_surface = self.font.render(text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=button_rect.center)
        self.screen.blit(text_surface, text_rect)

    def check_button_click(self, mouse_pos, button_name):
        button_width = self.left_sidebar_width - int(self.screen.get_width() * 0.02)
        button_y = int(self.screen.get_height() * 0.7)

        if button_name == "Show Agent 1 Network":
            button_rect = pygame.Rect(int(self.screen.get_width() * 0.01), button_y, button_width, self.button_height)
        elif button_name == "Show Agent 2 Network":
            button_y += self.button_height + self.button_spacing
            button_rect = pygame.Rect(int(self.screen.get_width() * 0.01), button_y, button_width, self.button_height)
        elif button_name == "Save Game":
            button_y += 2 * (self.button_height + self.button_spacing)
            button_rect = pygame.Rect(int(self.screen.get_width() * 0.01), button_y, button_width, self.button_height)
        elif button_name == "Main Menu":
            button_y += 3 * (self.button_height + self.button_spacing)
            button_rect = pygame.Rect(int(self.screen.get_width() * 0.01), button_y, button_width, self.button_height)
        else:
            return False

        return button_rect.collidepoint(mouse_pos)

    def draw_performance_bar(self, game_instance):
        bar_width = int(self.screen.get_width() * 0.4)
        bar_height = int(self.screen.get_height() * 0.03)
        bar_x = (self.screen.get_width() - bar_width) // 2
        bar_y = int(self.screen.get_height() * 0.02)

        # Draw background
        pygame.draw.rect(self.screen, (100, 100, 100), (bar_x, bar_y, bar_width, bar_height))

        # Calculate the performance ratio
        performance_ratio = game_instance.get_performance_ratio()

        # Calculate the division point
        division_point = int(bar_width * performance_ratio)

        # Draw Agent 1's portion (left side)
        pygame.draw.rect(self.screen, (0, 255, 0), (bar_x, bar_y, division_point, bar_height))

        # Draw Agent 2's portion (right side)
        pygame.draw.rect(self.screen, (255, 0, 0), (bar_x + division_point, bar_y, bar_width - division_point, bar_height))

        # Draw dividing line
        pygame.draw.line(self.screen, (255, 255, 255), (bar_x + bar_width // 2, bar_y), (bar_x + bar_width // 2, bar_y + bar_height), 2)

        # Draw labels
        font = pygame.font.Font(None, int(bar_height * 0.8))
        agent1_text = font.render(f"Agent 1: {game_instance.score1} ({game_instance.total_hits1})", True, (255, 255, 255))
        agent2_text = font.render(f"Agent 2: {game_instance.score2} ({game_instance.total_hits2})", True, (255, 255, 255))
        self.screen.blit(agent1_text, (bar_x - agent1_text.get_width() - 5, bar_y + (bar_height - agent1_text.get_height()) // 2))
        self.screen.blit(agent2_text, (bar_x + bar_width + 5, bar_y + (bar_height - agent2_text.get_height()) // 2))

        # Draw percentage text
        percentage_text = f"{performance_ratio * 100:.1f}% - {100 - performance_ratio * 100:.1f}%"
        percentage_surface = font.render(percentage_text, True, (255, 255, 255))
        percentage_rect = percentage_surface.get_rect(center=(bar_x + bar_width // 2, bar_y + bar_height + 5))
        self.screen.blit(percentage_surface, percentage_rect)
