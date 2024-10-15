import pygame
import torch
import os
import shutil
from game.game_instance import GameInstance
from ai.agent import Agent
from ui.main_menu import MainMenu
from ui.game_ui import GameUI
from utils.settings import Settings
from ai.ai_factory import AIFactory
import glob
import time

class PongAISimulation:
    def __init__(self):
        pygame.init()
        info = pygame.display.Info()
        self.screen_width = int(info.current_w * 0.6)  # Increased from 0.5 to 0.6
        self.screen_height = int(info.current_h * 0.6)  # Increased from 0.5 to 0.6
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.RESIZABLE)
        pygame.display.set_caption("Pong AI Simulation")
        self.clock = pygame.time.Clock()
        self.update_font()
        self.instances = []
        self.current_instance = None
        self.settings = Settings(self.screen_width, self.screen_height)
        self.main_menu = MainMenu(self.screen, self.font)
        self.game_ui = GameUI(self.screen, self.font)
        self.paused = False
        self.ai_types = ["dqn", "random"]
        self.current_ai_type = 0
        self.training_mode = False
        self.training_speed = 5
        self.accumulated_events = []
        self.event_update_interval = 60  # Update console every 60 frames (1 second at 60 FPS)
        self.frame_count = 0
        self.autosave_interval = 300  # Autosave every 5 minutes (300 seconds)
        self.last_autosave_time = time.time()
        self.generation = 1  # Add this line to keep track of the current generation
        self.save_directory = "saves"
        os.makedirs(self.save_directory, exist_ok=True)

    def update_font(self):
        self.font = pygame.font.Font(None, int(self.screen_height * 0.03))

    def run(self):
        running = True
        while running:
            if self.current_instance:
                self.run_game()
            else:
                self.run_main_menu()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if self.current_instance:
                            self.paused = not self.paused
                        else:
                            running = False
                    elif event.key == pygame.K_TAB:
                        self.current_ai_type = (self.current_ai_type + 1) % len(self.ai_types)
                        print(f"Switched to AI type: {self.ai_types[self.current_ai_type]}")
                    elif event.key == pygame.K_t:
                        self.training_mode = not self.training_mode
                        print(f"Training mode: {'ON' if self.training_mode else 'OFF'}")
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left mouse button
                        if self.current_instance:
                            if self.game_ui.toggle_network_view(event.pos):
                                # Network view toggled, no need for additional action
                                pass
                            elif self.game_ui.check_button_click(event.pos, "Main Menu"):
                                self.current_instance = None
                            elif self.game_ui.check_button_click(event.pos, "Save Game"):
                                self.save_game()
                elif event.type == pygame.VIDEORESIZE:
                    self.screen_width, self.screen_height = event.size
                    self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.RESIZABLE)
                    self.update_font()
                    self.settings = Settings(self.screen_width, self.screen_height)
                    self.game_ui = GameUI(self.screen, self.font)
                    self.main_menu = MainMenu(self.screen, self.font)
                    if self.current_instance:
                        self.current_instance.settings = self.settings
                        self.current_instance.paddle1.update_settings(self.settings)
                        self.current_instance.paddle2.update_settings(self.settings)
                        self.current_instance.ball.update_settings(self.settings)

            pygame.display.flip()
            self.clock.tick(60 if not self.training_mode else 60 * self.training_speed)

        pygame.quit()

    def run_main_menu(self):
        action = self.main_menu.update()
        if action == "new_game":
            self.create_new_instance()
        elif action == "load_game":
            self.load_instance()
        elif action == "settings":
            self.settings = self.settings.show_settings_menu(self.screen, self.font)

    def run_game(self):
        if not self.paused:
            for _ in range(self.training_speed if self.training_mode else 1):
                self.current_instance.update()
                self.accumulated_events.extend(self.current_instance.events)
                self.current_instance.events.clear()  # Clear events after accumulating

        self.frame_count += 1

        # Process accumulated events every second
        if self.frame_count >= self.event_update_interval:
            if self.accumulated_events:
                for event in self.accumulated_events:
                    self.game_ui.add_console_message(event)
                self.accumulated_events.clear()  # Clear accumulated events after processing

            # Add total reward information
            self.game_ui.add_console_message(f"Total Agent 1 reward: {self.current_instance.total_reward1:.2f}")
            self.game_ui.add_console_message(f"Total Agent 2 reward: {self.current_instance.total_reward2:.2f}")

            self.frame_count = 0  # Reset frame count

        # Autosave
        current_time = time.time()
        if current_time - self.last_autosave_time >= self.autosave_interval:
            self.autosave()
            self.last_autosave_time = current_time

        self.game_ui.draw(self.current_instance, self.paused, self.training_mode)

    def create_new_instance(self):
        # Delete all existing save files
        self.delete_all_saves()
        
        agent1 = AIFactory.create_agent(self.ai_types[self.current_ai_type], self.settings)
        agent2 = AIFactory.create_agent(self.ai_types[self.current_ai_type], self.settings)
        self.current_instance = GameInstance(agent1, agent2, self.settings)
        self.instances.append(self.current_instance)
        self.generation = 1  # Reset generation counter
        self.game_ui.add_console_message("New game created. All previous saves deleted.")

    def load_instance(self):
        saves = glob.glob(os.path.join(self.save_directory, 'generation_*.pkl'))
        if saves:
            latest_save = max(saves, key=os.path.getctime)
            self.current_instance = GameInstance.load(latest_save)
            self.instances.append(self.current_instance)
            self.game_ui.add_console_message(f"Loaded latest save: {os.path.basename(latest_save)}")
            # Extract generation number from filename
            self.generation = int(os.path.basename(latest_save).split('_')[1].split('.')[0]) + 1
        else:
            self.game_ui.add_console_message("No saves found. Starting a new game.")
            self.create_new_instance()

    def autosave(self):
        if self.current_instance:
            filename = f'generation_{self.generation:04d}.pkl'
            filepath = os.path.join(self.save_directory, filename)
            self.current_instance.save(filepath)
            self.game_ui.add_console_message(f"Game autosaved: {filename}")
            self.generation += 1

    def save_game(self):
        if self.current_instance:
            filename = f'generation_{self.generation:04d}.pkl'
            filepath = os.path.join(self.save_directory, filename)
            self.current_instance.save(filepath)
            self.game_ui.add_console_message(f"Game manually saved: {filename}")
            self.generation += 1

    def delete_all_saves(self):
        for file in glob.glob(os.path.join(self.save_directory, 'generation_*.pkl')):
            os.remove(file)
        self.game_ui.add_console_message("All save files deleted.")

if __name__ == "__main__":
    simulation = PongAISimulation()
    simulation.run()
