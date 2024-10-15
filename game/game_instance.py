import pygame
from game.paddle import Paddle
from game.ball import Ball
from ai.agent import Agent  # Add this import
import pickle
import os
import math

class GameInstance:
    def __init__(self, agent1, agent2, settings):
        self.agent1 = agent1
        self.agent2 = agent2
        self.settings = settings
        self.paddle1 = Paddle(settings, side="left")
        self.paddle2 = Paddle(settings, side="right")
        self.ball = Ball(settings)
        self.score1 = 0
        self.score2 = 0
        self.events = []  # Store significant events for UI to display
        self.last_hit = None  # Track which paddle last hit the ball
        self.total_reward1 = 0
        self.total_reward2 = 0
        self.ball_hits1 = 0
        self.ball_hits2 = 0
        self.performance_score1 = 0
        self.performance_score2 = 0
        self.last_distance1 = self._get_paddle_ball_distance(self.paddle1)
        self.last_distance2 = self._get_paddle_ball_distance(self.paddle2)
        self.total_hits1 = 0
        self.total_hits2 = 0

    def update(self):
        state1 = self.get_state(self.paddle1, self.paddle2)
        state2 = self.get_state(self.paddle2, self.paddle1)

        action1 = self.agent1.get_action(state1)
        action2 = self.agent2.get_action(state2)

        self.paddle1.move(action1)
        self.paddle2.move(action2)
        self.ball.move()

        reward1 = self._calculate_reward(self.paddle1, action1, 1)
        reward2 = self._calculate_reward(self.paddle2, action2, 2)

        # Check for collisions and update rewards
        if self.paddle1.collides_with(self.ball):
            self.ball.bounce()
            reward1 += 1.0  # Increased reward for hitting the ball
            self.last_hit = self.paddle1
            self.events.append("Agent 1 hit the ball")
            if isinstance(self.agent1, Agent):
                self.agent1.add_rebound()
            self.ball_hits1 += 1
            self.total_hits1 += 1
        elif self.paddle2.collides_with(self.ball):
            self.ball.bounce()
            reward2 += 1.0  # Increased reward for hitting the ball
            self.last_hit = self.paddle2
            self.events.append("Agent 2 hit the ball")
            if isinstance(self.agent2, Agent):
                self.agent2.add_rebound()
            self.ball_hits2 += 1
            self.total_hits2 += 1

        if self.ball.is_out():
            if self.ball.x < self.settings.width / 2:
                self.score2 += 1
                reward1 -= 2.0  # Increased penalty for losing a point
                reward2 += 2.0  # Increased reward for scoring a point
                self.events.append("Agent 2 scores!")
                if isinstance(self.agent1, Agent):
                    self.agent1.reset_rebounds()
            else:
                self.score1 += 1
                reward1 += 2.0  # Increased reward for scoring a point
                reward2 -= 2.0  # Increased penalty for losing a point
                self.events.append("Agent 1 scores!")
                if isinstance(self.agent2, Agent):
                    self.agent2.reset_rebounds()
            self.ball.reset()

        new_state1 = self.get_state(self.paddle1, self.paddle2)
        new_state2 = self.get_state(self.paddle2, self.paddle1)

        self.agent1.update(state1, action1, reward1, new_state1)
        self.agent2.update(state2, action2, reward2, new_state2)

        self.total_reward1 += reward1
        self.total_reward2 += reward2

        # Only add reward events if there's a significant change
        if abs(reward1) >= 0.1:
            self.events.append(f"Agent 1 reward: {reward1:.2f}")
        if abs(reward2) >= 0.1:
            self.events.append(f"Agent 2 reward: {reward2:.2f}")

        # Update performance scores
        self.update_performance_scores()

        # Update last distances for the next iteration
        self.last_distance1 = self._get_paddle_ball_distance(self.paddle1)
        self.last_distance2 = self._get_paddle_ball_distance(self.paddle2)

    def _calculate_reward(self, paddle, action, player_num):
        reward = 0

        # Reward for moving towards the ball
        current_distance = self._get_paddle_ball_distance(paddle)
        last_distance = self.last_distance1 if player_num == 1 else self.last_distance2
        distance_reward = (last_distance - current_distance) * 0.1
        reward += distance_reward

        # Penalty for unnecessary movement
        if action != 0 and abs(paddle.y + paddle.height/2 - self.ball.y) < paddle.height/4:
            reward -= 0.05

        # Reward for staying in the middle when the ball is far
        if abs(self.ball.x - paddle.x) > self.settings.width / 2:
            middle_y = self.settings.height / 2
            distance_to_middle = abs(paddle.y + paddle.height/2 - middle_y)
            middle_reward = 1 - (distance_to_middle / (self.settings.height/2))
            reward += middle_reward * 0.1

        # Reward for keeping the paddle in play area
        if paddle.y < 0 or paddle.y + paddle.height > self.settings.height:
            reward -= 0.1

        return reward

    def _get_paddle_ball_distance(self, paddle):
        return math.sqrt((paddle.x - self.ball.x)**2 + (paddle.y + paddle.height/2 - self.ball.y)**2)

    def get_state(self, paddle, opponent_paddle):
        return [
            paddle.y / self.settings.height,
            opponent_paddle.y / self.settings.height,
            self.ball.x / self.settings.width,
            self.ball.y / self.settings.height,
            (self.ball.x - paddle.x) / self.settings.width,  # Relative x position
            (self.ball.y - paddle.y) / self.settings.height,  # Relative y position
            self.ball.dx / self.settings.ball_speed,
            self.ball.dy / self.settings.ball_speed
        ]

    def save(self, filename):
        save_data = {
            'agent1': self.agent1,
            'agent2': self.agent2,
            'score1': self.score1,
            'score2': self.score2,
            'settings': self.settings,
            'total_reward1': self.total_reward1,
            'total_reward2': self.total_reward2
        }
        with open(filename, 'wb') as f:
            pickle.dump(save_data, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            save_data = pickle.load(f)
        
        instance = cls(save_data['agent1'], save_data['agent2'], save_data['settings'])
        instance.score1 = save_data['score1']
        instance.score2 = save_data['score2']
        instance.total_reward1 = save_data.get('total_reward1', 0)
        instance.total_reward2 = save_data.get('total_reward2', 0)
        return instance

    def get_learning_progress(self):
        if isinstance(self.agent1, Agent):
            progress1 = self.agent1.get_learning_progress()
        else:
            progress1 = 1.0  # Assume non-learning agents are always at 100%

        if isinstance(self.agent2, Agent):
            progress2 = self.agent2.get_learning_progress()
        else:
            progress2 = 1.0

        return (progress1 + progress2) / 2  # Average progress of both agents

    def get_confidence(self):
        confidence1 = self.agent1.get_confidence() if isinstance(self.agent1, Agent) else 1.0
        confidence2 = self.agent2.get_confidence() if isinstance(self.agent2, Agent) else 1.0
        return confidence1, confidence2

    def update_performance_scores(self):
        # Calculate performance scores based on various factors
        score_weight = 1
        hits_weight = 0.1

        self.performance_score1 = (
            self.score1 * score_weight +
            self.total_hits1 * hits_weight
        )
        self.performance_score2 = (
            self.score2 * score_weight +
            self.total_hits2 * hits_weight
        )

    def get_performance_ratio(self):
        total_score = self.performance_score1 + self.performance_score2
        if total_score == 0:
            return 0.5  # If no score, return a neutral value
        return self.performance_score1 / total_score
