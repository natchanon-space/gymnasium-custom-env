from typing import Any
import numpy as np
import gymnasium as gym
import pygame


class CarRentalEnv(gym.Env):
    # problem parameters
    max_cars = 20
    max_transfer = 5
    expected_requests_returns = np.array([
        [3, 4],
        [3, 2]
    ], dtype=int)
    rental_cost = 10
    transfer_expense = 2
    
    # PyGame window settings
    window = None
    window_size = 512
    clock = None
    
    # info data
    requests_returns = np.array(((None, None), (None, None)))
    action = None
    reward = None
    
    def __init__(self, render_mode="human") -> None:
        super().__init__()
        self.render_mode = render_mode
        
        # initiate observation and action spaces
        self.observation_space = gym.spaces.MultiDiscrete([self.max_cars+1]*2)
        self.action_space = gym.spaces.Discrete(
            n=2*self.max_transfer+1,
            start=-self.max_transfer
        )
        self.reset()
        
        if self.render_mode == "human":
            self.render()
        
    def reset(self, *, seed=None, options=None) -> tuple[Any, dict[str, Any]]:
        """Initiate new episode."""
        super().reset(seed=seed, options=options)
        # Choose intial state
        self.state = np.array([self.max_cars, self.max_cars])
        self.close()
        return self._get_obs(), self._get_info()
    
    def step(self, action: int) -> tuple[Any, Any, bool, bool, dict[str, Any]]:
        requests_returns = np.random.poisson(self.expected_requests_returns)
        self.state, reward = self._transition(
            self.state, action, requests_returns
        )
        self.requests_returns = requests_returns
        self.action = action
        self.reward = reward
        
        terminated = False  # Is agent reach terminal state?
        truncated = False  # Is truncation condition outside the scope of the MDP satisfied?
        
        if self.render_mode == "human":
            self.render()
        
        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    def render(self) -> None:
        return self._render_frame()
    
    def _render_frame(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            pygame.event
            self.window = pygame.display.set_mode(
                (self.window_size*2, self.window_size)
            )
        if self.clock is None:
            self.clock = pygame.time.Clock()
            
        canvas = pygame.Surface((self.window_size*2, self.window_size))
        canvas.fill((255, 255, 255))
        self.window.blit(canvas, canvas.get_rect())
    
        font = pygame.font.Font(pygame.font.get_default_font(), 40)
        # current state display
        for car, idx, pos in zip(self._get_obs(), [(0, "A"), (1, "B")], [(self.window_size/4, self.window_size/2), (self.window_size*3/4, self.window_size/2)]):
            pos = (pos[0], pos[1]-75)
            # set font
            # name
            font_surface = font.render(f"{idx[1]}", True, (0, 0, 0))
            font_rect = font_surface.get_rect()
            font_rect.midtop = (pos[0], pos[1]-50)
            self.window.blit(font_surface, font_rect)
            # car stats
            font_surface = font.render(f"{car}", True, (0, 0, 0))
            font_rect = font_surface.get_rect()
            font_rect.midtop = pos
            self.window.blit(font_surface, font_rect)
            # get info
            requests, returns = self._get_info()["requests_returns"]
            action = self._get_info()["action"]
            profit = self._get_info()["reward"]
            # requests
            font_surface = font.render(f"requests: {requests[idx[0]]}", True, (0, 0, 0))
            font_rect = font_surface.get_rect()
            font_rect.midtop = (pos[0], pos[1]+50)
            self.window.blit(font_surface, font_rect)
            # returns
            font_surface = font.render(f"returns: {returns[idx[0]]}", True, (0, 0, 0))
            font_rect = font_surface.get_rect()
            font_rect.midtop = (pos[0], pos[1]+100)
            self.window.blit(font_surface, font_rect)
        action = self._get_info()["action"]
        profit = self._get_info()["reward"]
        # actions
        action_str = ""
        if action is not None:
            if action >= 0:
                action_str = f"A -> B ({action})"
            else:
                action_str = f"A <- B ({-action})"
        font_surface = font.render(f"action: {action_str}", True, (0, 0, 0))
        font_rect = font_surface.get_rect()
        font_rect.midtop = (self.window_size/2, pos[1]+150)
        self.window.blit(font_surface, font_rect)
        # profit
        font_surface = font.render(f"profit: {profit}", True, (0, 0, 0))
        font_rect = font_surface.get_rect()
        font_rect.midtop = (self.window_size/2, pos[1]+200)
        self.window.blit(font_surface, font_rect)
            
        # draw graph
                    
        # update
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(20)
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
    
    def _transition(self, state: np.array, action: int, requests_returns: np.array) -> tuple[Any, dict[str, Any]]:
        # check validity of input action and state
        assert self.observation_space.contains(state), f"Invalid input state: {state}"
        assert self.action_space.contains(action), f"Invalid input action: {action}"
        
        next_state = np.copy(state)
        
        #== night time transfer ==#
        total_expense = self.transfer_expense * abs(action)
        if action >= 0:
            # transfer from A to B
            n_transferred = min(abs(action), state[0])
            next_state += np.array([-n_transferred, n_transferred])
        else:
            # transfer from B to A
            n_transferred = min(abs(action), state[1])
            next_state += np.array([n_transferred, -n_transferred])
        next_state = np.clip(next_state, 0, self.max_cars)
        assert self.observation_space.contains(next_state), "Invalid state after transfering"
        
        #== day time rental ==#
        requests, returns = requests_returns
        # return cars
        next_state += returns
        # request cars
        n_rented = np.minimum(next_state, requests)
        next_state -= n_rented
        total_income = self.rental_cost * n_rented.sum()
        # send exceed car to HQ
        next_state = np.clip(next_state, 0, self.max_cars)
        assert self.observation_space.contains(next_state), "Invalid state after rental"

        reward = total_income - total_expense
        return next_state, reward

    def _get_obs(self):
        return np.copy(self.state)
    
    def _get_info(self):
        return {
            "requests_returns": self.requests_returns,
            "action": self.action,
            "reward": self.reward,
        }
