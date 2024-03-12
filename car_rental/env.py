from typing import Any
import numpy as np
import gymnasium as gym


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
    
    def __init__(self) -> None:
        super().__init__()
        
        # initiate observation and action spaces
        self.observation_space = gym.spaces.MultiDiscrete([self.max_cars+1]*2)
        self.action_space = gym.spaces.Discrete(
            n=2*self.max_transfer+1,
            start=-self.max_transfer
        )
        self.reset()
        
        # TODO: set render mode
        
    def reset(self, *, seed=None, options=None) -> tuple[Any, dict[str, Any]]:
        """Initiate new episode."""
        super().reset(seed=seed, options=options)
        
        # Choose intial state randomly from obseravation space
        self.state = self.observation_space.sample()
        
        # TODO: initiate rendering
        
        return self._get_obs(), self._get_info()
    
    def step(self, action: int) -> tuple[Any, Any, bool, bool, dict[str, Any]]:
        requests_returns = np.random.poisson(self.expected_requests_returns)
        self.state, reward = self._transition(
            self.state, action, requests_returns
        )
        
        terminated = False  # Is agent reach terminal state?
        truncated = False  # Is truncation condition outside the scope of the MDP satisfied?
        return self._get_obs(), reward, terminated, truncated, self._get_info() 
    
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
        return {}
