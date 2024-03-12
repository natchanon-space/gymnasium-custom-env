import numpy as np
import itertools as it
from scipy.stats import poisson
import os
import pickle
from tqdm import tqdm
from .env import CarRentalEnv


class CarRentalAgentImpl:
    def __init__(self, env: CarRentalEnv) -> None:
        self.env = env
        
    def fit(self) -> None:
        pass
    
    def get_action(self, state: np.array) -> int:
        pass
    

class NaiveAgent(CarRentalAgentImpl):
    def get_action(self, state: np.array) -> int:
        return self.env.action_space.sample()
    

class PolicyIterationAgent(CarRentalAgentImpl):
    def __init__(self, max_poisson: int, env: CarRentalEnv) -> None:
        super().__init__(env)
        
        # set dimension for 4 independent poisson distribution
        self.max_poisson = max_poisson
        prob_shape = [self.max_poisson] * 4
        self.probability = np.fromfunction(
            self._transition_probability,
            shape=prob_shape,
            dtype=np.float32
        )
        # state
        self.states = np.array(list(it.product(
            range(self.env.max_cars+1),
            range(self.env.max_cars+1)
        )))
        self.state_values = np.zeros([self.env.max_cars+1]*2, dtype=np.float32)
        self.actions = np.array([i for i in range(-self.env.max_transfer, self.env.max_transfer+1)], dtype=int)
        # initiate arbitrarily policy with an action in action space
        self.policy = np.empty([self.env.max_cars+1]*2, dtype=int)
        self.policy.fill(0)

    def fit(self, max_iteration: int = 10, max_evaluation_iteration: int = 10, log_dir: str = None) -> None:
        for counter in range(max_iteration):
            print(f"Fitting... (iteration {counter+1})")
            self._policy_evaluation(max_iteration=max_evaluation_iteration)
            stable_policy = self._policy_improvement()
            
            if log_dir is not None:
                with open(os.path.join(log_dir, f"iter-{counter+1}.pkl"), "wb") as file:
                    pickle.dump(self, file)

            if stable_policy:
                break
    
    def get_action(self, state: np.array) -> int:
        assert self.env.observation_space.contains(state), f"Invalid input state: {state}"
        return self.policy[state[0], state[1]]
        
    def _transition_probability(self, rq_a, rq_b, rt_a, rt_b):
        expeted_requests_returns = self.env.expected_requests_returns.flatten()
        prob = 1.0
        for k, mu in zip([rq_a, rq_b, rt_a, rt_b], expeted_requests_returns):
            prob *= poisson.pmf(k, mu)
        return prob
    
    def _policy_evaluation(self, gamma: float = 0.9, theta: float = 0.1, max_iteration: int = 10):
        print("Evaluating policy...")
        for iter_count in range(max_iteration):
            
            delta = 0
            
            pbar = tqdm(self.states)
            for i, j in pbar:
                pbar.set_description(f"iter {iter_count}")
                pbar.refresh()
                temp_state_value = self.state_values[i, j]
                new_state_value = 0
                
                for k, prob in np.ndenumerate(self.probability):
                    rq_a, rq_b, rt_a, rt_b = k
                    next_state, reward = self.env._transition(
                        state=np.array([i, j]),
                        action=self.policy[i, j],
                        requests_returns=np.array([[rq_a, rq_b], [rt_a, rt_b]])
                    )
                    new_state_value += prob * (reward + gamma*self.state_values[next_state[0], next_state[1]])
                self.state_values[i, j] = new_state_value
                
                delta = max(delta, abs(temp_state_value - self.state_values[i, j]))
                pbar.set_description(f"iter {iter_count} (delta={delta:.3f})")
                pbar.refresh()
            if delta < theta:
                print(f"stopped (delta < {theta})")
                break

    def _policy_improvement(self, gamma: float = 0.9) -> bool:
        print("Improving policy...")
        policy_stable = True
        pbar = tqdm(self.states)
        
        for i, j in pbar:
            old_action = self.policy[i, j]

            mmax = -np.inf
            new_action = old_action
            
            for action in self.actions:
                value = 0.0
                for k, prob in np.ndenumerate(self.probability):
                    rq_a, rq_b, rt_a, rt_b = k
                    next_state, reward = self.env._transition(
                        state=np.array([i, j]),
                        action=action,
                        requests_returns=np.array([[rq_a, rq_b], [rt_a, rt_b]])
                    )
                    value += prob * (reward + gamma*self.state_values[next_state[0], next_state[1]])
                if value > mmax:
                    mmax = value
                    new_action = action

            self.policy[i, j] = new_action

            if old_action != self.policy[i, j]:
                policy_stable = False
                pbar.set_description(f"policy_stable: {policy_stable}")

        return policy_stable
