"""
Q-Learning Agent Implementation

This module implements a Q-Learning agent for reinforcement learning.
Q-Learning is a model-free reinforcement learning algorithm that learns
the value of an action in a particular state.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
import pickle


class QAgent:
    """
    Q-Learning Agent for Reinforcement Learning
    
    Q-Learning is an off-policy temporal difference learning algorithm that
    learns the optimal action-value function Q*(s, a). The agent learns by
    updating Q-values based on the Bellman equation:
    
    Q(s, a) = Q(s, a) + α * [r + γ * max(Q(s', a')) - Q(s, a)]
    
    where:
    - s: current state
    - a: action taken
    - r: reward received
    - s': next state
    - α: learning rate
    - γ: discount factor
    
    Attributes:
        learning_rate (float): The learning rate (alpha) for Q-value updates
        discount_factor (float): The discount factor (gamma) for future rewards
        epsilon (float): Exploration rate for epsilon-greedy policy
        epsilon_decay (float): Rate at which epsilon decays over time
        epsilon_min (float): Minimum epsilon value
        q_table (dict): The Q-table storing state-action values
    """
    
    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01
    ):
        """
        Initialize the Q-Learning agent.
        
        Args:
            learning_rate: Learning rate (alpha) for Q-value updates (0 < α ≤ 1)
            discount_factor: Discount factor (gamma) for future rewards (0 ≤ γ ≤ 1)
            epsilon: Initial exploration rate for epsilon-greedy policy (0 ≤ ε ≤ 1)
            epsilon_decay: Rate at which epsilon decays after each episode
            epsilon_min: Minimum value for epsilon
            
        Raises:
            ValueError: If parameters are outside valid ranges
        """
        # Validate learning rate
        if not (0 < learning_rate <= 1):
            raise ValueError(f"learning_rate must be in (0, 1], got {learning_rate}")
        
        # Validate discount factor
        if not (0 <= discount_factor <= 1):
            raise ValueError(f"discount_factor must be in [0, 1], got {discount_factor}")
        
        # Validate epsilon
        if not (0 <= epsilon <= 1):
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
        
        # Validate epsilon_min
        if not (0 <= epsilon_min <= 1):
            raise ValueError(f"epsilon_min must be in [0, 1], got {epsilon_min}")
        
        # Validate epsilon_decay
        if epsilon_decay <= 0:
            raise ValueError(f"epsilon_decay must be positive, got {epsilon_decay}")
        
        # Validate epsilon >= epsilon_min
        if epsilon < epsilon_min:
            raise ValueError(f"epsilon ({epsilon}) must be >= epsilon_min ({epsilon_min})")
        
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table: Dict[Tuple, Dict[Any, float]] = {}
        
    def get_q_value(self, state: Tuple, action: Any) -> float:
        """
        Get the Q-value for a given state-action pair.
        
        Args:
            state: The current state (must be hashable)
            action: The action to take
            
        Returns:
            The Q-value for the state-action pair
        """
        if state not in self.q_table:
            self.q_table[state] = {}
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0.0
        return self.q_table[state][action]
    
    def get_max_q_value(self, state: Tuple, possible_actions: list) -> float:
        """
        Get the maximum Q-value for a given state across all possible actions.
        
        Args:
            state: The current state
            possible_actions: List of possible actions from this state
            
        Returns:
            The maximum Q-value for the state
        """
        if not possible_actions:
            return 0.0
        return max([self.get_q_value(state, action) for action in possible_actions])
    
    def get_best_action(self, state: Tuple, possible_actions: list) -> Any:
        """
        Get the best action for a given state based on current Q-values.
        
        Args:
            state: The current state
            possible_actions: List of possible actions from this state
            
        Returns:
            The action with the highest Q-value
        """
        if not possible_actions:
            return None
        
        q_values = {action: self.get_q_value(state, action) for action in possible_actions}
        max_q = max(q_values.values())
        # Return randomly among best actions if there are ties
        best_actions = [action for action, q in q_values.items() if q == max_q]
        return np.random.choice(best_actions)
    
    def choose_action(self, state: Tuple, possible_actions: list) -> Any:
        """
        Choose an action using epsilon-greedy policy.
        
        With probability epsilon, choose a random action (exploration).
        Otherwise, choose the best action based on Q-values (exploitation).
        
        Args:
            state: The current state
            possible_actions: List of possible actions from this state
            
        Returns:
            The chosen action
        """
        if not possible_actions:
            return None
            
        # Exploration: random action
        if np.random.random() < self.epsilon:
            return np.random.choice(possible_actions)
        
        # Exploitation: best action
        return self.get_best_action(state, possible_actions)
    
    def update(
        self,
        state: Tuple,
        action: Any,
        reward: float,
        next_state: Tuple,
        possible_next_actions: list,
        done: bool = False
    ) -> None:
        """
        Update the Q-value for a state-action pair using the Q-learning update rule.
        
        Q(s, a) = Q(s, a) + α * [r + γ * max(Q(s', a')) - Q(s, a)]
        
        Args:
            state: The current state
            action: The action taken
            reward: The reward received
            next_state: The next state after taking the action
            possible_next_actions: List of possible actions from the next state
            done: Whether the episode is finished
        """
        current_q = self.get_q_value(state, action)
        
        if done:
            # If episode is done, there's no future reward
            target_q = reward
        else:
            # Calculate target Q-value using Bellman equation
            max_next_q = self.get_max_q_value(next_state, possible_next_actions)
            target_q = reward + self.discount_factor * max_next_q
        
        # Update Q-value
        new_q = current_q + self.learning_rate * (target_q - current_q)
        
        # Store updated Q-value
        if state not in self.q_table:
            self.q_table[state] = {}
        self.q_table[state][action] = new_q
    
    def decay_epsilon(self) -> None:
        """
        Decay the exploration rate epsilon.
        
        Should be called at the end of each episode to gradually
        shift from exploration to exploitation.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filename: str) -> None:
        """
        Save the Q-table and agent parameters to a file.
        
        Args:
            filename: Path to the file where the agent should be saved
        """
        agent_data = {
            'q_table': self.q_table,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min
        }
        with open(filename, 'wb') as f:
            pickle.dump(agent_data, f)
    
    def load(self, filename: str) -> None:
        """
        Load the Q-table and agent parameters from a file.
        
        WARNING: Loading pickle files from untrusted sources is a security risk.
        Only load files that you have created yourself or trust completely.
        
        Args:
            filename: Path to the file containing the saved agent
        """
        import warnings
        warnings.warn(
            "Loading pickle files from untrusted sources is a security risk. "
            "Only load files you trust.",
            UserWarning
        )
        
        with open(filename, 'rb') as f:
            agent_data = pickle.load(f)
        
        self.q_table = agent_data['q_table']
        self.learning_rate = agent_data['learning_rate']
        self.discount_factor = agent_data['discount_factor']
        self.epsilon = agent_data['epsilon']
        self.epsilon_decay = agent_data['epsilon_decay']
        self.epsilon_min = agent_data['epsilon_min']
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the agent's learning progress.
        
        Returns:
            Dictionary containing agent statistics
        """
        num_states = len(self.q_table)
        total_state_action_pairs = sum(len(actions) for actions in self.q_table.values())
        
        return {
            'num_states_visited': num_states,
            'total_state_action_pairs': total_state_action_pairs,
            'current_epsilon': self.epsilon,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor
        }
    
    def __repr__(self) -> str:
        """String representation of the agent."""
        return (f"QAgent(learning_rate={self.learning_rate}, "
                f"discount_factor={self.discount_factor}, "
                f"epsilon={self.epsilon:.3f})")
