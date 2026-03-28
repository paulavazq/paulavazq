"""
Example: Q-Learning Agent on a Simple Grid World

This example demonstrates how to use the QAgent class to solve
a simple grid world navigation problem using Q-learning.
"""

from src.q_agent import QAgent
import numpy as np


class SimpleGridWorld:
    """
    A simple 4x4 grid world environment.
    
    The agent starts at (0, 0) and needs to reach the goal at (3, 3).
    The agent receives a reward of +10 for reaching the goal,
    -1 for each step, and the episode ends when the goal is reached.
    """
    
    MAX_STEPS_PER_EPISODE = 100  # Maximum steps allowed per episode
    
    def __init__(self, size=4):
        self.size = size
        self.reset()
        
    def reset(self):
        """Reset the environment to the initial state."""
        self.position = (0, 0)
        return self.position
    
    def get_possible_actions(self):
        """Get list of possible actions from current position."""
        return ['up', 'down', 'left', 'right']
    
    def step(self, action):
        """
        Take an action and return (next_state, reward, done).
        
        Args:
            action: One of 'up', 'down', 'left', 'right'
            
        Returns:
            tuple: (next_state, reward, done)
        """
        row, col = self.position
        
        # Calculate new position based on action
        if action == 'up':
            row = max(0, row - 1)
        elif action == 'down':
            row = min(self.size - 1, row + 1)
        elif action == 'left':
            col = max(0, col - 1)
        elif action == 'right':
            col = min(self.size - 1, col + 1)
        
        self.position = (row, col)
        
        # Check if goal is reached
        goal = (self.size - 1, self.size - 1)
        if self.position == goal:
            return self.position, 10.0, True
        else:
            return self.position, -1.0, False


def train_agent(episodes=500):
    """
    Train a Q-learning agent on the grid world.
    
    Args:
        episodes: Number of training episodes
        
    Returns:
        Trained QAgent
    """
    env = SimpleGridWorld(size=4)
    agent = QAgent(
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    rewards_per_episode = []
    
    print("Training Q-Learning Agent on Grid World...")
    print(f"Training for {episodes} episodes")
    print("-" * 50)
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < SimpleGridWorld.MAX_STEPS_PER_EPISODE:
            # Choose action
            possible_actions = env.get_possible_actions()
            action = agent.choose_action(state, possible_actions)
            
            # Take action
            next_state, reward, done = env.step(action)
            total_reward += reward
            steps += 1
            
            # Update Q-values
            possible_next_actions = env.get_possible_actions()
            agent.update(state, action, reward, next_state, possible_next_actions, done)
            
            state = next_state
        
        # Decay epsilon
        agent.decay_epsilon()
        rewards_per_episode.append(total_reward)
        
        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(f"Episode {episode + 1}/{episodes} - "
                  f"Avg Reward: {avg_reward:.2f} - "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    print("-" * 50)
    print("Training completed!")
    print(f"Final average reward (last 100 episodes): {np.mean(rewards_per_episode[-100:]):.2f}")
    
    # Display agent statistics
    stats = agent.get_stats()
    print(f"\nAgent Statistics:")
    print(f"  States visited: {stats['num_states_visited']}")
    print(f"  State-action pairs learned: {stats['total_state_action_pairs']}")
    print(f"  Final epsilon: {stats['current_epsilon']:.3f}")
    
    return agent


def test_agent(agent, num_tests=5):
    """
    Test the trained agent.
    
    Args:
        agent: Trained QAgent
        num_tests: Number of test episodes
    """
    env = SimpleGridWorld(size=4)
    
    print(f"\n{'=' * 50}")
    print("Testing Trained Agent (Greedy Policy)")
    print('=' * 50)
    
    # Temporarily set epsilon to 0 for greedy policy
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    for test in range(num_tests):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False
        path = [state]
        
        while not done and steps < SimpleGridWorld.MAX_STEPS_PER_EPISODE:
            possible_actions = env.get_possible_actions()
            action = agent.choose_action(state, possible_actions)
            next_state, reward, done = env.step(action)
            
            path.append(next_state)
            total_reward += reward
            steps += 1
            state = next_state
        
        print(f"\nTest {test + 1}:")
        print(f"  Steps taken: {steps}")
        print(f"  Total reward: {total_reward}")
        print(f"  Path: {' -> '.join([str(p) for p in path])}")
        print(f"  Goal reached: {'Yes' if done else 'No'}")
    
    # Restore original epsilon
    agent.epsilon = original_epsilon


if __name__ == "__main__":
    # Train the agent
    agent = train_agent(episodes=500)
    
    # Test the agent
    test_agent(agent, num_tests=5)
    
    # Optionally save the trained agent
    # agent.save('trained_agent.pkl')
    print("\n" + "=" * 50)
    print("Example completed!")
    print("You can now import and use QAgent in your own projects:")
    print("  from src.q_agent import QAgent")
    print("=" * 50)
