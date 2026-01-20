# QAgent Module

A Python implementation of a **Q-Learning agent** for reinforcement learning tasks.

## What is QAgent?

`QAgent` is a class that implements the **Q-Learning algorithm**, a fundamental model-free reinforcement learning technique. Q-Learning enables an agent to learn optimal decision-making policies through trial and error by interacting with an environment.

### Key Concepts

**Q-Learning** is based on learning an action-value function Q(s, a) that represents the expected future reward for taking action `a` in state `s`. The agent learns by updating Q-values using the Bellman equation:

```
Q(s, a) = Q(s, a) + α * [r + γ * max(Q(s', a')) - Q(s, a)]
```

Where:
- `s`: current state
- `a`: action taken
- `r`: reward received
- `s'`: next state
- `α`: learning rate (alpha)
- `γ`: discount factor (gamma)

## Installation

The module requires NumPy. Install it with:

```bash
pip install numpy
```

## Usage

### Basic Example

```python
from src.q_agent import QAgent

# Create a Q-Learning agent
agent = QAgent(
    learning_rate=0.1,      # How fast the agent learns
    discount_factor=0.95,   # How much the agent values future rewards
    epsilon=1.0,            # Initial exploration rate
    epsilon_decay=0.995,    # How quickly exploration decreases
    epsilon_min=0.01        # Minimum exploration rate
)

# Training loop (simplified)
for episode in range(1000):
    state = env.reset()
    done = False
    
    while not done:
        # Choose an action using epsilon-greedy policy
        action = agent.choose_action(state, possible_actions)
        
        # Take action in environment
        next_state, reward, done = env.step(action)
        
        # Update Q-values
        agent.update(state, action, reward, next_state, 
                    possible_next_actions, done)
        
        state = next_state
    
    # Decay exploration rate
    agent.decay_epsilon()
```

### Running the Example

A complete working example is provided in `example_usage.py` that demonstrates training a Q-Learning agent on a simple 4x4 grid world:

```bash
python example_usage.py
```

This example shows:
- How to create and configure a QAgent
- Training the agent over multiple episodes
- Tracking learning progress
- Testing the trained agent
- Saving and loading trained models

## Features

### Core Methods

- **`choose_action(state, possible_actions)`**: Select an action using epsilon-greedy policy
- **`update(state, action, reward, next_state, possible_next_actions, done)`**: Update Q-values based on experience
- **`get_q_value(state, action)`**: Retrieve Q-value for a state-action pair
- **`get_best_action(state, possible_actions)`**: Get the optimal action based on current Q-values
- **`decay_epsilon()`**: Reduce exploration rate over time
- **`save(filename)`**: Save the trained Q-table to disk
- **`load(filename)`**: Load a previously saved Q-table
- **`get_stats()`**: Get learning statistics

### Parameters

- **`learning_rate`** (α): Controls how quickly the agent updates its Q-values (0 < α ≤ 1)
  - Higher values: faster learning but less stable
  - Lower values: slower but more stable learning
  
- **`discount_factor`** (γ): Determines importance of future rewards (0 ≤ γ ≤ 1)
  - γ = 0: only immediate rewards matter
  - γ = 1: future rewards are equally important as immediate rewards
  
- **`epsilon`**: Exploration rate for epsilon-greedy policy (0 ≤ ε ≤ 1)
  - ε = 1: fully random (exploration)
  - ε = 0: fully greedy (exploitation)
  
- **`epsilon_decay`**: Rate of epsilon reduction per episode
  
- **`epsilon_min`**: Minimum value for epsilon

## Applications

Q-Learning and the QAgent can be applied to various problems:

- **Game AI**: Teaching agents to play games (chess, tic-tac-toe, etc.)
- **Robotics**: Navigation and path planning
- **Resource Management**: Optimizing resource allocation
- **Control Systems**: Learning optimal control policies
- **Finance**: Portfolio optimization and trading strategies

## Technical Details

### Exploration vs. Exploitation

The agent uses an **epsilon-greedy policy**:
- With probability ε: take a random action (exploration)
- With probability 1-ε: take the best known action (exploitation)

As training progresses, ε typically decreases, shifting from exploration to exploitation.

### State and Action Representation

- States must be **hashable** (tuples, strings, numbers)
- Actions can be any type (strings, numbers, tuples)
- Q-table is stored as a nested dictionary: `{state: {action: q_value}}`

### Convergence

Q-Learning is proven to converge to the optimal policy Q* under certain conditions:
1. All state-action pairs are visited infinitely often
2. Learning rate decreases appropriately over time
3. Sufficient exploration is maintained

## Author

Paula Vázquez - Data Scientist with expertise in machine learning and bioinformatics

## License

This module is part of Paula's portfolio showcasing data science and machine learning skills.

## Related Projects

Check out other machine learning projects:
- [ECG Anomaly Detection with Autoencoders](https://github.com/paulavazq/ecg-anomaly-detection-autoencoders)
- [RNA-seq Prostate Cancer Analysis](https://github.com/paulavazq/Final_Project_RNAseq)
- [Audio Classification with Deep Learning](https://github.com/paulavazq/Module-6_Music-Speach)

## References

- Watkins, C.J.C.H. (1989). Learning from Delayed Rewards (PhD thesis)
- Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction
