# Summary: What is `from src.q_agent import QAgent`?

## Quick Answer

`from src.q_agent import QAgent` is a Python import statement that brings in a **Q-Learning Agent** class - a reinforcement learning implementation that can learn to make optimal decisions through trial and error.

## What This Repository Contains

This repository now includes a complete Q-Learning implementation with the following files:

### 1. Core Implementation (`src/`)
- **`src/q_agent.py`**: The main QAgent class with full Q-learning algorithm
- **`src/__init__.py`**: Package initialization for clean imports

### 2. Documentation
- **`QAGENT_README.md`**: Comprehensive documentation covering:
  - What Q-learning is and how it works
  - Complete API reference
  - Usage examples
  - Applications and use cases

### 3. Example & Dependencies
- **`example_usage.py`**: Working demonstration on a 4x4 grid world problem
- **`requirements.txt`**: Required dependencies (NumPy)

### 4. Configuration
- **`.gitignore`**: Python project gitignore configuration

## How to Use It

### Installation
```bash
pip install -r requirements.txt
```

### Basic Usage
```python
from src.q_agent import QAgent

# Create an agent
agent = QAgent(
    learning_rate=0.1,      # How fast the agent learns
    discount_factor=0.95,   # How much it values future rewards
    epsilon=1.0,            # Initial exploration rate
    epsilon_decay=0.995     # How quickly exploration decreases
)

# In your training loop:
action = agent.choose_action(state, possible_actions)
next_state, reward, done = environment.step(action)
agent.update(state, action, reward, next_state, possible_next_actions, done)
agent.decay_epsilon()  # Call at end of each episode
```

### Running the Example
```bash
python example_usage.py
```

This trains an agent to navigate a 4x4 grid world from (0,0) to (3,3) using Q-learning.

## What is Q-Learning?

Q-Learning is a model-free reinforcement learning algorithm that:
1. **Learns from experience**: The agent tries actions and learns from the results
2. **Doesn't need a model**: No prior knowledge of the environment is required
3. **Finds optimal policies**: Converges to the best action for each state
4. **Uses the Bellman equation**: Q(s,a) = Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]

## Key Features

✓ **Parameter validation**: Ensures learning parameters are in valid ranges  
✓ **Epsilon-greedy exploration**: Balances exploration vs exploitation  
✓ **Model persistence**: Save and load trained agents  
✓ **Learning statistics**: Track progress during training  
✓ **Well-documented**: Comprehensive docstrings and README  
✓ **Security-conscious**: Warnings for potentially unsafe operations  

## Applications

The QAgent can be applied to:
- Game AI (board games, video games)
- Robot navigation and path planning
- Resource optimization
- Trading strategies
- Control systems
- Any sequential decision-making problem

## Test Results

The implementation has been tested and successfully:
- ✓ Validates all input parameters
- ✓ Trains agents to solve navigation problems
- ✓ Achieves optimal policies (6 steps to goal in 4x4 grid)
- ✓ Passes security scanning (0 vulnerabilities)

## For More Information

- **Complete API documentation**: See `QAGENT_README.md`
- **Working example**: Run `python example_usage.py`
- **Technical details**: See docstrings in `src/q_agent.py`

---

**Author**: Paula Vázquez  
**Part of**: Personal data science portfolio  
**Related projects**: See [Paula's GitHub profile](https://github.com/paulavazq)
