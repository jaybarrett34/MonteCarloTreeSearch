# UCT Monte Carlo Tree Search (MCTS) for OpenAI Gym's FrozenLake

This project implements the Upper Confidence Bound for Trees (UCT) variant of Monte Carlo Tree Search (MCTS) to solve the FrozenLake environment from OpenAI Gym. The agent learns to navigate a grid world and reach a goal state while avoiding holes.

## Features

- Customizable grid size (4x4, 8x8) and custom maps
- Stochastic environment simulation for more robust planning
- Heuristic evaluation function to guide the search
- Visualization of the agent's performance over multiple episodes

## Requirements

- Python 3.6+
- OpenAI Gym
- NumPy
- Matplotlib

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/uct-mcts-frozenlake.git
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

There are two main scripts in this project:

1. `run_mcts.py`: This script runs a single episode of the UCT MCTS agent on the FrozenLake environment. It displays the grid, chosen actions, and rewards at each step. To run this script, use:
   ```
   python run_mcts.py
   ```

2. `mrun_mcts.py`: This script runs multiple episodes of the UCT MCTS agent and provides a more comprehensive analysis of the agent's performance. You will be prompted to choose a map (4x4, 8x8, or custom) and the number of episodes to run. The agent will then start learning and display its progress. After the specified number of episodes, the script will display the success rate, average steps, and average reward, as well as a plot of the agent's performance over the episodes. To run this script, use:
   ```
   python mrun_mcts.py
   ```

You can also add the `-v` or `--verbose` flag to `mrun_mcts.py` for more detailed output during the learning process.

## How it Works

The UCT MCTS algorithm works by building a search tree incrementally and biasing its growth towards promising branches. Each node in the tree represents a state, and each edge represents an action leading to a new state.

The algorithm follows four main steps:

1. **Selection**: Starting from the root node, the algorithm selects the most promising child node based on the UCT score, which balances exploitation (choosing the best-performing action) and exploration (trying less-visited actions).

2. **Expansion**: When a leaf node is reached, the algorithm expands the tree by adding one or more child nodes corresponding to the available actions.

3. **Simulation**: From the newly added node(s), the algorithm performs a random rollout (or playout) until a terminal state is reached or a maximum depth is exceeded. The reward is then backpropagated to update the statistics of the nodes along the path.

4. **Backpropagation**: The reward obtained from the simulation is used to update the Q-values and visit counts of the nodes along the path from the expanded node to the root.

These steps are repeated for a specified number of iterations or until a time budget is exhausted. The action with the highest Q-value at the root node is then selected as the best action to take in the current state.

## License

This project is open-source and available under the [MIT License](LICENSE).