# UCT Monte Carlo Tree Search (MCTS) for OpenAI Gym's FrozenLake

This project implements the Upper Confidence Bound for Trees (UCT) variant of Monte Carlo Tree Search (MCTS) to solve the FrozenLake environment from OpenAI Gym [2]. The agent learns to navigate a grid world and reach a goal state while avoiding holes. The implementation is based on the UCT algorithm described in the paper "Bandit based Monte-Carlo Planning" by Kocsis and Szepesvári [1].

## Features

- Customizable grid size (4x4, 8x8) and custom maps
- Stochastic environment simulation for more robust planning
- Heuristic evaluation function to guide the search
- Visualization of the agent's performance over multiple episodes
- Verbose mode for detailed output during the learning process

## Requirements

- Python 3.6+
- OpenAI Gym
- NumPy
- Matplotlib

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/jaybarrett34/MonteCarloTreeSearch.git
   cd MonteCarloTreeSearch
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

2. `mrun_mcts.py`: This script runs multiple episodes of the UCT MCTS agent and provides a comprehensive analysis of the agent's performance. You will be prompted to choose a map (4x4, 8x8, or custom) and the number of episodes to run.

   If you choose 'custom', the script will look for a file named `custom_map.csv` in the same directory. The file should contain the custom map layout, where each character represents a cell in the grid:
   - 'S': Start position
   - 'F': Frozen (safe) cell
   - 'H': Hole (terminal state)
   - 'G': Goal position

   Example `custom_map.csv`:
   ```
   SFFF
   FHFH
   FFFH
   HFFG
   ```

   If the `custom_map.csv` file is not found, the script will use the default 4x4 map.

   After selecting the map, enter the number of episodes to run. The agent will start learning and display its progress. If the `-v` or `--verbose` flag is provided, the script will display detailed output for each step, including the current state, chosen action, next state, reward, and grid visualization.

   After the specified number of episodes, the script will display the success rate, average number of steps, average reward, and a plot of the agent's performance over the episodes.

   To run this script, use:
   ```
   python mrun_mcts.py [-v]
   ```

## Customization

To create a custom map, follow these steps:

1. Create a file named `custom_map.csv` in the same directory as the scripts.

2. In the `custom_map.csv` file, define your custom map layout using the following characters:
   - 'S': Start position (only one allowed)
   - 'F': Frozen (safe) cell
   - 'H': Hole (terminal state)
   - 'G': Goal position (only one allowed)

   Each character represents a cell in the grid. The map should be rectangular, and all rows should have the same length.

3. Save the `custom_map.csv` file.

4. Run the `mrun_mcts.py` script and choose the 'custom' map option when prompted.

Note: Make sure that the custom map has a valid path from the start position to the goal position.

## How it Works

The UCT MCTS algorithm builds a search tree by iteratively selecting and expanding promising nodes based on the UCB1 formula [1]. The algorithm balances exploitation (selecting actions with high estimated value) and exploration (trying less visited actions) to find the best path to the goal state.

The main steps of the algorithm are:

1. **Selection**: Start from the root node and recursively select the child node with the highest UCB1 score until a leaf node is reached.

2. **Expansion**: If the selected leaf node is not a terminal state, create one or more child nodes representing the available actions.

3. **Simulation**: From the expanded node, perform a random rollout (simulation) until a terminal state is reached or a maximum depth is exceeded. The reward is accumulated during the rollout.

4. **Backpropagation**: Propagate the accumulated reward back through the selected nodes in the tree, updating their estimated values and visit counts.

These steps are repeated for a specified number of iterations (`max_iterations`) to build the search tree. After the iterations, the action with the highest estimated value at the root node is selected as the best action to take.

The project also includes an evaluation function (`evaluate`) that guides the search by estimating the value of a state based on its distance to the goal, proximity to holes, and a penalty for revisiting states.

## License

This project is open-source and available under the [MIT License](LICENSE).

## References

[1] L. Kocsis and C. Szepesvári, "Bandit based Monte-Carlo Planning," in European Conference on Machine Learning (ECML), 2006, pp. 282-293. http://ggp.stanford.edu/readings/uct.pdf

[2] OpenAI Gym FrozenLake environment: https://www.gymlibrary.dev/environments/toy_text/frozen_lake/