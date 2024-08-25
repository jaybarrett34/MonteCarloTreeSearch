from model import MonteCarloTreeSearch
from environment_wrapper import EnvironmentWrapper, SimulatorWrapper
import random
# import pdb; pdb.set_trace()

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

def print_grid(env, current_state):
    grid = [['.' for _ in range(env.ncol)] for _ in range(env.nrow)]
    
    for hole in env.hole_states:
        row, col = env.index_to_pos(hole)
        grid[row][col] = 'H'

    goal_row, goal_col = env.index_to_pos(env.goal_state)
    grid[goal_row][goal_col] = 'G'
    
    current_row, current_col = env.index_to_pos(current_state)
    grid[current_row][current_col] = 'A'

    for row in grid:
        print(' '.join(row))
    print()

def main():
    random.seed(1)
    env = EnvironmentWrapper()
    simulator = SimulatorWrapper()
    mcts = MonteCarloTreeSearch(env=env, simulator=simulator)
    max_iterations = 5000

    state = env.reset()
    done = False
    total_reward = 0
    steps = 0

    action_names = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}

    while not done:
        print(f"\nStep {steps}, Current State: {state}")
        print_grid(env, state)
        
        action = mcts.monte_carlo_planning(state, max_iterations=max_iterations)
        action_name = action_names[action]
        print(f"Chosen action: {action} ({action_name})")
        
        env.set_state(state) 
        next_state, reward = env.take_action(action)
        print(f"Next State: {next_state}, Reward: {reward}")
        
        total_reward += reward
        done = env.is_terminal(next_state)
        state = next_state
        steps += 1

        if steps > 100:  # Increase the step limit
            print("Episode too long. Terminating.")
            break

    print("\nFinal state:")
    print_grid(env, state)
    print(f"\nEpisode ended after {steps} steps.")
    print(f"Total reward: {total_reward}")

    if env.is_goal(state):
        print(f"{GREEN}Goal state reached!{RESET}")
    else:
        print(f"{RED}Goal state not reached.{RESET}")

if __name__ == "__main__":
    main()