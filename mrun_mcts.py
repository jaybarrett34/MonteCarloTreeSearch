from model import MonteCarloTreeSearch
from environment_wrapper import EnvironmentWrapper, SimulatorWrapper
import random
import gym
import matplotlib.pyplot as plt
import time
import argparse

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

def run_episode(env, simulator, max_iterations, verbose=False):
    mcts = MonteCarloTreeSearch(env=env, simulator=simulator)
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0
    start_time = time.time()

    action_names = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}

    if verbose:
        print("\nStarting new episode")
        print_grid(env, state)

    while not done:
        if verbose:
            print(f"\nStep {steps}, Current State: {state}")
        
        action = mcts.monte_carlo_planning(state, max_iterations=max_iterations)
        action_name = action_names[action]
        if verbose:
            print(f"Chosen action: {action} ({action_name})")
        
        env.set_state(state) 
        next_state, reward = env.take_action(action)
        if verbose:
            print(f"Next State: {next_state}, Reward: {reward}")
            print_grid(env, next_state)
        
        total_reward += reward
        done = env.is_terminal(next_state)
        state = next_state
        steps += 1

        if steps > 100: 
            if verbose:
                print("Episode too long. Terminating.")
            break

    if verbose:
        print("\nFinal state:")
        print_grid(env, state)
        print(f"\nEpisode ended after {steps} steps.")
        print(f"Total reward: {total_reward}")

    episode_time = time.time() - start_time
    return env.is_goal(state), steps, total_reward, episode_time, state

def plot_results(episodes, successes, steps_list, rewards_list, times_list):
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    
    # Success rate
    cumulative_successes = [sum(successes[:i+1]) for i in range(len(successes))]
    success_rates = [s / (i + 1) for i, s in enumerate(cumulative_successes)]
    axs[0, 0].plot(episodes, success_rates)
    axs[0, 0].set_title('Success Rate over Episodes')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Success Rate')

    # Steps per episode
    axs[0, 1].plot(episodes, steps_list)
    axs[0, 1].set_title('Steps per Episode')
    axs[0, 1].set_xlabel('Episode')
    axs[0, 1].set_ylabel('Steps')

    # Reward per episode
    axs[1, 0].plot(episodes, rewards_list)
    axs[1, 0].set_title('Reward per Episode')
    axs[1, 0].set_xlabel('Episode')
    axs[1, 0].set_ylabel('Reward')

    # Time per episode
    axs[1, 1].plot(episodes, times_list)
    axs[1, 1].set_title('Time Taken per Episode')
    axs[1, 1].set_xlabel('Episode')
    axs[1, 1].set_ylabel('Time (seconds)')

    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Run MCTS on FrozenLake environment")
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity")
    args = parser.parse_args()

    random.seed(42)
    
    maps = {
        '4x4': 'FrozenLake-v1',
        '8x8': 'FrozenLake8x8-v1',
        'custom': None 
    }

    print("Available maps:")
    for key in maps.keys():
        print(f"- {key}")
    map_choice = input("Choose a map (4x4, 8x8, or custom): ").lower()

    if map_choice not in maps:
        print("Invalid choice. Using default 4x4 map.")
        map_choice = '4x4'

    custom_map = None
    if map_choice == 'custom':
        custom_map = [
            'SFFF',
            'FFFF',
            'FFFF',
            'HFFG'
        ]

    env = EnvironmentWrapper(env_name='FrozenLake-v1', is_slippery=False, custom_map=custom_map)
    simulator = SimulatorWrapper(env_name='FrozenLake-v1', is_slippery=False, custom_map=custom_map)

    print("\nInitial grid state:")
    print_grid(env, env.reset())

    max_iterations = 5000
    num_episodes = int(input("Enter the number of episodes to run: "))

    successes = []
    steps_list = []
    rewards_list = []
    times_list = []
    final_state = None

    for episode in range(num_episodes):
        success, steps, reward, episode_time, state = run_episode(env, simulator, max_iterations, verbose=args.verbose)
        successes.append(int(success))
        steps_list.append(steps)
        rewards_list.append(reward)
        times_list.append(episode_time)
        final_state = state
        
        print(f"Episode {episode + 1}: {'Success' if success else 'Failure'}")

    success_rate = sum(successes) / num_episodes * 100
    avg_steps = sum(steps_list) / num_episodes
    avg_reward = sum(rewards_list) / num_episodes

    print(f"\n{GREEN if success_rate > 50 else RED}Results:{RESET}")
    print(f"Success rate: {success_rate:.2f}%")
    print(f"Average steps: {avg_steps:.2f}")
    print(f"Average reward: {avg_reward:.2f}")

    print("\nFinal grid state:")
    print_grid(env, final_state)

    plot_results(range(1, num_episodes + 1), successes, steps_list, rewards_list, times_list)

if __name__ == "__main__":
    main()