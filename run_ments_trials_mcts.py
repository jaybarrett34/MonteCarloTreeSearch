from ments import MaximumEntropyTreeSearch
import random
from gym.envs.registration import register
import gym
from utilities.tree import Tree
import matplotlib.pyplot as plt

def init_env():
    register(
        id='FrozenLakeNotSlippery-v0',
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name': '4x4', 'is_slippery': False}
    )
    return gym.make('FrozenLakeNotSlippery-v0')

def print_grid(env):
    desc = env.unwrapped.desc
    color_map = {
        b'S': '\033[92m',  
        b'G': '\033[93m',
        b'F': '\033[94m', 
        b'H': '\033[91m',  
    }
    reset_color = '\033[0m'
    cell_width = 6 

    for i, row in enumerate(desc):
        for j, col in enumerate(row):
            color = color_map.get(col, reset_color)
            position = i * len(row) + j
            print(f"{color}{col.decode('utf-8')}[{position:2}]{reset_color}".ljust(cell_width), end=' ')
        print()

def run_experiment(temperature, printGrid, steps=100000):
    random.seed(1)
    env = init_env()
    tree = Tree()
    maximumEntropyTreeSearch = MaximumEntropyTreeSearch(env=env, tree=tree, temperature=temperature)
    goals_reached = 0

    if printGrid == True:
        print("Final Environment Grid:")
        print_grid(env)

    for _ in range(steps):
        env.reset()
        node = maximumEntropyTreeSearch.tree_policy(maximumEntropyTreeSearch.tree.root)
        reward = maximumEntropyTreeSearch.default_policy(node)
        if node.state == 15 and reward > 0:
            goals_reached += 1
        maximumEntropyTreeSearch.backward(node, reward)

    maximumEntropyTreeSearch.forward()

    print(f"Temperature: {temperature}, Goals Reached: {goals_reached}")

    return goals_reached

def main():
    temperatures = [i/10.0 for i in range(1, 11)]
    results = []
    print_once = True

    for temp in temperatures:
        goals_reached = run_experiment(temp, print_once)
        print_once = False
        results.append((temp, goals_reached))

    # Plot results
    temps, goals = zip(*results)
    plt.plot(temps, goals)
    plt.xlabel('Temperature')
    plt.ylabel('Goals Reached')
    plt.title('Goals Reached vs Temperature')
    plt.show()

if __name__ == "__main__":
    main()
