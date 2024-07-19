from uct import MonteCarloTreeSearch
import random
from gym.envs.registration import register
import gym
from utilities.tree import Tree

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

def main():
    random.seed()
    env = init_env()
    tree = Tree()
    monteCarloTreeSearch = MonteCarloTreeSearch(env=env, tree=tree)
    steps = 100000

    for _ in range(0, steps):
        env.reset()
        node = monteCarloTreeSearch.tree_policy()
        reward = monteCarloTreeSearch.default_policy(node)
        monteCarloTreeSearch.backward(node, reward)

    # monteCarloTreeSearch.tree.show()
    monteCarloTreeSearch.forward()
    print("Final Environment Grid:")
    print_grid(env)

if __name__ == "__main__":
   main()
