from model import MonteCarloTreeSearch
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

def main():
    random.seed(2)
    env = init_env()
    tree = Tree()
    mcts = MonteCarloTreeSearch(env=env, tree=tree)
    max_iterations = 10000
    max_depth = 10

    state = env.reset()
    done = False

    while not done:
        best_action = mcts.monte_carlo_planning(state, max_iterations=max_iterations, max_depth=max_depth)
        print(f"Best action: {best_action}")
        
        next_state, reward, terminated, truncated, _ = env.step(best_action)
        done = terminated or truncated
        
        state = next_state

if __name__ == "__main__":
    main()