import gymnasium as gym
import numpy as np
from utilities.node import Node
from utilities.tree import Tree
from math import sqrt, log
from termcolor import colored

ACTION_MAP = {
    0: 'left',
    1: 'down',
    2: 'right',
    3: 'up'
}

class UCT:
    def __init__(self, env, default_policy, max_depth, max_iter):
        self.env = env
        self.default_policy = default_policy
        self.max_depth = max_depth
        self.max_iter = max_iter
        self.tree = Tree()
        self.action_space = env.action_space.n

    # I know that this has the problem altogether. I can't quite debug how
    def find_best_action(self, state):
        root = Node(state=state, action=None, action_space=self.action_space, reward=0, terminal=False)
        self.tree.add_node(root)

        for i in range(self.max_iter):
            # if i % 100 == 0:
            #     print(f"Iteration: {i}")
            leaf_node = self.tree_policy(root)
            reward = self.default_policy(self.env, leaf_node.state)
            self.backpropagate(leaf_node, reward)

        best_child = self.best_child(root, exploration_constant=0)
        return best_child.action

    # Potential issue spot
    def tree_policy(self, node):
        depth = 0
        while depth < self.max_depth and not node.terminal:
            if self.tree.is_expandable(node):
                return self.expand(node)
            else:
                node = self.best_child(node, exploration_constant=sqrt(2))
            self.env.env.s = node.state
            result = self.env.step(node.action)
            if len(result) == 4:
                state, reward, done, info = result
            else:
                state, reward, done, truncated, info = result
            node.state = state
            if done:
                node.terminal = True
            assert node.state == state, f"Node state {node.state} does not match environment state {state}"
            depth += 1
        return node

    def expand(self, node):
        action = node.untried_action()
        self.env.env.s = node.state
        result = self.env.step(action)
        if len(result) == 4:
            state, reward, done, info = result
        else:
            state, reward, done, truncated, info = result
        reward = 1 if state == 15 else 0
        new_node = Node(state=state, action=action, action_space=self.action_space, reward=reward, terminal=done)
        self.tree.add_node(new_node, node)
        return new_node

    def compute_value(self, parent, child, exploration_constant):
        exploitation_term = child.total_simulation_reward / child.num_visits
        exploration_term = exploration_constant * sqrt(2 * log(parent.num_visits) / child.num_visits)
        return exploitation_term + exploration_term

    # Potential issue spot
    def best_child(self, node, exploration_constant):
        children = self.tree.children(node)
        best_child = children[0]
        best_value = self.compute_value(node, best_child, exploration_constant)
        for child in children[1:]:
            value = self.compute_value(node, child, exploration_constant)
            if value > best_value:
                best_child = child
                best_value = value
        return best_child

    # Potential issue spot
    def backpropagate(self, node, reward):
        while node:
            node.num_visits += 1
            node.total_simulation_reward += reward
            parent = self.tree.parent(node)
            node = parent

def default_policy(env, state):
    env.reset()
    env.env.s = state
    done = False
    max_steps = 100
    for _ in range(max_steps):
        action = env.action_space.sample()
        state, _, done, _, _ = env.step(action)
        if state == 15:  # Goal state
            return 1
        if done:
            break
    return 0

# The outer loop portion to basically use the algorithm and get to the final path
# Part of test
def simulate_path_with_mcts(env, alg, start_state=0):
    current_state = start_state
    env.reset()
    env.env.s = current_state
    path = []
    
    while current_state != 15 and not env.unwrapped.desc[current_state // 4][current_state % 4] == b'H':
        action = alg.find_best_action(current_state)
        action_label = ACTION_MAP[action]
        path.append((current_state, action, action_label))
        
        print(f"Current State: {current_state}, Best Action: {action} ({action_label})")
        
        result = env.step(action)
        if len(result) == 4:
            new_state, reward, done, info = result
        else:
            new_state, reward, done, truncated, info = result
        
        print(f"After Action: New State: {new_state}, Reward: {reward}, Done: {done}")
        
        current_state = new_state
        
        if done:
            break
    
    if current_state == 15:
        print("Goal reached!")
    elif env.unwrapped.desc[current_state // 4][current_state % 4] == b'H':
        print("Fell into a hole!")
    else:
        print("Path terminated without reaching goal.")
    
    return path

# For testing ability to find the best action
def test_mcts_at_state(env, alg, test_state):
    env.reset()
    env.env.s = test_state
    action = alg.find_best_action(test_state)
    action_label = ACTION_MAP[action]
    print(f"Test State: {test_state}, Best Action: {action} ({action_label})")
    state, reward, done, _, _ = env.step(action)
    print(f"After Action: State: {state}, Reward: {reward}, Done: {done}")

# Pretty repr
def print_frozen_lake(env):
    desc = env.unwrapped.desc.astype(str)
    row_length = len(desc[0])
    for i, row in enumerate(desc):
        for j, cell in enumerate(row):
            index = i * row_length + j
            if cell == 'S':
                print(colored(f'[{index:2d} {cell}]', 'green'), end=' ')
            elif cell == 'F':
                print(colored(f'[{index:2d} {cell}]', 'cyan'), end=' ')
            elif cell == 'H':
                print(colored(f'[{index:2d} {cell}]', 'red'), end=' ')
            elif cell == 'G':
                print(colored(f'[{index:2d} {cell}]', 'yellow'), end=' ')
        print()

def main():
    # Construct env, generic inits
    env = gym.make("FrozenLake-v1", is_slippery=False)
    print("FrozenLake Map:")
    print_frozen_lake(env)
    max_depth = 10
    max_iter = 5000

    # The layout you suggested, where the algorithm is declared through a class
    # with all appropriate logic in it. I didn't know whether to make default
    # policy part of UCT or not, as I believe the random rollout can be universal
    alg = UCT(env, default_policy, max_depth, max_iter)

    # This is how I've been individually testing the algorithm on a single step
    test_mcts_at_state(env, alg, 0)

    # Testing what the original code was similar to basically.
    simulate_path_with_mcts(env, alg)


    # Test specific states
    # test_states = [0,1,2]
    # for test_state in test_states:
    #     print(f"\nTesting MCTS at State {test_state}")
    #     test_mcts_at_state(env, alg, test_state)

    env.close()

if __name__ == "__main__":
    main()
