import math
import random
from collections import defaultdict
import numpy as np
import pdb

def grid_to_index(row, col, num_cols):
    return row * num_cols + col

def index_to_grid(index, num_cols):
    row = index // num_cols
    col = index % num_cols
    return row, col

def get_valid_children(index, num_rows, num_cols):
    row, col = index_to_grid(index, num_cols)
    children = []

    if col > 0:
        children.append(grid_to_index(row, col - 1, num_cols))
    else:
        children.append(index)
    
    if col < num_cols - 1:
        children.append(grid_to_index(row, col + 1, num_cols))
    else:
        children.append(index)
    
    if row > 0:
        children.append(grid_to_index(row - 1, col, num_cols))
    else:
        children.append(index)
    
    if row < num_rows - 1:
        children.append(grid_to_index(row + 1, col, num_cols))
    else:
        children.append(index)
    
    return children

class MonteCarloTreeSearch:
    def __init__(self, env, simulator):
        self.env = env
        self.simulator = simulator
        self.Q = defaultdict(float)
        self.N = defaultdict(int) 
        self.children = defaultdict(list)
        self.exploration_weight = math.sqrt(2)
        self.max_depth = 200
        self.gamma = 0.95

    def expand(self, state):
        if state not in self.children:
            self.children[state] = [a for a in range(self.env.action_space.n) if self.is_valid_action(state, a)]
            self.N[state] = 0
            for action in self.children[state]:
                self.N[(state, action)] = 0
                self.Q[(state, action)] = 0.0

    def choose_action(self, state):
        if state not in self.children:
            return random.choice(range(self.env.action_space.n))
        
        def score(action):
            if (state, action) not in self.N or self.N[(state, action)] == 0:
                return float('inf')
            exploitation = self.Q[(state, action)] / self.N[(state, action)]
            exploration = self.exploration_weight * math.sqrt(math.log(self.N[state]) / self.N[(state, action)])
            return exploitation + exploration

        return max(range(self.env.action_space.n), key=score)


    def do_rollout(self, node):
        path = self.select(node)
        leaf = path[-1] if isinstance(path[-1], int) else path[-1][0]
        self.expand(leaf)
        reward = self.simulate(leaf)
        self.backpropagate(path, reward)

    def select(self, node):
        path = []
        while True:
            path.append(node)
            if self.env.is_terminal(node) or node not in self.children or not self.children[node]:
                return path
            unexplored = [a for a in range(self.env.action_space.n) if (node, a) not in self.N]
            if unexplored:
                action = random.choice(unexplored)
                path.append((node, action))
                return path
            action = self.choose_action(node)
            path.append((node, action))
            node = self.get_next_state(node, action)

    def update_value(self, state, action, q):
        self.N[state] += 1
        self.N[(state, action)] += 1
        self.Q[(state, action)] += (q - self.Q[(state, action)]) / self.N[(state, action)]

    def get_next_state(self, state, action):
        self.simulator.set_state(state)
        next_state, _ = self.simulator.take_action(action)
        return next_state

    def evaluate(self, state):
        row, col = self.env.index_to_pos(state)
        goal_row, goal_col = self.env.index_to_pos(self.env.goal_state)
        
        distance_to_goal = abs(row - goal_row) + abs(col - goal_col)
        
        hole_penalty = 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.env.nrow and 0 <= nc < self.env.ncol:
                neighbor_state = self.env.pos_to_index((nr, nc))
                if self.env.is_hole(neighbor_state):
                    hole_penalty += 1
        
        max_distance = self.env.nrow + self.env.ncol - 2
        normalized_distance = 1 - (distance_to_goal / max_distance)
        normalized_hole_penalty = hole_penalty / 4
        
        safety_weight = 0.3
        goal_weight = 0.7
        heuristic_value = (goal_weight * normalized_distance - 
                        safety_weight * normalized_hole_penalty)
        
        return heuristic_value

    def simulate(self, state):
        current_state = state
        total_reward = 0
        depth = 0
        
        while not self.simulator.is_terminal(current_state) and depth < self.max_depth:
            valid_actions = [a for a in range(self.env.action_space.n) if self.is_valid_action(current_state, a)]
            action = random.choice(valid_actions)
            
            next_state, reward = self.simulator.take_action(action)
            total_reward += reward * (self.gamma ** depth)
            current_state = next_state
            depth += 1
        
        if not self.simulator.is_terminal(current_state):
            total_reward += self.evaluate(current_state) * (self.gamma ** depth)
        
        return total_reward

    def is_valid_action(self, state, action):
        row, col = self.env.index_to_pos(state)
        nrow, ncol = self.env.nrow, self.env.ncol
        
        if action == 0 and col > 0:
            return True
        elif action == 1 and row < nrow - 1: 
            return True
        elif action == 2 and col < ncol - 1:
            return True
        elif action == 3 and row > 0:  
            return True
        
        return False
    
    def select_action(self, state):
        if state not in self.children or random.random() < 0.3:  # 30% chance of random exploration
            return random.choice([a for a in range(self.env.action_space.n) if self.is_valid_action(state, a)])
        
        return max(self.children[state], key=lambda a: self.uct_score(state, a))
    
    def uct_score(self, state, action):
        if self.N[(state, action)] == 0:
            return float('inf')
        
        exploitation = self.Q[(state, action)] / (self.N[(state, action)] + 1e-8)
        exploration = math.sqrt(2) * math.sqrt(math.log(self.N[state] + 1) / (self.N[(state, action)] + 1e-8))
        return exploitation + exploration

    def search(self, state, depth):
        if self.env.is_terminal(state) or depth >= self.max_depth:
            return self.evaluate(state)
        
        if state not in self.children:
            self.expand(state)
            return self.evaluate(state)
        
        action = self.select_action(state)
        next_state, reward = self.simulate_action(state, action)
        q = reward + self.gamma * self.search(next_state, depth + 1)
        self.update_value(state, action, q)
        return q

    def simulate_action(self, state, action):
        self.simulator.set_state(state)
        return self.simulator.take_action(action)

    def backpropagate(self, path, reward):
        for item in reversed(path):
            if isinstance(item, tuple):
                node, action = item
                self.N[(node, action)] += 1
                self.Q[(node, action)] += reward
            else:
                node = item
            self.N[node] = sum(self.N[(node, a)] for a in range(self.env.action_space.n))
            self.Q[node] = sum(self.Q[(node, a)] for a in range(self.env.action_space.n))

    def monte_carlo_planning(self, state, max_iterations=5000):
        if state not in self.children:
            self.expand(state)
        
        for _ in range(max_iterations):
            self.search(state, depth=0)
        
        return self.best_action(state)

    def best_action(self, state):
        valid_actions = [a for a in self.children[state] if self.is_valid_action(state, a)]
        return max(valid_actions, key=lambda a: self.Q[(state, a)] / (self.N[(state, a)] + 1e-8))