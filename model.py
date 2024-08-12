import random
from math import sqrt, log
from utilities.node import Node

class MonteCarloTreeSearch:
    def __init__(self, env, tree):
        self.env = env
        self.tree = tree
        self.action_space = self.env.action_space.n
        state = self.env.reset()
        self.tree.add_node(Node(state=state, action=None, action_space=self.action_space, reward=0, terminal=False))

    def expand(self, node):
        action = node.untried_action()
        state, reward, done, _ = self.env.step(action)
        new_node = Node(state=state, action=action, action_space=self.action_space, reward=reward, terminal=done)
        self.tree.add_node(new_node, node)
        return new_node

    def simulate_action(self, state, action):
        next_state, reward, done, _ = self.env.step(action)
        return next_state, reward

    def select_action(self, node, depth, max_depth):
        if node.terminal or depth >= max_depth:
            return None
        if self.tree.is_expandable(node):
            return self.expand(node)
        else:
            return self.best_child(node, exploration_constant=1.0/sqrt(2.0))

    def update_value(self, node, action, q, depth):
        node.visit_count += 1
        node.value += (q - node.value) / node.visit_count

    def search(self, state, depth, max_depth):
        node = self.tree.nodes[state]
        if node.terminal or depth >= max_depth:
            return 0
        if self.tree.is_expandable(node):
            return self.evaluate(node)
        action = self.select_action(node, depth)
        next_state, reward = self.simulate_action(state, action)
        q = reward + self.search(next_state, depth + 1, max_depth)
        self.update_value(node, action, q, depth)
        return q

    def evaluate(self, state):
        if state.reward == 1:
            return 0
        elif state.reward == 0 and state.terminal:
            return -100
        else:
            return -1

    def best_action(self, node, depth):
        best_child = max(self.tree.children(node), key=lambda child: child.value)
        return best_child.action

    def monte_carlo_planning(self, state, max_iterations, max_depth):
        for _ in range(max_iterations):
            self.search(state, depth=0, max_depth=max_depth)
        node = self.tree.nodes[state]
        return self.best_action(node, depth=0)