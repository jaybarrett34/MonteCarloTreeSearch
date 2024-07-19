import random
from math import sqrt, log, exp
from utilities.node import Node

class MaximumEntropyTreeSearch:
    def __init__(self, env, tree, temperature=0.4):
        self.env = env
        self.tree = tree
        self.action_space = self.env.action_space.n
        self.temperature = temperature 
        state = self.env.reset()
        self.tree.add_node(Node(state=state, action=None, action_space=self.action_space, reward=0, terminal=False))

    def expand(self, node):
        action = node.untried_action()
        result = self.env.step(action)
        if len(result) == 4:
            state, reward, done, info = result
        else:
            state, reward, done, truncated, info = result
        new_node = Node(state=state, action=action, action_space=self.action_space, reward=reward, terminal=done)
        self.tree.add_node(new_node, node)
        return new_node

    def default_policy(self, node):
        if node.terminal:
            return node.reward

        while True:
            action = random.randint(0, self.action_space - 1)
            result = self.env.step(action)
            if len(result) == 4:
                state, reward, done, info = result
            else:
                state, reward, done, truncated, info = result
            if done:
                return reward

    def compute_entropy(self, parent, child, exploration_constant):
        exploitation_term = child.total_simulation_reward / child.num_visits
        probabilities = [exploitation_term, 1 - exploitation_term]
        entropy = -sum(p * log(p) for p in probabilities if p > 0)
        exploration_term = exploration_constant * sqrt(2 * log(parent.num_visits) / child.num_visits)
        total_value = exploitation_term + self.temperature * entropy + exploration_term
        return total_value, exploitation_term, entropy, exploration_term

    def softmax(self, values):
        max_val = max(values)
        exp_values = [exp((v - max_val) / self.temperature) for v in values]
        sum_exp_values = sum(exp_values)
        return [ev / sum_exp_values for ev in exp_values]

    def best_child(self, node, exploration_constant):
        children = self.tree.children(node)
        values = [self.compute_entropy(node, child, exploration_constant)[0] for child in children]
        softmax_probs = self.softmax(values)
        chosen_index = random.choices(range(len(children)), weights=softmax_probs, k=1)[0]
        best_child = children[chosen_index]
        best_value, best_exploitation, best_entropy, best_exploration = self.compute_entropy(node, best_child, exploration_constant)
        return best_child, best_value, best_exploitation, best_entropy, best_exploration

    def tree_policy(self, node):
        while not node.terminal:
            if self.tree.is_expandable(node):
                return self.expand(node)
            else:
                node, _, _, _, _ = self.best_child(node, exploration_constant=1.0/sqrt(2.0))
                result = self.env.step(node.action)
                if len(result) == 4:
                    state, reward, done, info = result
                else:
                    state, reward, done, truncated, info = result
                assert node.state == state
        return node

    def backward(self, node, value):
        while node:
            node.num_visits += 1
            node.total_simulation_reward += value
            node.performance = node.total_simulation_reward / node.num_visits
            node = self.tree.parent(node)

    def forward(self):
        self._forward(self.tree.root)

    def _forward(self, node):
        best_child, best_value, best_exploitation, best_entropy, best_exploration = self.best_child(node, exploration_constant=0)
        print("****** {} ******".format(best_child.state))
        print(f"Best child for node {node.state} is {best_child.state} with value {best_value:.4f}")
        print(f"Components: Exploitation={best_exploitation:.4f}, Entropy={best_entropy:.4f}, Exploration={best_exploration:.4f}")
        
        for child in self.tree.children(best_child):
            _, exploitation, entropy, _ = self.compute_entropy(node, child, 0)
            print(f"{child.state}: Exploitation={exploitation:.4f}, Entropy={entropy:.4f}")

        if len(self.tree.children(best_child)) > 0:
            self._forward(best_child)

    def find_best_action(self, state):
        root = self.tree.root
        self.env.reset()
        self.env.s = state
        for _ in range(1000):
            node = root
            self.env.s = node.state
            node = self.tree_policy(node)
            reward = self.default_policy(node)
            self.backward(node, reward)
        best_node, _, _, _, _ = self.best_child(root, exploration_constant=0)
        return best_node.action if best_node else None
