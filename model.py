from math import sqrt
from utilities.node import Node

class MonteCarloTreeSearch:
    def __init__(self, env, tree):
        self.env = env
        self.tree = tree
        self.action_space = self.env.action_space.n
        
        # Initialize the environment and tree with the start state
        state = self.env.reset()
        self.tree.add_node(Node(state=state, action=None, action_space=self.action_space, reward=0, terminal=False))

    def expand(self, node):
        # 9. action := selectAction(state, depth)
        action = node.untried_action()
        
        # 10. (nextstate, reward) := simulateAction(state, action)
        state, reward, done, _ = self.env.step(action)
        
        # Create a new node with the resulting state, action, and reward
        new_node = Node(state=state, action=action, action_space=self.action_space, reward=reward, terminal=done)
        
        # Add the new node to the tree
        self.tree.add_node(new_node, node)
        return new_node

    def simulate_action(self, state, action):
        # 10. (nextstate, reward) := simulateAction(state, action)
        next_state, reward, done, _ = self.env.step(action)
        return next_state, reward

    def select_action(self, node, depth, max_depth):
        # 7. if Terminal(state) or depth >= max_depth then return 0
        if node.terminal or depth >= max_depth:
            return None
        
        # 8. if Leaf(state, d) then return Evaluate(state)
        if self.tree.is_expandable(node):
            return self.expand(node)
        else:
            # 9. action := selectAction(state, depth)
            return self.best_child(node, exploration_constant=1.0/sqrt(2.0))

    def update_value(self, node, action, q, depth):
        # 12. UpdateValue(state, action, q, depth)
        node.visit_count += 1
        node.value += (q - node.value) / node.visit_count

    def search(self, state, depth, max_depth):
        # 6. function search(state, depth)
        node = self.tree.nodes[state]
        
        # 7. if Terminal(state) or depth >= max_depth then return 0
        if node.terminal or depth >= max_depth:
            return 0
        
        # 8. if Leaf(state, d) then return Evaluate(state)
        if self.tree.is_expandable(node):
            return self.evaluate(node)
        
        # 9. action := selectAction(state, depth)
        action = self.select_action(node, depth)
        
        # 10. (nextstate, reward) := simulateAction(state, action)
        next_state, reward = self.simulate_action(state, action)
        
        # 11. q:= reward + γ search(nextstate, depth + 1)
        q = reward + 0.9 * self.search(next_state, depth + 1, max_depth)
        
        # 12. UpdateValue(state, action, q, depth)
        self.update_value(node, action, q, depth)
        
        # 13. return q
        return q

    def evaluate(self, state):
        # We can change reward values as needed. Represents goal, hole, tile
        if state.reward == 1:
            return 0
        elif state.reward == 0 and state.terminal:
            return -100
        else:
            return -1

    def best_action(self, node, depth):
        # After the search phase, return the best action from the root node
        best_child = max(self.tree.children(node), key=lambda child: child.value)
        return best_child.action

    def monte_carlo_planning(self, state, max_iterations, max_depth):
        # 1. function MonteCarloPlanning(state)
        for _ in range(max_iterations):
            # 2. repeat
            self.search(state, depth=0, max_depth=max_depth)
            # 4. until Timeout or max iter reached
        
        # 5. return bestAction(state,0)
        node = self.tree.nodes[state]
        return self.best_action(node, depth=0)
