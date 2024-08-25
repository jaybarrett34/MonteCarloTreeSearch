import gym
import numpy as np

class EnvironmentWrapper:
    def __init__(self, env_name='FrozenLake-v1', is_slippery=False, 
                 custom_map=None, env=None):
        if env is None:
            if custom_map:
                self.env = gym.make(env_name, desc=custom_map, is_slippery=is_slippery)
            else:
                self.env = gym.make(env_name, is_slippery=is_slippery)
        else:
            self.env = env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        desc = self.env.unwrapped.desc.astype(str).tolist()
        self.nrow, self.ncol = len(desc), len(desc[0])
        
        self.state = None
        
        self.hole_states = []
        self.goal_state = None
        self.start_state = None
        
        for i, row in enumerate(desc):
            for j, cell in enumerate(row):
                if cell == 'H':
                    self.hole_states.append(i * self.ncol + j)
                elif cell == 'G':
                    self.goal_state = i * self.ncol + j
                elif cell == 'S':
                    self.start_state = i * self.ncol + j
        
        self.shaped_rewards = self.calculate_shaped_reward()
        
        self.reset()

    def set_state(self, state):
        self.state = state

    def manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def calculate_shaped_reward(self):
        goal_pos = self.index_to_pos(self.goal_state)
        max_distance = self.manhattan_distance((0, 0), (self.nrow - 1, self.ncol - 1))
        
        shaped_rewards = {}
        for state in range(self.nrow * self.ncol):
            if state in self.hole_states:
                shaped_rewards[state] = -100.0
            elif state == self.goal_state:
                shaped_rewards[state] = 0.0
            else:
                pos = self.index_to_pos(state)
                distance = self.manhattan_distance(pos, goal_pos)
                shaped_rewards[state] = -5.0 + (5.0 * (max_distance - distance) / max_distance)
        
        return shaped_rewards

    def take_action(self, action):
        row, col = self.state // self.ncol, self.state % self.ncol
        
        if action == 0:  # Left
            col = max(0, col - 1)
        elif action == 1:  # Down
            row = min(self.nrow - 1, row + 1)
        elif action == 2:  # Right
            col = min(self.ncol - 1, col + 1)
        elif action == 3:  # Up
            row = max(0, row - 1)
        
        new_state = row * self.ncol + col
        reward = self.get_reward(new_state)
        self.state = new_state
        
        return new_state, reward

    def get_reward(self, state):
        return self.shaped_rewards[state]

    def is_goal(self, state):
        return state == self.goal_state
    
    def is_hole(self, state):
        return state in self.hole_states

    def is_terminal(self, state):
        return self.is_hole(state) or self.is_goal(state)

    def reset(self):
        self.state = self.start_state
        return self.state

    def get_state(self):
        return self.state

    def index_to_pos(self, index):
        row = index // self.ncol
        col = index % self.ncol
        return row, col

    def pos_to_index(self, pos):
        row, col = pos
        return row * self.ncol + col

class SimulatorWrapper(EnvironmentWrapper):
    def __init__(self, env_name='FrozenLake-v1', is_slippery=False, custom_map=None, env=None):
        super().__init__(env_name, is_slippery, custom_map, env)

    def take_action(self, action):
        if np.random.random() < 0.1:
            action = self.action_space.sample()
        
        return super().take_action(action)