import numpy as np

class CliffWalkingEnv:
    def __init__(self, rows=4, cols=12):
        self.rows = rows
        self.cols = cols
        self.start_state = (3, 0)
        self.goal_state = (3, 11)
        self.cliff = [(3, i) for i in range(1, 11)]
        self.actions = [0, 1, 2, 3]  # Up, Down, Left, Right

    def step(self, state, action):
        r, c = state
        if action == 0: r = max(0, r - 1)
        elif action == 1: r = min(self.rows - 1, r + 1)
        elif action == 2: c = max(0, c - 1)
        elif action == 3: c = min(self.cols - 1, c + 1)

        next_state = (r, c)
        
        if next_state in self.cliff:
            return self.start_state, -100, False
        
        if next_state == self.goal_state:
            return next_state, 0, True
            
        return next_state, -1, False

class Agent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}

    def get_q(self, state, action):
        return self.q_table.get(state, [0.0] * 4)[action]

    def set_q(self, state, action, value):
        if state not in self.q_table:
            self.q_table[state] = [0.0] * 4
        self.q_table[state][action] = value

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.env.actions)
        
        q_values = self.q_table.get(state, [0.0] * 4)
        max_q = max(q_values)
        actions_with_max_q = [i for i, v in enumerate(q_values) if v == max_q]
        return np.random.choice(actions_with_max_q)

class QLearningAgent(Agent):
    def update(self, state, action, reward, next_state):
        current_q = self.get_q(state, action)
        next_max_q = max(self.q_table.get(next_state, [0.0] * 4))
        new_q = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)
        self.set_q(state, action, new_q)

class SarsaAgent(Agent):
    def update(self, state, action, reward, next_state, next_action):
        current_q = self.get_q(state, action)
        next_q = self.get_q(next_state, next_action)
        new_q = current_q + self.alpha * (reward + self.gamma * next_q - current_q)
        self.set_q(state, action, new_q)
