from easy21 import Easy21
from easy21 import Action
from enum import Enum
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy
import montecarlo

class eGreedy(Enum):
	EXPLORE = 0
	EXPLOIT = 1

class TDLambdaSarsaAgent:

	def __init__(self):
		self.Q_values = np.zeros((2,10,21))
		self.N_stateaction = np.zeros((2,10,21))
		self.N_states = np.zeros((10,21))
		self.epsilon = 0

	def act(self, state):
		self.epsilon = 1000/(1000 + self.N_states[state["dealer_card"]-1, state["player_sum"]-1])
		population = list(eGreedy)
		probs = [self.epsilon, 1-self.epsilon]
		decision = random.choices(population, probs)[0]
		if decision == eGreedy.EXPLOIT:
			action = Action(np.argmax(self.Q_values[:, state["dealer_card"]-1, state["player_sum"]-1], axis=0))
		else:
			action = random.choice(list(Action))
		return action

	def plot_Q_values(self):
		optimal_state_values = np.max(self.Q_values, axis=0)

		hf = plt.figure()
		ha = hf.add_subplot(111, projection='3d')

		X, Y = np.meshgrid(range(21), range(10))  # `plot_surface` expects `x` and `y` data to be 2D
		ha.plot_surface(X, Y, optimal_state_values)

		plt.show()

	def run_episode(self, lam):
		e_trace = np.zeros((2,10,21))
		game = Easy21()
		state = copy.deepcopy(game.start_game())
		action = self.act(state)
		while state["terminated"] == False:
			self.N_states[state["dealer_card"]-1, state["player_sum"]-1] += 1
			self.N_stateaction[action.value, state["dealer_card"]-1, state["player_sum"]-1] += 1
			next_state, reward = copy.deepcopy(game.step(action))
			current_state_action_value = self.Q_values[action.value, state["dealer_card"]-1, state["player_sum"]-1]
			if next_state["terminated"] == False:
				next_action = self.act(next_state)
				next_state_action_value = self.Q_values[next_action.value, next_state["dealer_card"]-1, next_state["player_sum"]-1]
			else:
				next_action = 0
				next_state_action_value = 0
			delta = reward + next_state_action_value - current_state_action_value
			e_trace[action.value, state["dealer_card"]-1, state["player_sum"]-1] += 1
			alpha = np.zeros((2,10,21))
			alpha[np.nonzero(self.N_stateaction)] = 1/self.N_stateaction[np.nonzero(self.N_stateaction)]
			self.Q_values += alpha*delta*e_trace
			e_trace *= lam
			action = copy.deepcopy(next_action)
			state = copy.deepcopy(next_state)

	def train(self, num_episodes, lam):
		for i in range(num_episodes):
			self.run_episode(lam)

	def evaluate(self, proper_Q):
		pass
