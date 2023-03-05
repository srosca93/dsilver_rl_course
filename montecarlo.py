from easy21 import Easy21
from easy21 import Action
from enum import Enum
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy

class eGreedy(Enum):
	EXPLORE = 0
	EXPLOIT = 1

class MonteCarloAgent:

	def __init__(self):
		self.Q_values = np.zeros((2,10,21))
		self.N_stateaction = np.zeros((2,10,21))
		self.N_states = np.zeros((10,21))
		self.epsilon = 0

	def act(self, state):
		self.epsilon = 100000/(100000 + self.N_states[state["dealer_card"]-1, state["player_sum"]-1])
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

	def sample_episode(self):
		game = Easy21()
		index = 0
		episode = dict()
		episode[index] = dict()
		episode[index]["state"] = copy.deepcopy(game.start_game())
		while(episode[index]["state"]["terminated"] == False):
			episode[index]["action"] = self.act(episode[index]["state"])
			episode[index+1] = dict()
			episode[index+1]["state"], episode[index+1]["reward"] = copy.deepcopy(game.step(episode[index]["action"]))
			index += 1
		return episode

	def train(self, num_episodes):
		for i in range(num_episodes):
			episode = self.sample_episode()
			for timestep in episode:
				if episode[timestep]["state"]["terminated"] != True:
					action = episode[timestep]["action"].value
					dealer_card = episode[timestep]["state"]["dealer_card"]-1
					player_sum = episode[timestep]["state"]["player_sum"]-1
					self.N_states[dealer_card, player_sum] += 1
					self.N_stateaction[action, dealer_card, player_sum] += 1
					alpha = 1/self.N_stateaction[action, dealer_card, player_sum]
					reward = episode[len(episode)-1]["reward"]
					self.Q_values[action, dealer_card, player_sum] += alpha*(reward-self.Q_values[action, dealer_card, player_sum])
