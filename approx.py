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

class ApproxSarsaAgent:

	def __init__(self):
		self.weights = np.zeros((2,3,6))
		self.epsilon = 0.8
		self.step_size = 0.01

	def features_from_state_action(self, state, action):
		dealer_card_groups = []
		if state["dealer_card"] >= 1 and state["dealer_card"] <= 4:
			dealer_card_groups.append(0)
		if state["dealer_card"] >= 4 and state["dealer_card"] <= 7:
			dealer_card_groups.append(1)
		if state["dealer_card"] >= 7 and state["dealer_card"] <= 10:
			dealer_card_groups.append(2)

		player_sum_groups = []
		if state["player_sum"] >= 1 and state["player_sum"] <= 6:
			player_sum_groups.append(0)
		if state["player_sum"] >= 4 and state["player_sum"] <= 9:
			player_sum_groups.append(1)
		if state["player_sum"] >= 7 and state["player_sum"] <= 12:
			player_sum_groups.append(2)
		if state["player_sum"] >= 10 and state["player_sum"] <= 15:
			player_sum_groups.append(3)
		if state["player_sum"] >= 13 and state["player_sum"] <= 18:
			player_sum_groups.append(4)
		if state["player_sum"] >= 16 and state["player_sum"] <= 21:
			player_sum_groups.append(5)

		features = np.zeros((2,3,6))
		features[action.value,dealer_card_groups,player_sum_groups] = 1
		return features


	def value_from_state_action(self, state, action):
		features = self.features_from_state_action(state, action)
		return np.sum(features * self.weights)

	def act(self, state):
		population = list(eGreedy)
		probs = [self.epsilon, 1-self.epsilon]
		decision = random.choices(population, probs)[0]
		if decision == eGreedy.EXPLOIT:
			values = []
			values.append(self.value_from_state_action(state, Action.STICK))
			values.append(self.value_from_state_action(state, Action.HIT))
			action = Action(np.argmax(values))
		else:
			action = random.choice(list(Action))
		return action

	def run_episode(self, lam):
		e_trace = np.zeros((2,3,6))
		game = Easy21()
		state = copy.deepcopy(game.start_game())
		action = self.act(state)
		while state["terminated"] == False:
			features = self.features_from_state_action(state, action)
			next_state, reward = copy.deepcopy(game.step(action))
			current_state_action_value = self.value_from_state_action(state,action)
			if next_state["terminated"] == False:
				next_action = self.act(next_state)
				next_state_action_value = self.value_from_state_action(next_state,next_action)
			else:
				next_action = 0
				next_state_action_value = 0
			delta = reward + next_state_action_value - current_state_action_value
			e_trace = e_trace * lam + features
			self.weights += self.step_size * delta * e_trace
			action = copy.deepcopy(next_action)
			state = copy.deepcopy(next_state)

	def train(self, num_episodes, lam):
		for i in range(num_episodes):
			self.run_episode(lam)

	@property
	def Q_values(self):
		Q_values = np.zeros((2,10,21))
		for i in range(2):
			for j in range(10):
				for k in range(21):
					state = dict()
					state["dealer_card"]= j+1
					state["player_sum"] = k+1
					Q_values[i,j,k] = self.value_from_state_action(state, Action(i))
		return Q_values

	def plot_Q_values(self):
		optimal_state_values = np.max(self.Q_values, axis=0)

		hf = plt.figure()
		ha = hf.add_subplot(111, projection='3d')

		X, Y = np.meshgrid(range(21), range(10))  # `plot_surface` expects `x` and `y` data to be 2D
		ha.plot_surface(X, Y, optimal_state_values)

		plt.show()

if __name__ == "__main__":
	agent = ApproxSarsaAgent()
	agent.train(100000, 0.5)
	agent.plot_Q_values()