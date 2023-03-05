'''
Easy21 assignment from David Silver's Reinforcement Learning Class
'''

from enum import Enum
import random

class Action(Enum):
	STICK = 0
	HIT = 1

class Color(Enum):
	BLACK = 0
	RED = 1

class Easy21:

	def start_game(self):
		self.state = dict()
		self.state["dealer_card"] = self.draw_card(Color.BLACK)
		self.state["player_sum"] = self.draw_card(Color.BLACK)
		self.state["terminated"] = False
		self.reward = 0
		return self.state

	def step(self, action):
		if action == Action.HIT:
			card_value = self.draw_card()
			self.state["player_sum"] = self.state["player_sum"] + card_value
			self.state["terminated"] = (self.state["player_sum"] > 21) or (self.state["player_sum"] < 1)
			reward = -1 if self.state["terminated"] else 0
		else:
			self.state["terminated"] = True
			dealer_sum = self.state["dealer_card"]
			dealer_playing = True
			while dealer_playing:
				dealer_sum += self.draw_card()
				if dealer_sum >= 17:
					dealer_playing = False
					if dealer_sum > 21 or dealer_sum < self.state["player_sum"]:
						reward = 1
					elif dealer_sum > self.state["player_sum"]:
						reward = -1
					else:
						reward = 0
				elif dealer_sum < 1:
					dealer_playing = False
					reward = 1
				else:
					pass

			pass
		return self.state, reward

	def draw_card(self,color=None):
		population = [Color.BLACK, Color.RED]
		probs = [1, 2]
		card = dict()
		if color is not None:
			card["color"] = color
		else:
			card["color"] = random.choices(population, probs)[0]
		card["number"] = random.randint(1,10)
		return card["number"] * -1 if card["color"] == Color.RED else card["number"]
