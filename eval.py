from montecarlo import MonteCarloAgent
from tdlambdasarsa import TDLambdaSarsaAgent
import numpy as np
import matplotlib.pyplot as plt

def MSE(Q, Q_star):
	error = np.sum(np.square(Q - Q_star))
	return error

mcagent = MonteCarloAgent()
lambda_values = np.arange(0.0,1.1,0.1)
mcagent.train(100000)
mcagent.plot_Q_values()
index = 0
errors = []
for lam in lambda_values:
	tdagent = TDLambdaSarsaAgent()
	if lam == 0 or lam == 1:
		scores = []
		for i in range(1000):
			tdagent.train(1, lam)
			scores.append(MSE(tdagent.Q_values, mcagent.Q_values))
		fig, ax = plt.subplots()
		ax.plot(range(1000), scores)
		plt.show()
	else:
		tdagent.train(1000, lam)
	errors.append(MSE(tdagent.Q_values, mcagent.Q_values))
	index += 1

fig, ax = plt.subplots()
ax.plot(lambda_values, errors)
plt.show()