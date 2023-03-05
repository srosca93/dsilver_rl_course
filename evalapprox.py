from montecarlo import MonteCarloAgent
from approx import ApproxSarsaAgent
import numpy as np
import matplotlib.pyplot as plt

def MSE(Q, Q_star):
	error = np.sum(np.square(Q - Q_star))
	return error

mcagent = MonteCarloAgent()
lambda_values = np.arange(0.0,1.1,0.1)
mcagent.train(100000)
index = 0
errors = []
for lam in lambda_values:
	appagent = ApproxSarsaAgent()
	if lam == 0 or lam == 1:
		scores = []
		for i in range(10000):
			appagent.train(1, lam)
			scores.append(MSE(appagent.Q_values, mcagent.Q_values))
		fig, ax = plt.subplots()
		ax.plot(range(10000), scores)
		plt.show()
	else:
		appagent.train(10000, lam)
	errors.append(MSE(appagent.Q_values, mcagent.Q_values))
	index += 1

fig, ax = plt.subplots()
ax.plot(lambda_values, errors)
plt.show()