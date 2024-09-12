import pickle
import matplotlib.pyplot as plt
import pandas as pd
from config import tc_path, tr_path, smoothing

plt.style.use('ggplot')

with open(f'{tc_path}', 'rb') as f:
    cst = pickle.load(f)
with open(f'{tr_path}', 'rb') as f:
    rwd = pickle.load(f)

fig, axs = plt.subplots(1, 2, figsize=(20, 8))
seed = len(cst)
costs_smoothed = pd.Series(cst).rolling(smoothing, min_periods=smoothing).mean()

axs[0].plot(costs_smoothed)
axs[0].set_xlabel("Episode")
axs[0].set_ylabel("Episode total cost (Smoothed)")
axs[0].set_title(f"Episode total cost over {seed} episodes")

seed = len(rwd)
rewards_smoothed = pd.Series(rwd).rolling(smoothing, min_periods=smoothing).mean()
axs[1].plot(rewards_smoothed)
axs[1].set_xlabel("Episode")
axs[1].set_ylabel("Episode total reward (Smoothed)")
axs[1].set_title(f"Episode total reward over {seed} episodes")

# Add labels to the x and y axes
for ax in axs.flat:
    ax.set(xlabel='Episode', ylabel='Costs (Smoothed)')

plt.show()  # Explicitly show the plot
print(f"Minimum Cost: {min(cst)}")
index = cst.index(min(cst))
print(f'Minimum cost at simulation number {index}')
print(f"Maximum Cost: {max(cst)}")
index = cst.index(max(cst))
print(f'Maximum cost at simulation number {index}')