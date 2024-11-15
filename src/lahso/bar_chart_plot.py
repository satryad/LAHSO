import matplotlib.pyplot as plt
import pandas as pd

# import from other python files

disruption = "3"

path = "csv_output"
input = ["GP", "AW"]

# df_opt = pd.read_csv(f'{path}\gp_{disruption}_20_opt_v0.csv')
df_aw = pd.read_csv(f"{path}\\aw_{disruption}_20.csv")
df_gp = pd.read_csv(f"{path}\\gp_{disruption}_20.csv")


gp = df_gp["Total Cost"]
aw = df_aw["Total Cost"]
# opt = df_opt['Total Cost']

width = 0.35  # the width of the bars
x = range(len(gp))
labels = []
for eps in x:
    c = eps + 1
    case = f"C{c}"
    labels.append(case)

fig, ax = plt.subplots()
bars1 = ax.bar(x, gp, width, label="Greedy Policy")
bars2 = ax.bar([i + width for i in x], aw, width, label="Always Wait")
# bars2 = ax.bar([i + width for i in x], opt, width, label='Always Optimize', color='cyan')

# Adding labels, title, and customizing ticks
ax.set_xlabel("Sample Case")
ax.set_ylabel("Total Costs")
# ax.set_title('Performance Comparison')
ax.set_xticks([i + width / 2 for i in x])
ax.set_xticklabels(labels)
ax.legend()
# Setting minimum y-axis limit
min_y_limit = 400000  # Adjust this to set your desired minimum y-axis limit
# min_y_limit = 80000
max_y_limit = max(max(gp), max(aw)) * 1.1  # Adjust the multiplier for padding
# max_y_limit = 600000
# max_y_limit = 520000
ax.set_ylim(min_y_limit, max_y_limit)

# Showing the plot
plt.show()
print(gp)
