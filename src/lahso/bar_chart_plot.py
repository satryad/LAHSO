import matplotlib.pyplot as plt
import pandas as pd

cost = "Cost"
disruption = "3"
path = "csv_output"
simulations = 100

df_aw = pd.read_csv(f"{path}/aw_{disruption}_{simulations}.csv")
df_gp = pd.read_csv(f"{path}/gp_{disruption}_{simulations}.csv")
df_gp.rename(columns={"Total Cost": "Total Cost gp"}, inplace=True)
df_gp.rename(columns={"Total Storage Cost": "Total Storage Cost gp"}, inplace=True)
df_gp.rename(columns={"Total Travel Cost": "Total Travel Cost gp"}, inplace=True)
df_gp.rename(columns={"Total Handling Cost": "Total Handling Cost gp"}, inplace=True)
df_gp.rename(columns={"Total Delay Penalty": "Total Delay Penalty gp"}, inplace=True)
df_comparison = pd.merge(
    df_aw,
    df_gp[
        [
            "Episode",
            "Total Cost gp",
            "Total Storage Cost gp",
            "Total Travel Cost gp",
            "Total Handling Cost gp",
            "Total Delay Penalty gp",
        ]
    ],
    on="Episode",
    how="left",
)
df_comparison["Delta"] = (
    df_comparison[f"Total {cost}"] - df_comparison[f"Total {cost} gp"]
)
df_comparison.sort_values(by="Delta", ascending=False, inplace=True)
df_comparison = df_comparison.reset_index()

# Bar chart for Delta
plt.figure(figsize=(10, 6))

# df_opt = pd.read_csv(f'{path}/gp_{disruption}_20_opt_v0.csv')
df_aw = pd.read_csv(f"{path}/aw_{disruption}_20.csv")
df_gp = pd.read_csv(f"{path}/gp_{disruption}_20.csv")
# Assign colors based on the Delta values
colors = ["blue" if delta > 0 else "orange" for delta in df_comparison["Delta"]]

plt.bar(df_comparison.index, df_comparison["Delta"], color=colors)
# plt.bar(df_comparison['Episode'], df_comparison['Delta'], color=colors)

# Add a horizontal line at y=0
plt.axhline(0, color="black", linewidth=1, linestyle="--")

plt.xlabel("Episode")
plt.ylabel(f"Total {cost} Difference")
plt.xticks(rotation=90)

# Create custom legend
blue_patch = plt.Line2D([0], [0], color="blue", lw=4, label="RL-Assisted")
orange_patch = plt.Line2D([0], [0], color="orange", lw=4, label="Always Wait")
plt.legend(handles=[blue_patch, orange_patch])

min_y_limit = -60000
max_y_limit = 60000  # Adjust the multiplier for padding
plt.ylim(min_y_limit, max_y_limit)

plt.tight_layout()
plt.show()

print(df_comparison.head())
