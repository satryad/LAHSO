import matplotlib.pyplot as plt
import pandas as pd

cost = "Cost"
path = "csv_output"
simulations = 100


def column_color(delta, label1, label2):
    # We do file2 - file1, (lower values are better), we label with the better one, so
    #  positive delta means file1 was better.
    return label1 if delta > 0 else label2


def comparison(file1, file2, label1, label2):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df_comparison = df1.copy()
    df_comparison["Total Cost Delta"] = df2["Total Cost"] - df1["Total Cost"]
    df_comparison["Total Storage Cost Delta"] = (
        df2["Total Storage Cost"] - df1["Total Storage Cost"]
    )
    df_comparison["Total Travel Cost Delta"] = (
        df2["Total Travel Cost"] - df1["Total Travel Cost"]
    )
    df_comparison["Total Handling Cost Delta"] = (
        df2["Total Handling Cost"] - df1["Total Handling Cost"]
    )
    df_comparison["Total Delay Penalty Delta"] = (
        df2["Total Delay Penalty"] - df1["Total Delay Penalty"]
    )
    df_comparison["Total Cost Delta Sign"] = df_comparison["Total Cost Delta"].map(
        lambda d: column_color(d, label1, label2)
    )
    df_comparison["Total Storage Cost Delta Sign"] = df_comparison[
        "Total Storage Cost Delta"
    ].map(lambda d: column_color(d, label1, label2))
    df_comparison["Total Travel Cost Delta Sign"] = df_comparison[
        "Total Travel Cost Delta"
    ].map(lambda d: column_color(d, label1, label2))
    df_comparison["Total Handling Cost Delta Sign"] = df_comparison[
        "Total Handling Cost Delta"
    ].map(lambda d: column_color(d, label1, label2))
    df_comparison["Total Delay Penalty Delta Sign"] = df_comparison[
        "Total Delay Penalty Delta"
    ].map(lambda d: column_color(d, label1, label2))
    return df_comparison.reset_index()


def main():
    df_comparison = comparison(
        f"{path}/aw_Def_{simulations}.csv",
        f"{path}/gp_Def_{simulations}.csv",
        "blue",
        "orange",
    )
    df_comparison.sort_values(by=f"Total {cost}", ascending=False, inplace=True)

    # Bar chart for Delta
    plt.figure(figsize=(10, 6))

    # Assign colors based on the Delta values
    colors = df_comparison[f"Total {cost} Delta Sign"].to_list()

    plt.bar(df_comparison.index, df_comparison[f"Total {cost} Delta"], color=colors)
    # plt.bar(df_comparison['Episode'], df_comparison['Delta'], color=colors)

    # Add a horizontal line at y=0
    plt.axhline(0, color="black", linewidth=1, linestyle="--")

    plt.xlabel("Episode")
    plt.ylabel(f"Total {cost} Difference")
    plt.xticks(rotation=90)

    # Create custom legend
    blue_patch = plt.Line2D([0], [0], color="blue", lw=4, label="Greedy Policy")
    orange_patch = plt.Line2D([0], [0], color="orange", lw=4, label="Always Wait")
    plt.legend(handles=[blue_patch, orange_patch])

    min_y_limit = -60000
    max_y_limit = 60000  # Adjust the multiplier for padding
    plt.ylim(min_y_limit, max_y_limit)

    plt.tight_layout()
    plt.show()

    print(df_comparison.head())
