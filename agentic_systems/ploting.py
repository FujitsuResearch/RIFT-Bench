import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
file_path = "../agentic_ai_scanning_res/domain_systems__personal_assistant__defenses_totals_overall.csv"
df = pd.read_csv(file_path)

# Ensure numeric columns
df["BaseUtility_ratio"] = pd.to_numeric(
    df["BaseUtility_ratio"],
    errors="coerce"
)

df["Deterministic Attack Success ratio"] = pd.to_numeric(
    df["Deterministic Attack Success ratio"],
    errors="coerce"
)

# Keep relevant rows
df = df.dropna(subset=[
    "Defense name",
    "Attack status",
    "BaseUtility_ratio",
    "Deterministic Attack Success ratio"
])

# Average values for each (Defense name, Attack status)
avg_df = (
    df.groupby(["Defense name", "Attack status"], as_index=False)
    .agg({
        "BaseUtility_ratio": "mean",
        "Deterministic Attack Success ratio": "mean"
    })
)

# Plot
plt.figure(figsize=(12, 8))

statuses = avg_df["Attack status"].unique()

for status in statuses:
    subset = avg_df[avg_df["Attack status"] == status]

    plt.scatter(
        subset["BaseUtility_ratio"],
        subset["Deterministic Attack Success ratio"],
        s=140,
        alpha=0.8,
        label=status
    )

    # Annotate each point with defense name
    for _, row in subset.iterrows():
        plt.annotate(
            row["Defense name"],
            (
                row["BaseUtility_ratio"],
                row["Deterministic Attack Success ratio"]
            ),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
            alpha=0.8
        )

# Labels and title
plt.xlabel("Average BaseUtility_ratio")
plt.ylabel("Average Deterministic Attack Success ratio")
plt.title("Average Defense Performance by Attack Status")

# Grid and legend
plt.grid(True, alpha=0.3)
plt.legend(title="Attack status")

plt.tight_layout()
plt.show()
plt.savefig("../agentic_ai_scanning_res/defense_attack_plot.png", dpi=300, bbox_inches="tight")