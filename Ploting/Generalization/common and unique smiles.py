import pandas as pd
import matplotlib.pyplot as plt

# Load datasets
anticancer_file = "refined_anticancerNP.csv"
npass_file = "cleaned_NPASSstructureInfo.csv"

anticancer_df = pd.read_csv(anticancer_file)
npass_df = pd.read_csv(npass_file)

# Ensure SMILES column names match your data
if 'SMILES' not in anticancer_df.columns or 'SMILES' not in npass_df.columns:
    print("Error: Both files must have a 'SMILES' column.")
    exit()

# Extract SMILES columns
anticancer_smiles = anticancer_df['SMILES']
npass_smiles = npass_df['SMILES']

# Convert to sets for comparison
anticancer_set = set(anticancer_smiles)
npass_set = set(npass_smiles)

# Find common and unique SMILES
common_smiles = anticancer_set.intersection(npass_set)
unique_anticancer_smiles = anticancer_set.difference(npass_set)
unique_npass_smiles = npass_set.difference(anticancer_set)

# Save results
pd.DataFrame({'SMILES': list(common_smiles)}).to_csv("common3_smiles.csv", index=False)
pd.DataFrame({'SMILES': list(unique_anticancer_smiles)}).to_csv("unique3_anticancer_smiles.csv", index=False)
pd.DataFrame({'SMILES': list(unique_npass_smiles)}).to_csv("unique3_npass_smiles.csv", index=False)

# Print summary
print(f"Number of common SMILES: {len(common_smiles)}")
print(f"Number of unique SMILES in refined_anticancerNP.csv: {len(unique_anticancer_smiles)}")
print(f"Number of unique SMILES in cleaned_NPASSstructureInfo.csv: {len(unique_npass_smiles)}")

# Prepare data for plotting
categories = ['Common SMILES', 'Unique Anticancer SMILES', 'Unique NPASS SMILES']
counts = [
    len(common_smiles),
    len(unique_anticancer_smiles),
    len(unique_npass_smiles)
]

# Create a dataframe for plotting
plot_data = pd.DataFrame({
    'Category': categories,
    'Count': counts
})

# Set up the plot
fig, ax = plt.subplots(figsize=(8, 6))

# Define colors
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Different colors for each category

# Create a bar plot with different colors
bars = ax.bar(categories, counts, color=colors)

# Add annotations to the bars
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=12, fontweight='bold')

# Add title and labels
ax.set_title('Comparison of Unique and Common SMILES', fontsize=16, fontweight='bold')
ax.set_xlabel('Category', fontsize=14)
ax.set_ylabel('Count', fontsize=14)

# Add legend
ax.legend(bars, categories, loc='upper left', fontsize=12)

# Improve layout and display the plot
plt.tight_layout()
plt.savefig('comparison_of_smiles_plot.png', dpi=300)  # Save with high resolution
plt.show()