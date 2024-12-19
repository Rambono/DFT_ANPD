import pandas as pd
from rdkit import Chem
import matplotlib.pyplot as plt

# Function to check if a SMILES string is valid
def is_valid_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception as e:
        print(f"Error validating SMILES '{smiles}': {e}")
        return False

# Function to canonicalize SMILES strings
def canonicalize_smiles_column(smiles_column):
    canonical_smiles = []
    for smi in smiles_column:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                canonical_smi = Chem.MolToSmiles(mol, canonical=True)
                canonical_smiles.append(canonical_smi)
            else:
                canonical_smiles.append(None)
        except Exception as e:
            print(f"Error canonicalizing SMILES '{smi}': {e}")
            canonical_smiles.append(None)
    return canonical_smiles

# Load NPASSstructureInfo.csv
npass_structure_file = "NPASSstructureInfo.csv"

try:
    npass_df = pd.read_csv(npass_structure_file)
except FileNotFoundError as e:
    print(f"Error loading file: {e}")
    exit()

# Check that the necessary SMILES column exists
if 'SMILES' not in npass_df.columns:
    print("Error: Input file must have a 'SMILES' column.")
    exit()

# Remove invalid SMILES
print("Validating SMILES...")
npass_df['Valid'] = npass_df['SMILES'].apply(is_valid_smiles)
valid_npass_df = npass_df[npass_df['Valid']].drop(columns=['Valid'])

# Canonicalize SMILES
print("Canonicalizing SMILES...")
valid_npass_df['Canonical_SMILES'] = canonicalize_smiles_column(valid_npass_df['SMILES'])

# Drop rows with invalid canonicalization results
valid_npass_df = valid_npass_df.dropna(subset=['Canonical_SMILES'])

# Remove duplicates based on Canonical_SMILES
print("Removing duplicate SMILES...")
cleaned_npass_df = valid_npass_df.drop_duplicates(subset=['Canonical_SMILES'])

# Save cleaned data to a new CSV file
output_file = "cleaned_NPASSstructureInfo.csv"
cleaned_npass_df.to_csv(output_file, index=False)

print(f"Cleaned NPASSstructureInfo saved to {output_file}")



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

# Define shapes and colors
shapes = ['o', 's', '^']  # Circle, Square, Triangle
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Different colors for each category

# Plot each category with different shapes and colors
for i, (category, count) in enumerate(zip(categories, counts)):
    ax.scatter([category], [count], s=150, marker=shapes[i], color=colors[i], label=category)
    ax.text(category, count + 5, f"{count}", ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')

# Add title and labels
ax.set_title('Comparison of Unique and Common SMILES', fontsize=16, fontweight='bold')
ax.set_xlabel('Category', fontsize=14)
ax.set_ylabel('Count', fontsize=14)

# Add legend
ax.legend(loc='upper right', fontsize=12)

# Improve layout and display the plot
plt.tight_layout()
plt.savefig('comparison_of_smiles_plot.png', dpi=300)  # Save with high resolution
plt.show()


# Function to check if a SMILES string is valid
def is_valid_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception as e:
        print(f"Error validating SMILES '{smiles}': {e}")
        return False

# Function to canonicalize a column of SMILES
def canonicalize_smiles_column(smiles_column):
    canonical_smiles = []
    for smi in smiles_column:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                canonical_smi = Chem.MolToSmiles(mol, canonical=True)
                canonical_smiles.append(canonical_smi)
            else:
                canonical_smiles.append(None)
        except Exception as e:
            print(f"Error canonicalizing SMILES '{smi}': {e}")
            canonical_smiles.append(None)
    return canonical_smiles

# Load datasets
common_smiles_file = "common3_smiles.csv"
npass_structure_file = "NPASSstructureInfo.csv"

# Ensure the files are read into dataframes
try:
    common_smiles_df = pd.read_csv(common_smiles_file)
    npass_structure_df = pd.read_csv(npass_structure_file)
except FileNotFoundError as e:
    print(f"Error loading file: {e}")
    exit()

# Check that the necessary SMILES columns exist
if 'SMILES' not in common_smiles_df.columns or 'SMILES' not in npass_structure_df.columns:
    print("Error: Both input files must have a 'SMILES' column.")
    exit()

# Remove duplicate SMILES from NPASSstructureInfo.csv before canonicalization
npass_structure_df = npass_structure_df.drop_duplicates(subset=['SMILES'])

# Canonicalize SMILES columns for both datasets
print("Canonicalizing SMILES for common_smiles.csv...")
common_smiles_df['Canonical_SMILES'] = canonicalize_smiles_column(common_smiles_df['SMILES'])

print("Canonicalizing SMILES for NPASSstructureInfo.csv...")
npass_structure_df['Canonical_SMILES'] = canonicalize_smiles_column(npass_structure_df['SMILES'])

# Drop rows with invalid or uncanonicalizable SMILES
common_smiles_df = common_smiles_df.dropna(subset=['Canonical_SMILES'])
npass_structure_df = npass_structure_df.dropna(subset=['Canonical_SMILES'])

# Debugging: Check for duplicates in canonicalized NPASSstructureInfo.csv
duplicate_smiles_count = npass_structure_df['Canonical_SMILES'].duplicated().sum()
if duplicate_smiles_count > 0:
    print(f"Warning: {duplicate_smiles_count} duplicate canonical SMILES found in NPASSstructureInfo.csv. These will be removed.")
    npass_structure_df = npass_structure_df.drop_duplicates(subset=['Canonical_SMILES'])

# Find matching SMILES
print("Finding matching SMILES...")
matching_smiles = common_smiles_df['Canonical_SMILES'].isin(npass_structure_df['Canonical_SMILES'])
matched_common_smiles = common_smiles_df[matching_smiles]

# Debugging: Check unmatched SMILES
unmatched_smiles = common_smiles_df[~matching_smiles]
print(f"Number of unmatched SMILES: {len(unmatched_smiles)}")

# Merge with NPASS structure to get np_id
print("Merging matched SMILES with NPASSstructureInfo to get np_id...")
merged_df = pd.merge(
    matched_common_smiles, 
    npass_structure_df[['np_id', 'Canonical_SMILES']], 
    on='Canonical_SMILES'
)

# Debugging: Check for extra rows in the merged dataframe
extra_smiles = set(merged_df['Canonical_SMILES']) - set(common_smiles_df['Canonical_SMILES'])
if extra_smiles:
    print(f"Warning: {len(extra_smiles)} extra SMILES found in the merged dataframe. These may indicate issues with duplicates in NPASSstructureInfo.csv.")

# Save results
output_file = "Smiles3_with_np_id.csv"
merged_df.to_csv(output_file, index=False)

print(f"Saved matched SMILES with np_id to {output_file}")


# Load the CSV file with proper header handling
# Use header=0 to treat the first row as the column headers
smiles_df = pd.read_csv('Smiles3_with_np_id.csv', sep=',', header=0)

# Clean column names to remove any leading/trailing whitespace
smiles_df.columns = smiles_df.columns.str.strip()

# Check if 'SMILES' and 'np_id' columns are present
if 'SMILES' not in smiles_df.columns or 'np_id' not in smiles_df.columns:
    raise ValueError("Expected columns 'SMILES' and 'np_id' not found in the data.")

# Clean 'np_id' column values by removing leading/trailing whitespace
smiles_df['np_id'] = smiles_df['np_id'].astype(str).str.strip()

# Display the first few rows for verification
print(smiles_df.head())

# Load the output CSV for merging
output_df = pd.read_csv('F-output.csv', dtype={'np_id': str, 'assay_cell_type': str}, low_memory=False)

# Clean column names in the output DataFrame
output_df.columns = output_df.columns.str.strip()
output_df['np_id'] = output_df['np_id'].astype(str).str.strip()

# Merge the dataframes on 'np_id'
merged_df = pd.merge(smiles_df, output_df[['np_id', 'assay_cell_type']], on='np_id', how='inner')

# Select the required columns and save the result
result_df = merged_df[['np_id', 'SMILES', 'assay_cell_type']]
result_df.to_csv('new3_R_output.csv', index=False)

print("Merge successful. File 'new3_R_output.csv' has been created.")


# Load the CSV file into a DataFrame
df = pd.read_csv('new3_R_output.csv')

# Remove exact duplicates of 'np_id', 'SMILES', and 'assay_cell_type' (keeping the first occurrence)
df_unique = df.drop_duplicates(subset=['np_id', 'SMILES', 'assay_cell_type'], keep='first')

# Save the cleaned data to a new CSV file
df_unique.to_csv('cleaned3_R_cell_line_commomn_smiles.csv', index=False)

print("Duplicate assay_cell_type entries removed. File 'cleaned_cell_line_commomn_smiles.csv' has been created.")
import csv
from collections import Counter

# Step 1: Read the CSV file
file_path = '/cleaned_cell_line_commomn_smiles.csv'
with open(file_path, mode='r') as file:
    reader = csv.DictReader(file)
    # Step 2: Count the occurrences of each assay_cell_type
    cell_type_counts = Counter(row['assay_cell_type'] for row in reader)

# Step 3: Find the most frequent cell line
most_common_cell_type, most_common_count = cell_type_counts.most_common(1)[0]

# Step 4: Find unique cell lines
unique_cell_lines = [cell_type for cell_type, count in cell_type_counts.items() if count == 1]

print(f"The most frequent assay_cell_type is '{most_common_cell_type}' with {most_common_count} occurrences.")
print(f"The unique assay_cell_types are: {unique_cell_lines}")

# Function to check if a SMILES string is valid
def is_valid_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception as e:
        print(f"Error validating SMILES '{smiles}': {e}")
        return False

# Function to canonicalize a column of SMILES
def canonicalize_smiles_column(smiles_column):
    canonical_smiles = []
    for smi in smiles_column:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                canonical_smi = Chem.MolToSmiles(mol, canonical=True)
                canonical_smiles.append(canonical_smi)
            else:
                canonical_smiles.append(None)
        except Exception as e:
            print(f"Error canonicalizing SMILES '{smi}': {e}")
            canonical_smiles.append(None)
    return canonical_smiles

# Load datasets
common_smiles_file = "unique3_npass_smiles.csv"
npass_structure_file = "NPASSstructureInfo.csv"

# Ensure the files are read into dataframes
try:
    common_smiles_df = pd.read_csv(common_smiles_file)
    npass_structure_df = pd.read_csv(npass_structure_file)
except FileNotFoundError as e:
    print(f"Error loading file: {e}")
    exit()

# Check that the necessary SMILES columns exist
if 'SMILES' not in common_smiles_df.columns or 'SMILES' not in npass_structure_df.columns:
    print("Error: Both input files must have a 'SMILES' column.")
    exit()

# Remove duplicate SMILES from NPASSstructureInfo.csv before canonicalization
npass_structure_df = npass_structure_df.drop_duplicates(subset=['SMILES'])

# Canonicalize SMILES columns for both datasets
print("Canonicalizing SMILES for common_smiles.csv...")
common_smiles_df['Canonical_SMILES'] = canonicalize_smiles_column(common_smiles_df['SMILES'])

print("Canonicalizing SMILES for NPASSstructureInfo.csv...")
npass_structure_df['Canonical_SMILES'] = canonicalize_smiles_column(npass_structure_df['SMILES'])

# Drop rows with invalid or uncanonicalizable SMILES
common_smiles_df = common_smiles_df.dropna(subset=['Canonical_SMILES'])
npass_structure_df = npass_structure_df.dropna(subset=['Canonical_SMILES'])

# Debugging: Check for duplicates in canonicalized NPASSstructureInfo.csv
duplicate_smiles_count = npass_structure_df['Canonical_SMILES'].duplicated().sum()
if duplicate_smiles_count > 0:
    print(f"Warning: {duplicate_smiles_count} duplicate canonical SMILES found in NPASSstructureInfo.csv. These will be removed.")
    npass_structure_df = npass_structure_df.drop_duplicates(subset=['Canonical_SMILES'])

# Find matching SMILES
print("Finding matching SMILES...")
matching_smiles = common_smiles_df['Canonical_SMILES'].isin(npass_structure_df['Canonical_SMILES'])
matched_common_smiles = common_smiles_df[matching_smiles]

# Debugging: Check unmatched SMILES
unmatched_smiles = common_smiles_df[~matching_smiles]
print(f"Number of unmatched SMILES: {len(unmatched_smiles)}")

# Merge with NPASS structure to get np_id
print("Merging matched SMILES with NPASSstructureInfo to get np_id...")
merged_df = pd.merge(
    matched_common_smiles, 
    npass_structure_df[['np_id', 'Canonical_SMILES']], 
    on='Canonical_SMILES'
)

# Debugging: Check for extra rows in the merged dataframe
extra_smiles = set(merged_df['Canonical_SMILES']) - set(common_smiles_df['Canonical_SMILES'])
if extra_smiles:
    print(f"Warning: {len(extra_smiles)} extra SMILES found in the merged dataframe. These may indicate issues with duplicates in NPASSstructureInfo.csv.")

# Save results
output_file = "all_unique_npass_with_np_id.csv"
merged_df.to_csv(output_file, index=False)

print(f"Saved matched SMILES with np_id to {output_file}")


def match_and_create_csv(clean_file, output_file, result_file):
    # Load the CSV files
    clean_data = pd.read_csv(clean_file)
    output_data = pd.read_csv(output_file)
    
    # Merge the two files on 'np_id'
    merged_data = pd.merge(clean_data, output_data, on='np_id', how='inner')
    
    # Debug: Print column names to check available columns
    print("Merged data columns:", merged_data.columns)
    
    # Correct column selection based on actual column names
    required_columns = ['SMILES', 'np_id', 'activity_type', 'activity_value', 'activity_units']
    if 'assay_cell' in merged_data.columns:
        required_columns.append('assay_cell')
    elif 'assay_cell_type' in merged_data.columns:
        required_columns.append('assay_cell_type')  # Adjust for possible column name

    # Filter for the required columns
    try:
        filtered_data = merged_data[required_columns]
    except KeyError as e:
        print(f"Error: Some columns are missing. Available columns: {merged_data.columns}")
        raise e
    
    # Save the result to a new CSV file
    filtered_data.to_csv(result_file, index=False)
    print(f"Matched data saved to {result_file}. Total rows: {len(filtered_data)}")

# File paths
clean_file = "all_unique_npass_with_np_id.csv"
output_file = "output.csv"  # Replace with the actual output.csv file path
result_file = "m_finall_unique_npass.csv"  # New name for the final CSV file

# Run the function
match_and_create_csv(clean_file, output_file, result_file)


def match_and_create_csv(clean_file, output_file, result_file):
    """
    Matches two CSV files on 'np_id' and saves the selected columns to a new file.

    Parameters:
    clean_file (str): Path to the first CSV file (e.g., clean_all_unique_npass_with_np_id.csv).
    output_file (str): Path to the second CSV file (e.g., F-output.csv).
    result_file (str): Path to save the resulting CSV file (e.g., FN.csv).
    """
    # Load the CSV files
    clean_data = pd.read_csv(clean_file)
    output_data = pd.read_csv(output_file)
    
    # Merge the two files on 'np_id'
    merged_data = pd.merge(clean_data, output_data, on='np_id', how='inner')
    
    # Debug: Print column names to check available columns
    print("Merged data columns:", merged_data.columns)
    
    # Define required columns based on column availability
    required_columns = ['SMILES', 'np_id', 'activity_type', 'activity_value', 'activity_units']
    if 'assay_cell' in merged_data.columns:
        required_columns.append('assay_cell')
    elif 'assay_cell_type' in merged_data.columns:
        required_columns.append('assay_cell_type')  # Adjust for possible column name

    # Filter for the required columns
    try:
        filtered_data = merged_data[required_columns]
    except KeyError as e:
        print(f"Error: Some columns are missing. Available columns: {merged_data.columns}")
        raise e
    
    # Save the result to a new CSV file
    filtered_data.to_csv(result_file, index=False)
    print(f"Matched data saved to {result_file}. Total rows: {len(filtered_data)}")

# File paths
clean_file = "all_unique_npass_with_np_id.csv"  # Input CSV 1
output_file = "F-output.csv"  # Input CSV 2
result_file = "FN.csv"  # Output CSV file

# Run the function
match_and_create_csv(clean_file, output_file, result_file)
