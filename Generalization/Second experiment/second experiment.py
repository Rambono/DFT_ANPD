import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from keras.models import load_model
import json
from tqdm import tqdm
from rdkit import Chem
import tensorflow as tf  # Import TensorFlow

# Custom loss function
def weighted_binary_crossentropy(y_true, y_pred):
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
    loss = -tf.reduce_mean(1.5 * y_true * tf.math.log(y_pred) + 0.7 * (1 - y_true) * tf.math.log(1 - y_pred))
    return loss

# Step 1: Load the New Dataset

# Load the new dataset
data2_path = 'Data2.csv'
print("Starting data loading...")
data2 = pd.read_csv(data2_path)
print("Data loaded. Shape:", data2.shape)

# Extract SMILES strings and labels
smiles_list = data2['SMILES'].tolist()
labels = data2['label'].tolist()

# Normalize SMILES strings before processing
def normalize_smiles(smiles, original_single_chars):
    """Normalize SMILES strings to ensure consistent representation"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol, canonical=True)
    return ''.join([char if char in original_single_chars else '' for char in smiles])

original_char_set = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'B', 'Br', 'H', 'I', '#', '=', '-', '(', ')', '[', ']', 'n', 's', 'r', 'l', 'o', 'c', 'p', 'f', 'cl', 'br', 'i', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '.', '+', '-', '', '/']

# Convert original_char_set to a set of single characters
original_single_chars = set()
for char in original_char_set:
    original_single_chars.update(char)

smiles_list = [normalize_smiles(smiles, original_single_chars) for smiles in smiles_list]

# Find new characters
unique_chars = set()
for smiles in smiles_list:
    unique_chars.update(smiles)

new_chars = unique_chars - original_single_chars
print(f"New characters found: {new_chars}")

# Step 2: Expand the char_set and Update char_to_idx

# Expand char_set with new characters (this step is unnecessary now if unknown characters are removed)
expanded_char_set = original_char_set.copy()
for char in new_chars:
    expanded_char_set.append(char)

# Create char_to_idx dictionary
# We don't need this now since unknown characters are removed
# char_to_idx = {char: idx for idx, char in enumerate(expanded_char_set)}
# # Define a special token for unknown characters
# char_to_idx['UNK'] = len(char_to_idx)

# Step 3: Update Preprocessing Functions

def smiles_to_idx(smiles, original_single_chars):
    idx_vector = []
    for char in smiles:
        if char in original_single_chars:
            idx_vector.append(original_char_set.index(char))
    return idx_vector

def pad_idx_vector(idx_vector, max_len):
    if len(idx_vector) < max_len:
        padded_idx_vector = idx_vector + [0] * (max_len - len(idx_vector))
    else:
        padded_idx_vector = idx_vector[:max_len]
    return padded_idx_vector

# Step 4: Load and Prepare the New Data

# Convert SMILES to index vectors
smiles_max_len = 100  # Ensure this matches the model's input shape
drug_smiles = [pad_idx_vector(smiles_to_idx(smiles, original_single_chars), smiles_max_len) for smiles in smiles_list]

# Initialize tokenizer and model outside the function
tokenizer = AutoTokenizer.from_pretrained('unikei/bert-base-smiles')
model = AutoModel.from_pretrained('unikei/bert-base-smiles')

def extract_features(smiles_list, tokenizer, model, batch_size=32):
    all_outputs = []
    total_batches = (len(smiles_list) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(smiles_list), batch_size), desc="Extracting features", total=total_batches):
        batch_smiles = smiles_list[i:i+batch_size]
        inputs = tokenizer(batch_smiles, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        all_outputs.append(outputs.pooler_output.numpy())
    return np.concatenate(all_outputs, axis=0)

print("Extracting features...")
features_data2 = extract_features(smiles_list, tokenizer, model)
print("Features extracted.")

# Step 5: Load the Pre-trained Model

# Define the directory to save checkpoints
checkpoint_dir = 'c'

# Load the best model
best_model_path = os.path.join(checkpoint_dir, 'best_model.keras')

print("Loading the model...")
best_model = load_model(best_model_path, custom_objects={'weighted_binary_crossentropy': weighted_binary_crossentropy})
print("Model loaded.")

# Print model summary for debugging
best_model.summary()

# Step 6: Evaluate the Model on the New Data

# Prepare the input data for the model
x_data2 = {'drug1_input': np.array(drug_smiles), 'llm_input': features_data2}
y_data2 = np.array(labels)

# Print the first few encoded SMILES
print(drug_smiles[:5])

# Check input shapes
print("drug_smiles shape:", np.array(drug_smiles).shape)
print("features_data2 shape:", features_data2.shape)

print("Evaluating the model...")
test_results = best_model.evaluate(x_data2, y_data2)
print("\nFinal Test Results on Data2.csv:")
for metric, value in zip(best_model.metrics_names, test_results):
    print(f"{metric}: {value:.4f}")

# Predict on the new dataset
print("Predicting...")
y_pred = best_model.predict(x_data2).ravel()
print("Prediction complete.")

# Step 7: Sort Predictions and Extract Top 10

# Create a list of tuples (smiles, prediction)
predictions_with_smiles = list(zip(smiles_list, y_pred))

# Sort the predictions in descending order
predictions_with_smiles_sorted = sorted(predictions_with_smiles, key=lambda x: x[1], reverse=True)

# Extract the top 10 predictions
top_10_predictions = predictions_with_smiles_sorted[:10]

# Print the top 10 predictions
print("\nTop 10 Predictions:")
for smiles, pred in top_10_predictions:
    print(f"SMILES: {smiles}, Prediction: {pred:.4f}")

# Step 8: Save Top 10 Predictions to JSON File

# Define the directory to save results
result_dir = 'm_last'
os.makedirs(result_dir, exist_ok=True)

# Save the top 10 predictions
top_10_dict = {
    'top_10_predictions': [{'SMILES': smiles, 'Prediction': float(pred)} for smiles, pred in top_10_predictions]
}

with open(os.path.join(result_dir, 'top_10_predictions.json'), 'w') as f:
    json.dump(top_10_dict, f, indent=4)

print("Top 10 predictions saved to 'm2' directory.")

# Step 9: Save All Predictions to JSON File

# Save all predictions
all_predictions_dict = {
    'all_predictions': [{'SMILES': smiles, 'Prediction': float(pred)} for smiles, pred in predictions_with_smiles_sorted]
}

with open(os.path.join(result_dir, 'all_predictions.json'), 'w') as f:
    json.dump(all_predictions_dict, f, indent=4)

print("All predictions saved to 'm2' directory.")
