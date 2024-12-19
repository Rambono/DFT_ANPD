import os
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from transformers import AutoTokenizer, AutoModel
from keras.models import load_model
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, matthews_corrcoef
import tensorflow as tf
import json
from tqdm import tqdm

# Custom loss function
def weighted_binary_crossentropy(y_true, y_pred):
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
    loss = -tf.reduce_mean(1.5 * y_true * tf.math.log(y_pred) + 0.7 * (1 - y_true) * tf.math.log(1 - y_pred))
    return loss

# Original char_set and char_to_idx based on the training data
char_set = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'B', 'Br', 'H', 'I', '#', '=', '-', '(', ')', '[', ']', 'n', 's', 'r', 'l', 'o', 'c', 'p', 'f', 'cl', 'br', 'i', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '.', '+', '-', '', '/']
char_to_idx = {char: i for i, char in enumerate(char_set)}
char_to_idx['UNK'] = len(char_to_idx)  # Add a special token for unknown characters

# Function to convert SMILES to index vectors
def smiles_to_idx(smiles, char_to_idx):
    idx_vector = []
    for char in smiles:
        if char in char_to_idx:
            idx_vector.append(char_to_idx[char])
        else:
            # Handle unknown characters by using the 'UNK' token
            idx_vector.append(char_to_idx['UNK'])
    return idx_vector

# Function to pad index vectors
def pad_idx_vector(idx_vector, max_len):
    if len(idx_vector) < max_len:
        padded_idx_vector = idx_vector + [0] * (max_len - len(idx_vector))
    else:
        padded_idx_vector = idx_vector[:max_len]
    return padded_idx_vector

# Load the new dataset
data1_path = 'Data1.csv'
print("Starting data loading...")
data1 = pd.read_csv(data1_path)
print("Data loaded. Shape:", data1.shape)

# Extract SMILES strings and labels
smiles_list = data1['SMILES'].tolist()
labels = data1['label'].tolist()

# Convert SMILES to index vectors and pad them
smiles_max_len = 100  # Ensure this matches the model's input shape
drug_smiles = [pad_idx_vector(smiles_to_idx(smiles, char_to_idx), smiles_max_len) for smiles in tqdm(smiles_list, desc="Converting SMILES to index vectors")]

# Initialize tokenizer and model outside the function
tokenizer = AutoTokenizer.from_pretrained('unikei/bert-base-smiles')
model = AutoModel.from_pretrained('unikei/bert-base-smiles')

# Function to extract features using the BERT model
def extract_features(smiles_list, tokenizer, model, batch_size=64):
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
features_data1 = extract_features(smiles_list, tokenizer, model)
print("Features extracted.")

# Load the best model
best_model_path = 'c/best_model.keras'

print("Loading the model...")
best_model = load_model(best_model_path, custom_objects={'weighted_binary_crossentropy': weighted_binary_crossentropy})
print("Model loaded.")

# Prepare the input data for the model
x_data1 = {'drug1_input': np.array(drug_smiles), 'llm_input': features_data1}
y_data1 = np.array(labels)

# Check input shapes
print("drug_smiles shape:", np.array(drug_smiles).shape)
print("features_data1 shape:", features_data1.shape)

# Evaluate the model on the new data
print("Evaluating the model...")
test_results = best_model.evaluate(x_data1, y_data1)
print("\nFinal Test Results on Data1_final.csv:")
for metric, value in zip(best_model.metrics_names, test_results):
    print(f"{metric}: {value:.4f}")

# Predict on the new dataset
print("Predicting...")
y_pred = best_model.predict(x_data1).ravel()
print("Prediction complete.")

# Convert predictions to binary labels based on the threshold 0.5
y_pred_binary = (y_pred > 0.5).astype(int)

# Calculate metrics
auc_roc = roc_auc_score(y_data1, y_pred)
auc_pr = average_precision_score(y_data1, y_pred)
balanced_acc = balanced_accuracy_score(y_data1, y_pred_binary)
mcc = matthews_corrcoef(y_data1, y_pred_binary)

# Print the metrics
print("\nEvaluation Metrics on Data1.csv:")
print(f"AUC-ROC: {auc_roc:.4f}")
print(f"AUC-PR: {auc_pr:.4f}")
print(f"Balanced Accuracy: {balanced_acc:.4f}")
print(f"MCC: {mcc:.4f}")

# Directory to save evaluation results
eval_dir = ''
os.makedirs(eval_dir, exist_ok=True)

# Save evaluation metrics
metrics = {
    'AUC-ROC': float(auc_roc),
    'AUC-PR': float(auc_pr),
    'Balanced Accuracy': float(balanced_acc),
    'MCC': float(mcc)
}

with open(os.path.join(eval_dir, 'evaluation_metrics.json'), 'w') as f:
    json.dump(metrics, f)

# Save true labels, predicted probabilities, and predicted labels
labels_predictions = {
    'true_labels': y_data1.tolist(),
    'predicted_probabilities': y_pred.tolist(),
    'predicted_labels': y_pred_binary.tolist()
}

with open(os.path.join(eval_dir, 'labels_predictions.json'), 'w') as f:
    json.dump(labels_predictions, f)

print("Evaluation metrics and labels saved to 'm_g' directory.")
