import os
import json
from sklearn.metrics import confusion_matrix, matthews_corrcoef
import matplotlib.pyplot as plt
import seaborn as sns

# Define the data directory
DATA_DIR = "metrics"

def load_labels(file_path):
    full_path = os.path.join(DATA_DIR, file_path)
    if not os.path.exists(full_path):
        print(f"File not found: {full_path}")
        return [], []
    try:
        with open(full_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in {full_path}: {e}")
        return [], []
    true_labels = data.get('true_labels', [])
    predicted_probs = data.get('predicted_labels', [])
    if not true_labels or not predicted_probs:
        print(f"Missing 'true_labels' or 'predicted_labels' in {full_path}.")
        return [], []
    try:
        true_labels = [int(label) for label in true_labels]
        predicted_probs = [float(prob) for prob in predicted_probs]
    except (ValueError, TypeError) as e:
        print(f"Error converting labels in {full_path}: {e}")
        return [], []
    return true_labels, predicted_probs

def convert_to_labels(predicted_probs, threshold=0.5):
    return [1 if prob >= threshold else 0 for prob in predicted_probs]

def main():
    all_true_labels = []
    all_predicted_labels = []
    for fold in range(1, 6):
        file_name = f"fold_{fold}_labels.json"
        true_labels, predicted_probs = load_labels(file_name)
        if len(true_labels) != len(predicted_probs):
            print(f"Error in fold {fold}: true_labels and predicted_probs have different lengths.")
            continue
        predicted_labels = convert_to_labels(predicted_probs)
        all_true_labels.extend(true_labels)
        all_predicted_labels.extend(predicted_labels)
    # Compute confusion matrix
    cm = confusion_matrix(all_true_labels, all_predicted_labels)
    # Plot confusion matrix
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['True 0', 'True 1'])
    plt.title('Confusion Matrix Across All Folds')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('confusion_matrix_plot.png')
    plt.show()

if __name__ == "__main__":
    main()