import json
import matplotlib.pyplot as plt
import os

# Initialize lists to store metrics for all folds
all_auc_roc = []
all_auc_pr = []
all_balanced_accuracy = []
all_mcc = []

# Function to read the JSON file and extract metrics
def read_json_file(file_path):
    with open(file_path, 'r') as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {file_path}: {e}")
            return None, None, None, None
    try:
        return data['auc_roc'], data['auc_pr'], data['balanced_accuracy'], data['mcc']
    except KeyError as e:
        print(f"Key not found in {file_path}: {e}")
        return None, None, None, None

# Read the metrics from each fold's JSON file
for fold in range(1, 6):
    file_path = os.path.join('metrics', f'fold_{fold}_metrics.json')
    print(f"Checking file: {file_path}")
    if os.path.exists(file_path):
        print("File exists.")
        auc_roc, auc_pr, balanced_accuracy, mcc = read_json_file(file_path)
        if auc_roc is not None and auc_pr is not None and balanced_accuracy is not None and mcc is not None:
            all_auc_roc.append(auc_roc)
            all_auc_pr.append(auc_pr)
            all_balanced_accuracy.append(balanced_accuracy)
            all_mcc.append(mcc)
        else:
            print(f"Skipping fold {fold} due to missing data.")
    else:
        print(f"File not found: {file_path}")

# Plot separate plots for each metric
plt.figure(figsize=(14, 10))

# AUC-ROC
plt.subplot(2, 2, 1)
plt.plot(range(1, len(all_auc_roc)+1), all_auc_roc, marker='o', color='blue')
plt.title('AUC-ROC Across Folds')
plt.xlabel('Fold')
plt.ylabel('AUC-ROC')
plt.grid(True)

# AUC-PR
plt.subplot(2, 2, 2)
plt.plot(range(1, len(all_auc_pr)+1), all_auc_pr, marker='s', color='green')
plt.title('AUC-PR Across Folds')
plt.xlabel('Fold')
plt.ylabel('AUC-PR')
plt.grid(True)

# Balanced Accuracy
plt.subplot(2, 2, 3)
plt.plot(range(1, len(all_balanced_accuracy)+1), all_balanced_accuracy, marker='^', color='red')
plt.title('Balanced Accuracy Across Folds')
plt.xlabel('Fold')
plt.ylabel('Balanced Accuracy')
plt.grid(True)

# MCC
plt.subplot(2, 2, 4)
plt.plot(range(1, len(all_mcc)+1), all_mcc, marker='d', color='purple')
plt.title('MCC Across Folds')
plt.xlabel('Fold')
plt.ylabel('MCC')
plt.grid(True)

plt.tight_layout()
plt.savefig('separate2_metrics_plot.png')
plt.show()

# Plot all metrics in one plot
plt.figure(figsize=(14, 10))

plt.plot(range(1, len(all_auc_roc)+1), all_auc_roc, marker='o', color='blue', label='AUC-ROC')
plt.plot(range(1, len(all_auc_pr)+1), all_auc_pr, marker='s', color='green', label='AUC-PR')
plt.plot(range(1, len(all_balanced_accuracy)+1), all_balanced_accuracy, marker='^', color='red', label='Balanced Accuracy')
plt.plot(range(1, len(all_mcc)+1), all_mcc, marker='d', color='purple', label='MCC')

plt.title('Metrics Across All Folds')
plt.xlabel('Fold')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('combined2_metrics_plot.png')
plt.show()