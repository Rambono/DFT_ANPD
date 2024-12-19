import json
import matplotlib.pyplot as plt
import os

# Specify the file path
file_path = os.path.join('m', 'final_metrics.json')

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

# Check if the file exists and read metrics
if os.path.exists(file_path):
    print("File exists.")
    auc_roc, auc_pr, balanced_accuracy, mcc = read_json_file(file_path)
    if auc_roc is not None and auc_pr is not None and balanced_accuracy is not None and mcc is not None:
        # Prepare data for plotting
        metrics = {
            'AUC-ROC': auc_roc,
            'AUC-PR': auc_pr,
            'Balanced Accuracy': balanced_accuracy,
            'MCC': mcc
        }
    else:
        print("Skipping due to missing data.")
else:
    print(f"File not found: {file_path}")

# Plot the metrics
if 'metrics' in locals():
    # Plot separate plots for each metric
    plt.figure(figsize=(14, 10))

    # AUC-ROC
    plt.subplot(2, 2, 1)
    plt.bar(1, metrics['AUC-ROC'], color='blue')
    plt.title('AUC-ROC')
    plt.xticks([])
    plt.ylabel('Value')
    plt.grid(True)

    # AUC-PR
    plt.subplot(2, 2, 2)
    plt.bar(1, metrics['AUC-PR'], color='green')
    plt.title('AUC-PR')
    plt.xticks([])
    plt.ylabel('Value')
    plt.grid(True)

    # Balanced Accuracy
    plt.subplot(2, 2, 3)
    plt.bar(1, metrics['Balanced Accuracy'], color='red')
    plt.title('Balanced Accuracy')
    plt.xticks([])
    plt.ylabel('Value')
    plt.grid(True)

    # MCC
    plt.subplot(2, 2, 4)
    plt.bar(1, metrics['MCC'], color='purple')
    plt.title('MCC')
    plt.xticks([])
    plt.ylabel('Value')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('separate_metrics_plot.png')
    plt.show()

    # Plot all metrics in one plot
    plt.figure(figsize=(10, 6))

    plt.bar(range(1, 5), list(metrics.values()), tick_label=list(metrics.keys()), color=['blue', 'green', 'red', 'purple'])

    plt.title('Metrics for Experiment')
    plt.ylabel('Value')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('combined_metrics_plot.png')
    plt.show()