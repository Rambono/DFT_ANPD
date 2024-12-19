

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
# %cd /content/drive/My Drive/Ramboud
!pip install rdkit
from chemtonic_released.chemtonic.curation.refinement import refineComplete


# Load active/inactive compounds
active_compounds   = pd.read_csv("/2021-iANP-EC/data/original_data/AnticancerNP_active.csv")['SMILES'].tolist()
inactive_compounds = pd.read_csv("/2021-iANP-EC/data/original_data/AnticancerNP_inactive.csv")['SMILES'].tolist()

# Fullly refine
refined_actives   = active_compounds#refineComplete(active_compounds)
refined_inactives = inactive_compounds#refineComplete(inactive_compounds)

# Remove conflicted labels
active_labels   = [1]*len(refined_actives)
inactive_labels = [0]*len(refined_inactives)

#refined_compounds = refined_actives['SMILES'].tolist() + refined_inactives['SMILES'].tolist()
refined_compounds = pd.read_csv("/2021-iANP-EC/data/refined_data/refined_anticancerNP.csv")['SMILES'].tolist()
refined_labels    = pd.read_csv("/2021-iANP-EC/data/refined_data/refined_anticancerNP.csv")['class'].tolist()

print(len(refined_compounds))
print(len(refined_labels))

all_refind = pd.read_csv("/2021-iANP-EC/data/refined_data/refined_anticancerNP.csv")
# Check number of compounds per class
num_active = len(all_refind[all_refind['class']==1]) #367
num_inactive = len(all_refind[all_refind['class']==0]) #630

num_active , num_inactive

# Convert targets to tensor
import torch
targets_tensor = torch.tensor(refined_labels).float()

from sklearn.utils import shuffle
refined_compounds, targets_tensor = shuffle(refined_compounds, targets_tensor, random_state=42)

#!pip install rdkit-pypi
import rdkit
from rdkit import Chem

molecules = [Chem.MolFromSmiles(smile) for smile in refined_compounds]

!pip install torch-geometric
from torch_geometric.data import Data

def mol_to_graph(mol):
    atom_features = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    for i, atom in enumerate(mol.GetAtoms()):
      print(atom)
    atom_row, atom_col = zip(*[(i, j) for i, atom in enumerate(mol.GetAtoms()) for j, _ in enumerate(atom.GetNeighbors())])
    edge_index = torch.tensor([atom_row, atom_col], dtype=torch.long).t().contiguous()
    return Data(x=torch.tensor(atom_features, dtype=torch.float).reshape([-1,1]), edge_index=torch.transpose(edge_index, 0,1))

graphs = [mol_to_graph(mol) for mol in molecules]
for mol in molecules:
    for i, atom in enumerate(mol.GetAtoms()):
      for j, _ in enumerate(atom.GetNeighbors()):
         print(atom)

from transformers import AutoTokenizer, AutoModel
from keras.layers import Input, Dense, Dropout, Concatenate, Conv1D, GlobalMaxPooling1D, Embedding, GlobalMaxPooling1D
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import KFold
from sklearn.utils import class_weight
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef
from keras.callbacks import EarlyStopping
import numpy as np
import tensorflow as tf
from keras import backend as K
import os, sys, re, math, time, scipy
import argparse
from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import json
import pickle
import collections
from sklearn import metrics
from sklearn.metrics import r2_score
from collections import OrderedDict
#import tensorflow_addons
import tensorflow as tf, keras
#import tensorflow_addons as tfa
from tensorflow.keras import regularizers
from keras import layers
from keras import utils
from keras.losses import MeanSquaredError
from keras.metrics import MeanSquaredError
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping,LearningRateScheduler
from keras.layers import Input, Dense, Average, Dropout, Concatenate, Conv1D, GlobalAveragePooling1D, Embedding, GlobalMaxPooling1D, Maximum, Add

from copy import deepcopy
import tensorflow.keras.backend as K
from keras.layers import Layer
from tensorflow.python.framework import ops
from keras.callbacks import ModelCheckpoint
import os
from keras.callbacks import Callback

from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, roc_auc_score, average_precision_score


char_set = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'B', 'Br', 'H', 'I', '#', '=', '-', '(', ')', '[', ']', 'n', 's', 'r', 'l', 'o', 'c', 'p', 'f', 'cl', 'br', 'i', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '.', '+', '-', '', '/']
char_to_idx = {char: i for i, char in enumerate(char_set)}

def smiles_to_idx(smiles):
    idx_vector = []
    for char in smiles:
        if char in char_to_idx:
            idx_vector.append(char_to_idx[char])
        else:
            raise ValueError(f"Unknown character '{char}' in SMILES string")
    return idx_vector

def pad_idx_vector(idx_vector, max_len):
    if len(idx_vector) <= max_len:
        padded_idx_vector = idx_vector + [0] * (max_len - len(idx_vector))
    else:
        padded_idx_vector = idx_vector[0:max_len]
    return padded_idx_vector

model_name = 'unikei/bert-base-smiles'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def extract_features(smiles_list):
    inputs = tokenizer(smiles_list, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    return outputs.pooler_output

smiles_max_len = 100
smiles_dict_len = 44

Drug1_input = Input(shape=(smiles_max_len,), dtype='int32', name='drug1_input')

encode_smiles_layer1 = Embedding(input_dim=smiles_dict_len + 1, output_dim=128, input_length=smiles_max_len, name='smiles_embedding')
encode_smiles_layer2 = Conv1D(filters=32, kernel_size=4, activation='relu', padding='valid', strides=1, name='conv1_smiles')
encode_smiles_layer3 = Conv1D(filters=32 * 2, kernel_size=4, activation='relu', padding='valid', strides=1, name='conv2_smiles')
encode_smiles_layer4 = Conv1D(filters=32 * 3, kernel_size=4, activation='relu', padding='valid', strides=1, name='conv3_smiles')
encode_smiles_layer5 = GlobalMaxPooling1D()

drug1_layer1 = encode_smiles_layer1(Drug1_input)
drug1_layer2 = encode_smiles_layer2(drug1_layer1)
drug1_layer3 = encode_smiles_layer3(drug1_layer2)
drug1_layer4 = encode_smiles_layer4(drug1_layer3)
drug1_layer5 = encode_smiles_layer5(drug1_layer4)

llm_input = Input(shape=(768,), name='llm_input')
encode_llm_layer1 = Dense(32, activation='relu')(llm_input)
encode_llm_layer2 = Dense(32 * 2, activation='relu')(encode_llm_layer1)
encode_llm_layer3 = Dense(32 * 3, activation='relu')(encode_llm_layer2)

attention_weights = Dense(32 * 3, activation='softmax')(layers.Add()([drug1_layer5, encode_llm_layer3]))

attention_applied1 = layers.Multiply()([drug1_layer5, attention_weights])
attention_applied2 = layers.Multiply()([encode_llm_layer3, attention_weights])

embedding = layers.Add()([attention_applied1, attention_applied2])

FC1 = Dense(512, activation='relu', name='dense1_')(embedding)
FC2 = Dropout(0.1)(FC1)
FC2 = Dense(256, activation='relu', name='dense2_')(FC2)
FC2 = Dropout(0.1)(FC2)
FC3 = Dense(1, activation='sigmoid', name='dense3_')(FC2)

model_pred = Model(inputs=[Drug1_input, llm_input], outputs=[FC3])

def generate_data(batch_size, indices, class_weights=None):
    while True:  # This makes the generator repeat indefinitely
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:min(i + batch_size, len(indices))]
            smiles_lst = [refined_compounds[index] for index in batch_idx]
            lbl = targets_tensor[batch_idx]
            features_train = extract_features(smiles_lst)
            drug_smiles = [pad_idx_vector(smiles_to_idx(smiles), 100) for smiles in smiles_lst]
            if class_weights is not None:
                sample_weights = np.array([class_weights[int(label)] for label in lbl])
                yield {'drug1_input': np.array(drug_smiles), 'llm_input': features_train.detach().numpy()}, np.array(lbl), sample_weights
            else:
                yield {'drug1_input': np.array(drug_smiles), 'llm_input': features_train.detach().numpy()}, np.array(lbl)

def weighted_binary_crossentropy(y_true, y_pred):
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1 - epsilon)  # Clipping prediction
    loss = -K.mean(1.5 * y_true * K.log(y_pred) + 0.7 * (1 - y_true) * K.log(1 - y_pred))
    return loss

adam = Adam(learning_rate=0.001, clipnorm=1.0)
model_pred.compile(optimizer=adam, loss=[weighted_binary_crossentropy], metrics=['accuracy', keras.metrics.AUC(), keras.metrics.AUC(curve='PR')])

# Define the directory to save metrics and state
metrics_dir = 'Experiment_number_two/m'
os.makedirs(metrics_dir, exist_ok=True)

# Load or initialize the state
state_file = os.path.join(metrics_dir, 'training_state.json')
if os.path.exists(state_file):
    with open(state_file, 'r') as f:
        state = json.load(f)
    current_epoch = state['current_epoch']
    auc_roc_scores = state['auc_roc_scores']
    auc_pr_scores = state['auc_pr_scores']
    balanced_accuracies = state['balanced_accuracies']
    mcc_scores = state['mcc_scores']
else:
    current_epoch = 0
    auc_roc_scores = []
    auc_pr_scores = []
    balanced_accuracies = []
    mcc_scores = []

# Shuffle and split the dataset into train, validation, and test sets.
refined_compounds, targets_tensor = shuffle(refined_compounds, targets_tensor, random_state=42)

# Select a fixed test set with 36 anticancer and 64 non-anticancer
test_size = 100
while True:
    test_idx = np.random.choice(len(refined_compounds), test_size, replace=False)
    test_labels = targets_tensor[test_idx]
    test_active_count = torch.sum(test_labels == 1).item()
    test_inactive_count = torch.sum(test_labels == 0).item()
    if test_active_count == 36 and test_inactive_count == 64:
        break

# Define remaining data for train and validation split
train_val_idx = np.setdiff1d(np.arange(len(refined_compounds)), test_idx)
train_val_data = [refined_compounds[i] for i in train_val_idx]
train_val_labels = targets_tensor[train_val_idx].numpy()

# Separate the dataset into anticarcinogens and non-anticarcinogens
anticarcinogens = pd.DataFrame({'SMILES': refined_compounds, 'class': refined_labels})[pd.DataFrame({'SMILES': refined_compounds, 'class': refined_labels})['class'] == 1]
non_anticarcinogens = pd.DataFrame({'SMILES': refined_compounds, 'class': refined_labels})[pd.DataFrame({'SMILES': refined_compounds, 'class': refined_labels})['class'] == 0]

# Selecting samples for validation
anticarcinogens_val = anticarcinogens.sample(n=42, random_state=42)
non_anticarcinogens_val = non_anticarcinogens.sample(n=58, random_state=42)

# Creating the validation set
validation_set = pd.concat([anticarcinogens_val, non_anticarcinogens_val])
validation_set = validation_set.sample(frac=1, random_state=42).reset_index(drop=True)

# Remaining samples go to the training set
anticarcinogens_train = anticarcinogens.drop(anticarcinogens_val.index)
non_anticarcinogens_train = non_anticarcinogens.drop(non_anticarcinogens_val.index)
training_set = pd.concat([anticarcinogens_train, non_anticarcinogens_train])
training_set = training_set.sample(frac=1, random_state=42).reset_index(drop=True)

# Convert to lists
train_idx = training_set.index.tolist()
val_idx = validation_set.index.tolist()

print("Train size:", len(train_idx))
print("Validation size:", len(val_idx))
print("Test size:", len(test_idx), " - Anticancer:", test_active_count, ", Non-anticancer:", test_inactive_count)

# Print the number of anticarcinogens and non-anticarcinogens in both the training and validation sets
print(f"Number of anticarcinogens in training set: {len(anticarcinogens_train)}")
print(f"Number of non-anticarcinogens in training set: {len(non_anticarcinogens_train)}")
print(f"Number of anticarcinogens in validation set: {len(anticarcinogens_val)}")
print(f"Number of non-anticarcinogens in validation set: {len(non_anticarcinogens_val)}")

# Calculate class weights
train_labels = targets_tensor[train_idx].numpy()
unique_classes = np.unique(train_labels).astype(int)  # Use unique labels from train_labels
class_weights = class_weight.compute_class_weight('balanced', classes=unique_classes, y=train_labels.astype(int))
class_weights_dict = dict(enumerate(class_weights))

# Define the directory to save checkpoints
checkpoint_dir = 'Experiment_number_two/c'
os.makedirs(checkpoint_dir, exist_ok=True)

# Define a custom metric to monitor the best model based on multiple metrics
class MultiMetricCallback(tf.keras.callbacks.Callback):
    def __init__(self, monitor_metrics, mode='max'):
        super(MultiMetricCallback, self).__init__()
        self.monitor_metrics = monitor_metrics
        self.mode = mode
        self.best_score = -np.inf if mode == 'max' else np.inf
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs={}):
        current_score = sum([logs[metric] for metric in self.monitor_metrics])
        if (self.mode == 'max' and current_score > self.best_score) or (self.mode == 'min' and current_score < self.best_score):
            self.best_score = current_score
            self.best_epoch = epoch
            self.model.save(os.path.join(checkpoint_dir, 'best_model.keras'))

# Define the metrics to monitor
monitor_metrics = ['val_auc_4', 'val_auc_5', 'val_accuracy']

# Define the callbacks
checkpoint_callback = MultiMetricCallback(monitor_metrics=monitor_metrics, mode='max')
early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=15)

class MetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_data, log_dir):
        self.test_data = test_data
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, 'training_log.txt')

    def on_epoch_end(self, epoch, logs={}):
        x_test, y_test = self.test_data
        y_pred = self.model.predict(x_test)
        y_pred_class = (y_pred > 0.5).astype(int).flatten()

        balanced_acc = balanced_accuracy_score(y_test, y_pred_class)
        mcc = matthews_corrcoef(y_test, y_pred_class)

        log_entry = f'\nEpoch {epoch + 1}\n'
        log_entry += f'Balanced Accuracy: {balanced_acc:.4f}, MCC: {mcc:.4f}\n'
        for metric, value in logs.items():
            log_entry += f'{metric}: {value:.4f}, '
        log_entry += '\n'

        with open(self.log_file, 'a') as f:
            f.write(log_entry)

        print(log_entry)  # Ensure this line is present to print the metrics during training

metrics_callback = MetricsCallback(next(generate_data(len(test_idx), test_idx)), metrics_dir)

# Define a learning rate schedule
def lr_schedule(epoch):
    initial_lr = 0.001
    drop = 0.5
    epochs_drop = 10.0
    lr = initial_lr * (drop ** np.floor((1 + epoch) / epochs_drop))
    return lr

lr_scheduler = LearningRateScheduler(lr_schedule)

# Check for existing checkpoints
checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.keras')])
if checkpoints:
    latest_checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])
    print(f"Resuming training from checkpoint: {latest_checkpoint}")
    model_pred = tf.keras.models.load_model(latest_checkpoint, custom_objects={'weighted_binary_crossentropy': weighted_binary_crossentropy})
    initial_epoch = int(checkpoints[-1].split('_')[-1].split('.')[0])
else:
    print("Starting training from scratch")
    initial_epoch = 0

# Fit model
model_pred.fit(
    generate_data(16, train_idx, class_weights_dict),
    epochs=100,
    initial_epoch=initial_epoch,
    steps_per_epoch=len(train_idx) // 16,
    validation_data=generate_data(16, val_idx),
    validation_steps=len(val_idx) // 16,
    callbacks=[checkpoint_callback, early_stopping, metrics_callback, lr_scheduler],
    verbose=1  # Change to 1 for more detailed output
)

# Load the best model
best_model_path = os.path.join(checkpoint_dir, 'best_model.keras')
best_model = tf.keras.models.load_model(best_model_path, custom_objects={'weighted_binary_crossentropy': weighted_binary_crossentropy})

# Final evaluation on test set using the best model
test_generator = generate_data(16, test_idx)
steps = int(np.ceil(len(test_idx) / 16))  # Ensure all samples are covered
test_results = best_model.evaluate(test_generator, steps=steps)
print("\nFinal Test Results:")
for metric, value in zip(best_model.metrics_names, test_results):
    print(f"{metric}: {value:.4f}")

# Predict on test data using the best model
test_generator = generate_data(16, test_idx)
test_predictions = best_model.predict(test_generator, steps=steps).ravel()  # Ensure all samples are covered and flatten predictions
test_labels = targets_tensor[test_idx]

# Ensure the shapes match
if len(test_labels) != len(test_predictions):
    print(f"Warning: Number of test labels ({len(test_labels)}) does not match number of test predictions ({len(test_predictions)})")
    # Trim the longer one to match the shorter one
    min_len = min(len(test_labels), len(test_predictions))
    test_labels = test_labels[:min_len]
    test_predictions = test_predictions[:min_len]

# Calculate metrics
auc_roc = roc_auc_score(test_labels, test_predictions)
auc_pr = average_precision_score(test_labels, test_predictions)
balanced_acc = balanced_accuracy_score(test_labels, np.round(test_predictions))
mcc = matthews_corrcoef(test_labels, np.round(test_predictions))

# Store metrics
auc_roc_scores.append(auc_roc)
auc_pr_scores.append(auc_pr)
balanced_accuracies.append(balanced_acc)
mcc_scores.append(mcc)

# Save metrics to a file
metrics_file = os.path.join(metrics_dir, 'final_metrics.json')
with open(metrics_file, 'w') as f:
    json.dump({
        'auc_roc': float(auc_roc),
        'auc_pr': float(auc_pr),
        'balanced_accuracy': float(balanced_acc),
        'mcc': float(mcc)
    }, f)

# Save true labels and predicted labels
labels_file = os.path.join(metrics_dir, 'final_labels.json')
with open(labels_file, 'w') as f:
    json.dump({
        'true_labels': test_labels.tolist(),
        'predicted_labels': test_predictions.tolist()
    }, f)

print("\nFinal Metrics:")
print(f"AUC-ROC: {auc_roc:.4f}, AUC-PR: {auc_pr:.4f}, Balanced Accuracy: {balanced_acc:.4f}, MCC: {mcc:.4f}")

# Save the current state
current_epoch += 1
with open(state_file, 'w') as f:
    json.dump({
        'current_epoch': current_epoch,
        'auc_roc_scores': auc_roc_scores,
        'auc_pr_scores': auc_pr_scores,
        'balanced_accuracies': balanced_accuracies,
        'mcc_scores': mcc_scores
    }, f)

