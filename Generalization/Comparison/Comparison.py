import pandas as pd

# Load the refined data
refined_compounds = pd.read_csv("data/refined_data/refined_anticancerNP.csv")['SMILES'].tolist()
refined_labels = pd.read_csv("data/refined_data/refined_anticancerNP.csv")['class'].tolist()
# Shuffle data with a fixed random state for reproducibility
refined_compounds, refined_labels = shuffle(refined_compounds, refined_labels, random_state=42)
test_size = 100
while True:
    test_idx = np.random.choice(len(refined_compounds), test_size, replace=False)
    test_labels = [refined_labels[i] for i in test_idx]
    test_active_count = sum(test_labels)
    test_inactive_count = test_size - test_active_count
    if test_active_count == 36 and test_inactive_count == 64:
        break
    train_val_idx = np.setdiff1d(np.arange(len(refined_compounds)), test_idx)
train_val_labels = [refined_labels[i] for i in train_val_idx]
# Separate into anticarcinogens and non-anticarcinogens
anticarcinogens = pd.DataFrame({'SMILES': refined_compounds, 'class': refined_labels})[pd.DataFrame({'SMILES': refined_compounds, 'class': refined_labels})['class'] == 1]
non_anticarcinogens = pd.DataFrame({'SMILES': refined_compounds, 'class': refined_labels})[pd.DataFrame({'SMILES': refined_compounds, 'class': refined_labels})['class'] == 0]
# Select validation samples
anticarcinogens_val = anticarcinogens.sample(n=42, random_state=42)
non_anticarcinogens_val = non_anticarcinogens.sample(n=58, random_state=42)
# Create validation and training sets
validation_set = pd.concat([anticarcinogens_val, non_anticarcinogens_val])
validation_set = validation_set.sample(frac=1, random_state=42).reset_index(drop=True)
anticarcinogens_train = anticarcinogens.drop(anticarcinogens_val.index)
non_anticarcinogens_train = non_anticarcinogens.drop(non_anticarcinogens_val.index)
training_set = pd.concat([anticarcinogens_train, non_anticarcinogens_train])
training_set = training_set.sample(frac=1, random_state=42).reset_index(drop=True)
# Convert to lists
train_idx = training_set.index.tolist()
val_idx = validation_set.index.tolist()
train_smiles = [refined_compounds[i] for i in train_idx]
train_labels = [refined_labels[i] for i in train_idx]
train_df = pd.DataFrame({'Smiles': train_smiles, 'label': train_labels})
val_smiles = [refined_compounds[i] for i in val_idx]
val_labels = [refined_labels[i] for i in val_idx]
val_df = pd.DataFrame({'Smiles': val_smiles, 'label': val_labels})
test_smiles = [refined_compounds[i] for i in test_idx]
test_labels = [refined_labels[i] for i in test_idx]
test_df = pd.DataFrame({'Smiles': test_smiles, 'label': test_labels})
def extract_Mol2Vec(compounds, getFailedSMILES=False, exportCSV=False, outputPath=None, tag=None):
    import logging
    from gensim.models import Word2Vec
    from rdkit import Chem
    from mol2vec.features import mol2alt_sentence, MolSentence
    
    logging.basicConfig(level=logging.INFO)
    model_path = 'feature_extraction/featurizer/Mol2vec/mol2vec/models/model_300dim.pkl'
    logging.info(f"Loading Mol2Vec model from: {model_path}")
    model = Word2Vec.load(model_path)
    df = pd.DataFrame({'SMILES': compounds})
    df['ROMol'] = df['SMILES'].apply(lambda x: Chem.MolFromSmiles(x))
    df['sentence'] = df['ROMol'].apply(lambda x: MolSentence(mol2alt_sentence(x, 1)) if x is not None else [])
    mol2vec_array = np.zeros((len(df), model.vector_size))
    failed_smiles = []
    for i, sentence in enumerate(df['sentence']):
        if sentence:
            try:
                vec = np.mean(model.wv[sentence], axis=0)
                mol2vec_array[i, :] = vec
            except KeyError:
                mol2vec_array[i, :] = np.zeros(model.vector_size)
                failed_smiles.append(df.loc[i, 'SMILES'])
        else:
            mol2vec_array[i, :] = np.zeros(model.vector_size)
            failed_smiles.append(df.loc[i, 'SMILES'])
    if getFailedSMILES:
        failed_smiles = df[df['ROMol'].isnull()]['SMILES']
        if exportCSV and outputPath:
            failed_smiles.to_csv(os.path.join(outputPath, f'failed_smiles_{tag}.csv'), index=False)
        return mol2vec_array, failed_smiles
    return mol2vec_array
print("train_idx length:", len(train_idx))
print("val_idx length:", len(val_idx))
print("test_idx length:", len(test_idx))
print("train_df shape:", train_df.shape)
print("val_df shape:", val_df.shape)
print("test_df shape:", test_df.shape)
import os
#from featurizer.Mol2vec.getMol2vec import extract_Mol2Vec
from featurizer.rdkitMD.getRdkitMD import extract_rdkitMD
from featurizer.MorganFP.getMorganFP import extract_MorganFP
import numpy as np
# Set output directory
PATH = "featurised2_data"
if not os.path.exists(PATH):
    os.makedirs(PATH)
def extract_and_save_features(data, dataset_name, path):
    rdkit_md = extract_rdkitMD(data['Smiles'])
    rdkit_md = rdkit_md.iloc[:, 1:].values  # Exclude SMILES column
    mf1024 = extract_MorganFP(data['Smiles'], bit_type=1024)
    mf1024 = mf1024.iloc[:, 1:].values  # Exclude SMILES column
    mf2048 = extract_MorganFP(data['Smiles'], bit_type=2048)
    mf2048 = mf2048.iloc[:, 1:].values  # Exclude SMILES column
    mol2vec = extract_Mol2Vec(data['Smiles'])
    print("mol2vec shape before saving:", mol2vec.shape)
    np.save(os.path.join(path, f'rdkit_md_{dataset_name}.npy'), rdkit_md)
    np.save(os.path.join(path, f'fp1024_{dataset_name}.npy'), mf1024)
    np.save(os.path.join(path, f'fp2048_{dataset_name}.npy'), mf2048)
    np.save(os.path.join(path, f'mol2vec_{dataset_name}.npy'), mol2vec)
    np.save(os.path.join(path, f'label_{dataset_name}.npy'), data['label'].values)
extract_and_save_features(train_df, 'train', PATH)
extract_and_save_features(val_df, 'val', PATH)
extract_and_save_features(test_df, 'test', PATH)import os
PATH = "featurised_data"
extract_and_save_features(test_df, 'test', PATH)
import random
from pyswarm import pso
from utils import *
from imblearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
#============================================================
# Load data
# Define the path to the feature files
# Load features and labels
train_rdkit_md = np.load(os.path.join(PATH, 'rdkit_md_train.npy'), allow_pickle=True)
train_fp1024 = np.load(os.path.join(PATH, 'fp1024_train.npy'), allow_pickle=True)
train_fp2048 = np.load(os.path.join(PATH, 'fp2048_train.npy'), allow_pickle=True)
train_mol2vec = np.load(os.path.join(PATH, 'mol2vec_train.npy'), allow_pickle=True)
train_label = np.load(os.path.join(PATH, 'label_train.npy'), allow_pickle=True)
val_rdkit_md = np.load(os.path.join(PATH, 'rdkit_md_val.npy'), allow_pickle=True)
val_fp1024 = np.load(os.path.join(PATH, 'fp1024_val.npy'), allow_pickle=True)
val_fp2048 = np.load(os.path.join(PATH, 'fp2048_val.npy'), allow_pickle=True)
val_mol2vec = np.load(os.path.join(PATH, 'mol2vec_val.npy'), allow_pickle=True)
val_label = np.load(os.path.join(PATH, 'label_val.npy'), allow_pickle=True)
test_rdkit_md = np.load(os.path.join(PATH, 'rdkit_md_test.npy'), allow_pickle=True)
test_fp1024 = np.load(os.path.join(PATH, 'fp1024_test.npy'), allow_pickle=True)
test_fp2048 = np.load(os.path.join(PATH, 'fp2048_test.npy'), allow_pickle=True)
test_mol2vec = np.load(os.path.join(PATH, 'mol2vec_test.npy'), allow_pickle=True)
test_label = np.load(os.path.join(PATH, 'label_test.npy'), allow_pickle=True)
# Ensure the data is 2D
train_mol2vec = np.atleast_2d(train_mol2vec)
val_mol2vec = np.atleast_2d(val_mol2vec)
test_mol2vec = np.atleast_2d(test_mol2vec)
# Check shapes and contents
print("train_mol2vec shape:", train_mol2vec.shape)
print("val_mol2vec shape:", val_mol2vec.shape)
print("test_mol2vec shape:", test_mol2vec.shape)
# Ensure data is numeric
try:
    train_mol2vec = train_mol2vec.astype(float)
except ValueError as e:
    print("Error converting train_mol2vec to float:", e)
    train_mol2vec = np.array([]).reshape(0, 1)
    val_mol2vec = val_mol2vec.astype(float)
    print("Error converting val_mol2vec to float:", e)
    val_mol2vec = np.array([]).reshape(0, 1)
    test_mol2vec = test_mol2vec.astype(float)
    print("Error converting test_mol2vec to float:", e)
    test_mol2vec = np.array([]).reshape(0, 1)
# Handle NaN values by removing them
if np.any(np.isnan(train_mol2vec)):
    print("Removing NaNs from train_mol2vec")
    train_mol2vec = train_mol2vec[~np.isnan(train_mol2vec).any(axis=1)]
    train_label = train_label[~np.isnan(train_mol2vec).any(axis=0)]
if np.any(np.isnan(val_mol2vec)):
    print("Removing NaNs from val_mol2vec")
    val_mol2vec = val_mol2vec[~np.isnan(val_mol2vec).any(axis=1)]
    val_label = val_label[~np.isnan(val_mol2vec).any(axis=0)]
if np.any(np.isnan(test_mol2vec)):
    print("Removing NaNs from test_mol2vec")
    test_mol2vec = test_mol2vec[~np.isnan(test_mol2vec).any(axis=1)]
    test_label = test_label[~np.isnan(test_mol2vec).any(axis=0)]
# Check if train_mol2vec has only one sample or is empty after removing NaNs
if train_mol2vec.size == 0 or train_mol2vec.shape[0] <= 1:
    print("train_mol2vec has only one sample or is empty after removing NaNs.")
    # Optionally, handle this case by skipping VarianceThreshold or loading more data
    # For now, we'll skip VarianceThreshold
    my_xgb_mol2vec = make_pipeline(XGBClassifier(random_state=42, 
                                                 n_estimators=100,
                                                 max_depth=9,
                                                 colsample_bytree=0.2,
                                                 learning_rate=0.01))
else:
    my_xgb_mol2vec = make_pipeline(VarianceThreshold(),
                                   XGBClassifier(random_state=42, 
                                                 learning_rate=0.01))  
    #============================================================
# Define base classifiers
my_svm_mf1024 = make_pipeline(VarianceThreshold(),
                              SVC(C=0.001, gamma=0.1, probability=True))
my_svm_mf2048 = make_pipeline(VarianceThreshold(),
my_rf_md = make_pipeline(MinMaxScaler(),
                         VarianceThreshold(),
                         RandomForestClassifier(random_state=42,
                                                n_estimators=100,
                                                max_depth=9,
                                                max_features=0.4,
                                                min_samples_split=3))
# Validation
# Assuming my_svm_mf1024, my_svm_mf2048, my_rf_md, my_xgb_mol2vec are already defined
# and train_fp1024, train_fp2048, train_rdkit_md, train_mol2vec, val_fp1024, val_fp2048,
# val_rdkit_md, val_mol2vec, train_label, val_label are properly defined and loaded.
val_pred_svm_mf1024  = my_svm_mf1024.fit(train_fp1024, train_label).predict_proba(val_fp1024)[:, 1]
val_pred_svm_mf2048  = my_svm_mf2048.fit(train_fp2048, train_label).predict_proba(val_fp2048)[:, 1]
val_pred_rf_md       = my_rf_md.fit(train_rdkit_md, train_label).predict_proba(val_rdkit_md)[:, 1]  # Changed from train_rdkit_md to val_rdkit_md if needed
# Check if train_mol2vec has only one sample or is empty
    print("train_mol2vec has only one sample or is empty during validation. Skipping VarianceThreshold.")
    if train_mol2vec.shape[0] == 0:
        print("No valid data in train_mol2vec. Skipping XGBClassifier training.")
        val_pred_xgb_mol2vec = np.zeros(val_mol2vec.shape[0])
    else:
        val_pred_xgb_mol2vec = my_xgb_mol2vec.fit(train_mol2vec, train_label).predict_proba(val_mol2vec)[:, 1]
    val_pred_xgb_mol2vec = my_xgb_mol2vec.fit(train_mol2vec, train_label).predict_proba(val_mol2vec)[:, 1]
# Optimization
# Ensure that all prediction arrays have the same shape
assert val_pred_svm_mf1024.shape == val_pred_svm_mf2048.shape == val_pred_rf_md.shape == val_pred_xgb_mol2vec.shape
def get_optimisation_function(val_ensemble_pred, val_label):
    preds_1, preds_2, preds_3, preds_4 = val_ensemble_pred
    labels = val_label
    def aucroc_optimisation(weight):
        # Ensure weight is a 1D array
        weight = np.asarray(weight).flatten()
        w1, w2, w3, w4 = weight
        # Normalize the weights
        total = w1 + w2 + w3 + w4
        if total == 0:
            total = 1e-8  # Avoid division by zero
        w1_norm = w1 / total
        w2_norm = w2 / total
        w3_norm = w3 / total
        w4_norm = w4 / total
        # Calculate the ensemble prediction
        preds_ensemble = (preds_1 * w1_norm +
                          preds_2 * w2_norm +
                          preds_3 * w3_norm +
                          preds_4 * w4_norm)
        # Calculate the ROC-AUC score
        roc_auc = roc_auc_score(labels, preds_ensemble)
        # Since PSO minimizes the objective, return the negative AUC
        return -roc_auc
    return aucroc_optimisation
# Define bounds for the weights
lb = [0.1, 0.1, 0.1, 0.1]
ub = [0.5, 0.5, 0.5, 0.5]
# Prepare the list of validation predictions
val_ensemble_pred = [val_pred_svm_mf1024, val_pred_svm_mf2048, val_pred_rf_md, val_pred_xgb_mol2vec]
# Get the optimization function
auc_optimisation = get_optimisation_function(val_ensemble_pred, val_label)
# Seed range for reproducibility
seed_range = np.arange(0, 10)
weight_list = []
for s in seed_range:
    np.random.seed(s)
    # Ensure compatibility with Python 3.9+ for random.seed
    # random.seed(s) is deprecated for hashing-based seeding
    # Use a fixed seed or remove if not necessary
    # random.seed(s)  # Uncomment if necessary, but be aware of the deprecation
    # Perform PSO optimization
    optx, fopt = pso(auc_optimisation, lb, ub, swarmsize=100, seed=s, maxiter=50)
    weight_list.append(optx)  # Assuming extract_weight is not necessary and optx is the weight vector
    print("Round {}: Completed".format(s+1))
# Collect individual weights from the weight list
w1_list, w2_list, w3_list, w4_list = [], [], [], []
for weight in weight_list:
    w1_list.append(weight[0])
    w2_list.append(weight[1])
    w3_list.append(weight[2])
    w4_list.append(weight[3])
# Calculate the average weights
w1_average = np.mean(w1_list)
w2_average = np.mean(w2_list)
w3_average = np.mean(w3_list)
w4_average = np.mean(w4_list)
# Normalize the average weights
total_average = w1_average + w2_average + w3_average + w4_average
w1_avage_norm = np.round(w1_average / total_average, 4)
w2_avage_norm = np.round(w2_average / total_average, 4)
w3_avage_norm = np.round(w3_average / total_average, 4)
w4_avage_norm = np.round(w4_average / total_average, 4)
# Testing 
test_pred_svm_mf1024   = my_svm_mf1024.fit(np.concatenate([train_fp1024, val_fp1024]), np.concatenate([train_label, val_label])).predict_proba(test_fp1024)[:, 1]
test_pred_svm_mf2048   = my_svm_mf2048.fit(np.concatenate([train_fp2048, val_fp2048]), np.concatenate([train_label, val_label])).predict_proba(test_fp2048)[:, 1]
test_pred_rf_md        = my_rf_md.fit(np.concatenate([train_rdkit_md, val_rdkit_md]), np.concatenate([train_label, val_label])).predict_proba(test_rdkit_md)[:, 1]
    print("train_mol2vec has only one sample or is empty during testing. Skipping VarianceThreshold.")
        test_pred_xgb_mol2vec = np.zeros(test_mol2vec.shape[0])
        test_pred_xgb_mol2vec = my_xgb_mol2vec.fit(np.concatenate([train_mol2vec, val_mol2vec]), np.concatenate([train_label, val_label])).predict_proba(test_mol2vec)[:, 1]
    test_pred_xgb_mol2vec = my_xgb_mol2vec.fit(np.concatenate([train_mol2vec, val_mol2vec]), np.concatenate([train_label, val_label])).predict_proba(test_mol2vec)[:, 1]
# Ensure all predictions have the same length
test_pred_list = [test_pred_svm_mf1024, test_pred_svm_mf2048, test_pred_rf_md, test_pred_xgb_mol2vec]
min_length = min(len(pred) for pred in test_pred_list)
# Truncate predictions to the shortest length
test_pred_svm_mf1024 = test_pred_svm_mf1024[:min_length]
test_pred_svm_mf2048 = test_pred_svm_mf2048[:min_length]
test_pred_rf_md = test_pred_rf_md[:min_length]
test_pred_xgb_mol2vec = test_pred_xgb_mol2vec[:min_length]
test_label = test_label[:min_length]
test_pred_top4ensemble = (test_pred_svm_mf1024*w1_avage_norm + 
                          test_pred_svm_mf2048*w2_avage_norm + 
                          test_pred_rf_md*w3_avage_norm + 
                          test_pred_xgb_mol2vec*w4_avage_norm)
test_roc_auc_svm_mf1024   = roc_auc_score(test_label, test_pred_svm_mf1024)
test_roc_auc_svm_mf2048   = roc_auc_score(test_label, test_pred_svm_mf2048)
test_roc_auc_rf_md        = roc_auc_score(test_label, test_pred_rf_md)
test_roc_auc_xgb_mol2vec  = roc_auc_score(test_label, test_pred_xgb_mol2vec)
test_roc_auc_top4ensemble = roc_auc_score(test_label, test_pred_top4ensemble)
 
#===========================================================
# Export results
test_roc_list = [test_roc_auc_svm_mf1024, test_roc_auc_svm_mf2048, test_roc_auc_rf_md, test_roc_auc_xgb_mol2vec, test_roc_auc_top4ensemble]
weight_list   = [w1_avage_norm, w2_avage_norm, w3_avage_norm, w4_avage_norm, 1]
fea_list = ['svm_mf1024', 'svm_mf2048', 'rf_md2048', 'xgb_mol2vec', 'top4ensemble']
weight_path = "results/pso/singlerun/weights"
if not os.path.isdir(weight_path):
    os.makedirs(weight_path)
pred_path = "results/pso/singlerun/pred/top4"
if not os.path.isdir(pred_path):
    os.makedirs(pred_path)
         
pd.DataFrame(zip(test_roc_list, weight_list), index=fea_list, columns=['ROC-AUC', 'Weight']).to_csv(f"{weight_path}/roc_auc_test_top4ensemble_seed0.csv",  index=None)
pd.DataFrame(zip(test_pred_svm_mf1024, test_label),   columns=["predicted_prob", "true_class"]).to_csv(f"{pred_path}/y_prob_test_SVM_mf1024_seed0.csv",   index=None)
pd.DataFrame(zip(test_pred_svm_mf2048, test_label),   columns=["predicted_prob", "true_class"]).to_csv(f"{pred_path}/y_prob_test_SVM_mf2048_seed0.csv",   index=None)
pd.DataFrame(zip(test_pred_rf_md, test_label),        columns=["predicted_prob", "true_class"]).to_csv(f"{pred_path}/y_prob_test_RF_md_seed0.csv",        index=None)
pd.DataFrame(zip(test_pred_xgb_mol2vec, test_label),  columns=["predicted_prob", "true_class"]).to_csv(f"{pred_path}/y_prob_test_XGB_mol2vec_seed0.csv",  index=None)
pd.DataFrame(zip(test_pred_top4ensemble, test_label), columns=["predicted_prob", "true_class"]).to_csv(f"{pred_path}/y_prob_test_top4ensemble_seed0.csv", index=None)
# Load the trained models
from sklearn.pipeline import make_pipeline
import logging
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, roc_curve, auc, precision_recall_curve
# Set up logging
logging.basicConfig(level=logging.INFO)
# Define the models
my_svm_mf1024 = make_pipeline(VarianceThreshold(), SVC(C=0.001, gamma=0.1, probability=True))
my_svm_mf2048 = make_pipeline(VarianceThreshold(), SVC(C=0.001, gamma=0.1, probability=True))
my_rf_md = make_pipeline(MinMaxScaler(), VarianceThreshold(), RandomForestClassifier(random_state=42, n_estimators=100, max_depth=9, max_features=0.4, min_samples_split=3))
my_xgb_mol2vec = make_pipeline(VarianceThreshold(), XGBClassifier(random_state=42, n_estimators=100, max_depth=9, colsample_bytree=0.2, learning_rate=0.01))
# Load the training and validation data
train_fp1024 = np.load("featurised2_data/fp1024_train.npy", allow_pickle=True)
val_fp1024 = np.load("featurised2_data/fp1024_val.npy", allow_pickle=True)
train_fp2048 = np.load("featurised2_data/fp2048_train.npy", allow_pickle=True)
val_fp2048 = np.load("featurised2_data/fp2048_val.npy", allow_pickle=True)
train_rdkit_md = np.load("featurised2_data/rdkit_md_train.npy", allow_pickle=True)
val_rdkit_md = np.load("featurised2_data/rdkit_md_val.npy", allow_pickle=True)
train_mol2vec = np.load("featurised2_data/mol2vec_train.npy", allow_pickle=True)
val_mol2vec = np.load("featurised2_data/mol2vec_val.npy", allow_pickle=True)
train_label = np.load("featurised2_data/label_train.npy", allow_pickle=True)
val_label = np.load("featurised2_data/label_val.npy", allow_pickle=True)
# Fit the models on the combined training and validation data
train_val_fp1024 = np.concatenate([train_fp1024, val_fp1024])
train_val_fp2048 = np.concatenate([train_fp2048, val_fp2048])
train_val_rdkit_md = np.concatenate([train_rdkit_md, val_rdkit_md])
train_val_mol2vec = np.concatenate([train_mol2vec, val_mol2vec])
train_val_label = np.concatenate([train_label, val_label])
logging.info("Fitting SVM MF1024 model...")
my_svm_mf1024.fit(train_val_fp1024, train_val_label)
logging.info("Fitting SVM MF2048 model...")
my_svm_mf2048.fit(train_val_fp2048, train_val_label)
logging.info("Fitting RF MD model...")
my_rf_md.fit(train_val_rdkit_md, train_val_label)
logging.info("Fitting XGB Mol2Vec model...")
my_xgb_mol2vec.fit(train_val_mol2vec, train_val_label)
# After extracting rdkit_md
new_rdkit_md = extract_rdkitMD(new_smiles)
# Check the type of new_rdkit_md
print("Type of new_rdkit_md after extraction:", type(new_rdkit_md))
# Slice the data appropriately
if isinstance(new_rdkit_md, pd.DataFrame):
    # If it's a DataFrame, exclude the SMILES column
    new_rdkit_md = new_rdkit_md.iloc[:, 1:].values
elif isinstance(new_rdkit_md, np.ndarray):
    # If it's a NumPy array, exclude the first column
    new_rdkit_md = new_rdkit_md[:, 1:]
    raise TypeError("Unexpected type for new_rdkit_md")
# Ensure it's a float NumPy array
new_rdkit_md = new_rdkit_md.astype(float)
# Verify the shape and data
print("Shape of new_rdkit_md:", new_rdkit_md.shape)
print("Sample data:", new_rdkit_md[:5])
# Save the features
np.save(os.path.join(feature_path, 'new_rdkit_md.npy'), new_rdkit_md)
# Check for NaN values
if np.any(np.isnan(new_rdkit_md)):
    logging.error("new_rdkit_md still contains NaN values after imputation.")
    logging.info("new_rdkit_md is free of NaN values after imputation.")
# Handle NaN values in new_rdkit_md by imputing them
imputer = SimpleImputer(strategy='mean')
new_rdkit_md = imputer.fit_transform(new_rdkit_md)
# Check if new_rdkit_md still contains NaN values
    import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
# Define the path to load the extracted features
feature_path = "extracted_features_e"
# Load the new data
new_data = pd.read_csv("data/Data1_R_final_cleaned (1).csv")
new_labels = new_data['label'].tolist()
# Function to load numpy files with error handling
def load_numpy_file(filepath, allow_pickle=True):
    try:
        data = np.load(filepath, allow_pickle=allow_pickle)
        if data.dtype == 'object':
            logging.warning(f"File {filepath} contains object arrays. Attempting to convert to float64.")
                data = data.astype(np.float64)
            except ValueError as e:
                logging.error(f"Failed to convert {filepath} to float64: {e}")
                data = np.array([]).reshape(0, 1)
        return data
    except Exception as e:
        logging.error(f"Error loading {filepath}: {e}")
        return np.array([]).reshape(0, 1)
# Load the extracted features
new_rdkit_md = np.load( 'extracted_features_e/new_rdkit_md.npy', allow_pickle=True)
new_mf1024 = np.load('extracted_features_e/new_mf1024.npy', allow_pickle=True)
new_mf2048 = np.load('extracted_features_e/new_mf2048.npy', allow_pickle=True)
new_mol2vec = np.load('extracted_features_e/new_mol2vec.npy', allow_pickle=True)
# Ensure the data is 2D and numeric for new_mol2vec
if new_mol2vec is not None and new_mol2vec.size > 0:
    new_mol2vec = np.atleast_2d(new_mol2vec)
    if new_mol2vec.dtype != np.float64:
        new_mol2vec = new_mol2vec.astype(np.float64)
    new_mol2vec = np.array([]).reshape(0, 1)
# Ensure data is numeric and convert to float64 for new_rdkit_md
if new_rdkit_md is not None and new_rdkit_md.size > 0:
    new_rdkit_md = new_rdkit_md.astype(np.float64)
    new_rdkit_md = np.array([]).reshape(0, 1)
if new_rdkit_md.size > 0:
    new_rdkit_md = imputer.fit_transform(new_rdkit_md)
    logging.warning("new_rdkit_md is empty.")
# Check for infinite values after imputation
if np.any(np.isinf(new_rdkit_md)):
    logging.error("new_rdkit_md contains infinite values after imputation.")
    logging.info("new_rdkit_md is free of infinite values after imputation.")
# Define custom transformer to convert data to float64
to_float64 = FunctionTransformer(lambda x: x.astype(np.float64))
# Define the models with imputers and correct data types
my_svm_mf1024 = make_pipeline(
    SimpleImputer(strategy='mean'),
    VarianceThreshold(),
    SVC(C=0.001, gamma=0.1, probability=True)
)
my_svm_mf2048 = make_pipeline(
my_rf_md = make_pipeline(
    MinMaxScaler(),
    to_float64,
    RandomForestClassifier(random_state=42, n_estimators=100, max_depth=9, max_features=0.4, min_samples_split=3)
my_xgb_mol2vec = make_pipeline(
    XGBClassifier(random_state=42, n_estimators=100, max_depth=9, colsample_bytree=0.2, learning_rate=0.01)
# Ensure labels are in integer format
if train_label.size > 0:
    train_label = train_label.astype(int)
    logging.warning("train_label is empty.")
if val_label.size > 0:
    val_label = val_label.astype(int)
    logging.warning("val_label is empty.")
# Combine training and validation data
train_val_fp1024 = np.concatenate([train_fp1024, val_fp1024]) if train_fp1024.size > 0 and val_fp1024.size > 0 else np.array([]).reshape(0, 1)
train_val_fp2048 = np.concatenate([train_fp2048, val_fp2048]) if train_fp2048.size > 0 and val_fp2048.size > 0 else np.array([]).reshape(0, 1)
train_val_rdkit_md = np.concatenate([train_rdkit_md, val_rdkit_md]) if train_rdkit_md.size > 0 and val_rdkit_md.size > 0 else np.array([]).reshape(0, 1)
train_val_mol2vec = np.concatenate([train_mol2vec, val_mol2vec]) if train_mol2vec.size > 0 and val_mol2vec.size > 0 else np.array([]).reshape(0, 1)
train_val_label = np.concatenate([train_label, val_label]) if train_label.size > 0 and val_label.size > 0 else np.array([])
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, matthews_corrcoef
feature_path = ""
new_data = pd.read_csv("data/Data1.csv")
new_rdkit_md = load_numpy_file('new_rdkit_md.npy')
new_mf1024 = load_numpy_file('/new_mf1024.npy')
new_mf2048 = load_numpy_file('/new_mf2048.npy')
new_mol2vec = load_numpy_file('/new_mol2vec.npy')
# Handle infinite values and ensure data type for new_rdkit_md
    new_rdkit_md = np.nan_to_num(new_rdkit_md, posinf=np.nan, neginf=np.nan)
    imputer = SimpleImputer(strategy='mean')
    lower_bound = -1e30
    upper_bound = 1e30
    new_rdkit_md = np.clip(new_rdkit_md, lower_bound, upper_bound)
# Handle infinite values and ensure data type for new_mf1024
if new_mf1024.size > 0:
    new_mf1024 = np.nan_to_num(new_mf1024, posinf=np.nan, neginf=np.nan)
    new_mf1024 = imputer.fit_transform(new_mf1024)
    new_mf1024 = new_mf1024.astype(np.float64)
    logging.warning("new_mf1024 is empty.")
    new_mf1024 = np.array([]).reshape(0, 1)
# Handle infinite values and ensure data type for new_mf2048
if new_mf2048.size > 0:
    new_mf2048 = np.nan_to_num(new_mf2048, posinf=np.nan, neginf=np.nan)
    new_mf2048 = imputer.fit_transform(new_mf2048)
    new_mf2048 = new_mf2048.astype(np.float64)
    logging.warning("new_mf2048 is empty.")
    new_mf2048 = np.array([]).reshape(0, 1)
# Handle infinite values and ensure data type for new_mol2vec
if new_mol2vec.size > 0:
    new_mol2vec = np.nan_to_num(new_mol2vec, posinf=np.nan, neginf=np.nan)
    new_mol2vec = imputer.fit_transform(new_mol2vec)
    new_mol2vec = new_mol2vec.astype(np.float64)
    logging.warning("new_mol2vec is empty.")
train_fp1024 = load_numpy_file("/fp1024_train.npy")
val_fp1024 = load_numpy_file("/fp1024_val.npy")
train_fp2048 = load_numpy_file("/fp2048_train.npy")
val_fp2048 = load_numpy_file("/fp2048_val.npy")
train_rdkit_md = load_numpy_file("/rdkit_md_train.npy")
val_rdkit_md = load_numpy_file("/rdkit_md_val.npy")
train_mol2vec = load_numpy_file("/mol2vec_train.npy")
val_mol2vec = load_numpy_file("/mol2vec_val.npy")
train_label = load_numpy_file("/label_train.npy")
val_label = load_numpy_file("/label_val.npy")
# Fit the models
if train_val_fp1024.size > 0 and train_val_label.size > 0:
    logging.info("Fitting SVM MF1024 model...")
    my_svm_mf1024.fit(train_val_fp1024, train_val_label)
    logging.warning("train_val_fp1024 or train_val_label is empty. Skipping SVM MF1024 model fitting.")
if train_val_fp2048.size > 0 and train_val_label.size > 0:
    logging.info("Fitting SVM MF2048 model...")
    my_svm_mf2048.fit(train_val_fp2048, train_val_label)
    logging.warning("train_val_fp2048 or train_val_label is empty. Skipping SVM MF2048 model fitting.")
if train_val_rdkit_md.size > 0 and train_val_label.size > 0:
    logging.info("Fitting RF MD model...")
    my_rf_md.fit(train_val_rdkit_md, train_val_label)
    logging.warning("train_val_rdkit_md or train_val_label is empty. Skipping RF MD model fitting.")
if train_val_mol2vec.size > 0 and train_val_label.size > 0:
    logging.info("Fitting XGB Mol2Vec model...")
    my_xgb_mol2vec.fit(train_val_mol2vec, train_val_label)
    logging.warning("train_val_mol2vec or train_val_label is empty. Skipping XGB Mol2Vec model fitting.")
# Make predictions on the new data
logging.info("Making predictions on new data...")
new_pred_svm_mf1024 = my_svm_mf1024.predict_proba(new_mf1024)[:, 1] if new_mf1024.size > 0 else np.array([])
new_pred_svm_mf2048 = my_svm_mf2048.predict_proba(new_mf2048)[:, 1] if new_mf2048.size > 0 else np.array([])
new_pred_rf_md = my_rf_md.predict_proba(new_rdkit_md)[:, 1] if new_rdkit_md.size > 0 else np.array([])
new_pred_xgb_mol2vec = my_xgb_mol2vec.predict_proba(new_mol2vec)[:, 1] if new_mol2vec.size > 0 else np.array([])
w1_avage_norm =  # Replace with actual values
w2_avage_norm = 
w3_avage_norm = 
w4_avage_norm = 
new_pred_list = [new_pred_svm_mf1024, new_pred_svm_mf2048, new_pred_rf_md, new_pred_xgb_mol2vec]
min_length = min(len(pred) for pred in new_pred_list if pred.size > 0)
new_pred_svm_mf1024 = new_pred_svm_mf1024[:min_length] if new_pred_svm_mf1024.size > 0 else np.array([])
new_pred_svm_mf2048 = new_pred_svm_mf2048[:min_length] if new_pred_svm_mf2048.size > 0 else np.array([])
new_pred_rf_md = new_pred_rf_md[:min_length] if new_pred_rf_md.size > 0 else np.array([])
new_pred_xgb_mol2vec = new_pred_xgb_mol2vec[:min_length] if new_pred_xgb_mol2vec.size > 0 else np.array([])
new_labels = new_labels[:min_length]
# Ensemble prediction
ensemble_weights = []
ensemble_predictions = []
if new_pred_svm_mf1024.size > 0:
    ensemble_weights.append(w1_avage_norm)
    ensemble_predictions.append(new_pred_svm_mf1024)
if new_pred_svm_mf2048.size > 0:
    ensemble_weights.append(w2_avage_norm)
    ensemble_predictions.append(new_pred_svm_mf2048)
if new_pred_rf_md.size > 0:
    ensemble_weights.append(w3_avage_norm)
    ensemble_predictions.append(new_pred_rf_md)
if new_pred_xgb_mol2vec.size > 0:
    ensemble_weights.append(w4_avage_norm)
    ensemble_predictions.append(new_pred_xgb_mol2vec)
if ensemble_predictions:
    new_pred_top4ensemble = np.average(ensemble_predictions, weights=ensemble_weights, axis=0)
    new_pred_top4ensemble = np.array([])
# Evaluate the predictions
logging.info("Evaluating predictions...")
# Define the threshold for converting probabilities to binary labels
threshold = 0.5
# Function to compute metrics
def compute_metrics(true_labels, pred_probs, threshold):
    if len(pred_probs) == 0:
        return {
            'ROC-AUC': 0.0,
            'AUC-PR': 0.0,
            'Balanced Accuracy': 0.0,
            'MCC': 0.0
        }
    pred_labels = (pred_probs >= threshold).astype(int)
    roc_auc = roc_auc_score(true_labels, pred_probs) if len(pred_probs) > 0 else 0.0
    auc_pr = average_precision_score(true_labels, pred_probs) if len(pred_probs) > 0 else 0.0
    balanced_acc = balanced_accuracy_score(true_labels, pred_labels) if len(pred_labels) > 0 else 0.0
    mcc = matthews_corrcoef(true_labels, pred_labels) if len(pred_labels) > 0 else 0.0
    return {
        'ROC-AUC': roc_auc,
        'AUC-PR': auc_pr,
        'Balanced Accuracy': balanced_acc,
        'MCC': mcc
    }
# Compute metrics for each model
metrics_svm_mf1024 = compute_metrics(new_labels, new_pred_svm_mf1024, threshold) if new_pred_svm_mf1024.size > 0 else {
    'ROC-AUC': 0.0,
    'AUC-PR': 0.0,
    'Balanced Accuracy': 0.0,
    'MCC': 0.0
}
metrics_svm_mf2048 = compute_metrics(new_labels, new_pred_svm_mf2048, threshold) if new_pred_svm_mf2048.size > 0 else {
metrics_rf_md = compute_metrics(new_labels, new_pred_rf_md, threshold) if new_pred_rf_md.size > 0 else {
metrics_xgb_mol2vec = compute_metrics(new_labels, new_pred_xgb_mol2vec, threshold) if new_pred_xgb_mol2vec.size > 0 else {
metrics_top4ensemble = compute_metrics(new_labels, new_pred_top4ensemble, threshold) if new_pred_top4ensemble.size > 0 else {
# Collect all metrics
all_metrics = {
    'svm_mf1024': metrics_svm_mf1024,
    'svm_mf2048': metrics_svm_mf2048,
    'rf_md2048': metrics_rf_md,
    'xgb_mol2vec': metrics_xgb_mol2vec,
    'top4ensemble': metrics_top4ensemble
# Prepare DataFrame for saving
metrics_df = pd.DataFrame({
    'Feature': ['svm_mf1024', 'svm_mf2048', 'rf_md2048', 'xgb_mol2vec', 'top4ensemble'],
    'ROC-AUC': [all_metrics['svm_mf1024']['ROC-AUC'], all_metrics['svm_mf2048']['ROC-AUC'],
                all_metrics['rf_md2048']['ROC-AUC'], all_metrics['xgb_mol2vec']['ROC-AUC'],
                all_metrics['top4ensemble']['ROC-AUC']],
    'AUC-PR': [all_metrics['svm_mf1024']['AUC-PR'], all_metrics['svm_mf2048']['AUC-PR'],
               all_metrics['rf_md2048']['AUC-PR'], all_metrics['xgb_mol2vec']['AUC-PR'],
               all_metrics['top4ensemble']['AUC-PR']],
    'Balanced Accuracy': [all_metrics['svm_mf1024']['Balanced Accuracy'], all_metrics['svm_mf2048']['Balanced Accuracy'],
                         all_metrics['rf_md2048']['Balanced Accuracy'], all_metrics['xgb_mol2vec']['Balanced Accuracy'],
                         all_metrics['top4ensemble']['Balanced Accuracy']],
    'MCC': [all_metrics['svm_mf1024']['MCC'], all_metrics['svm_mf2048']['MCC'],
            all_metrics['rf_md2048']['MCC'], all_metrics['xgb_mol2vec']['MCC'],
            all_metrics['top4ensemble']['MCC']],
    'Weight': [w1_avage_norm, w2_avage_norm, w3_avage_norm, w4_avage_norm, 1]
})
# Save the evaluation results
output_path = "results/pso/singlerun/pred/top4/"
if not os.path.isdir(output_path):
    os.makedirs(output_path)
metrics_df.to_csv(f"{output_path}/metrics_new_top4ensemble_seed0.csv", index=False)