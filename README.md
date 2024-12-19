# DFT_ANPD :A Dual-Feature Two-sided Attention Network for Anticancer Natural Products Detection 

![rambod's First Illustration (1)](https://github.com/user-attachments/assets/6845efb2-a1bf-45f2-a984-8e18e1ab7372)






## Objectives

Cancer remains the leading cause of death globally despite progress in research. Natural products have historically played a key role in anticancer drug discovery. Experimental and computational methods are used to identify anticancer compounds. This study introduces **DFT_ANPD**, a deep learning framework for predicting anticancer properties in natural compounds. It integrates molecular structural data with embeddings from large language models (LLMs). A 1D-CNN processes SMILES strings for chemical patterns, while a fine-tuned SMILES-BERT extracts semantic information. These insights are fused using a two-sided attention mechanism to prioritize relevant features, enabling prediction via a sigmoid activation function. 

## Findings

DFT_ANPD outperformed existing methods, like iANP-EC, on NPACT, CancerHSP, and NPASS datasets, showing superior accuracy and generalization. Actinomycin D and Clavariopsin C were identified, proving the model's robustness and clinical relevance.

## Installation Guide

### 1. Prerequisites
Ensure you have the following installed on your system:
* Python 3.7 or newer
* pandas
* numpy
* rdkit
* scikit-learn
* tensorflow
* keras
* torch
* matplotlib
* seaborn 
* pyswarm

### 2. Clone the Repository

```bash
git clone https://github.com/Rambono/DFT_ANPD.git

cd DFT_ANPD
```
## Componnents
* Generalization_data preprocess.py - 
