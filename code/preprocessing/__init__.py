# Preprocessing subpackage
"""
Preprocessing module for IML a3.

Contains:
- extract.py: Image preprocessing and extraction utilities
- vocab.py: Vocabulary building from itos.json
- preprocessing.ipynb: Jupyter notebook for data preprocessing
"""

import os

# Define the preprocessing directory path
PREPROCESSING_DIR = os.path.dirname(os.path.abspath(__file__))
PKL_FILES_DIR = os.path.join(os.path.dirname(PREPROCESSING_DIR), 'Pkl Files')

# Common file paths
def get_vocab_path():
    """Get the path to vocab.pkl file."""
    paths = [
        os.path.join(PKL_FILES_DIR, 'vocab.pkl'),
        os.path.join(PREPROCESSING_DIR, 'vocab.pkl'),
    ]
    for path in paths:
        if os.path.exists(path):
            return path
    return paths[0]

def get_dataset_path():
    """Get the path to preprocessed dataset pickle."""
    paths = [
        os.path.join(PKL_FILES_DIR, 'preprocessed_dataset_with_tokens.pkl'),
        os.path.join(PREPROCESSING_DIR, 'preprocessed_dataset_with_tokens.pkl'),
    ]
    for path in paths:
        if os.path.exists(path):
            return path
    return paths[0]

def get_object_tags_path():
    """Get the path to object_tags_precomputed.json."""
    project_root = os.path.dirname(os.path.dirname(PREPROCESSING_DIR))
    paths = [
        os.path.join(PKL_FILES_DIR, 'object_tags_precomputed.json'),
        os.path.join(PREPROCESSING_DIR, 'object_tags_precomputed.json'),
        os.path.join(project_root, 'data', 'object_tags_precomputed.json'),
    ]
    for path in paths:
        if os.path.exists(path):
            return path
    return paths[0]
