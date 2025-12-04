# Paths configuration for IML a3 project
"""
Central path configuration for the IML a3 project.
All file paths should be defined here for easy management.
"""

import os

# Get the project root directory (IML a3 folder)
# This works regardless of which script is executed
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Alternatively, look for the IML a3 folder specifically
def find_project_root():
    """Find the IML a3 project root directory."""
    current = os.path.dirname(os.path.abspath(__file__))
    while current != os.path.dirname(current):  # Not at filesystem root
        if os.path.basename(current) == 'IML a3':
            return current
        if os.path.exists(os.path.join(current, 'IML a3')):
            return os.path.join(current, 'IML a3')
        current = os.path.dirname(current)
    # Fallback to hardcoded path
    return r'E:\A3\IML a3'

PROJECT_ROOT = find_project_root()

# Main directories
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
CODE_DIR = os.path.join(PROJECT_ROOT, 'code')
PREPROCESSING_DIR = os.path.join(CODE_DIR, 'preprocessing')
NOTEBOOKS_DIR = os.path.join(PROJECT_ROOT, 'notebooks')
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, 'outputs')
CONFIGS_DIR = os.path.join(PROJECT_ROOT, 'configs')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')
REPORT_DIR = os.path.join(PROJECT_ROOT, 'report')

# Data files
IMAGE_FOLDER = os.path.join(DATA_DIR, 'final_data')
IMAGE_FOLDER_RESIZED = os.path.join(DATA_DIR, 'final_data_resized')
OBJECT_TAGS_PATH = os.path.join(DATA_DIR, 'object_tags_precomputed.json')
DATA_CSV_PATH = os.path.join(DATA_DIR, 'data.csv')

# Preprocessed files (in code/preprocessing or code/Pkl Files)
PKL_FILES_DIR = os.path.join(CODE_DIR, 'Pkl Files')
VOCAB_PATH = os.path.join(PKL_FILES_DIR, 'vocab.pkl') if os.path.exists(os.path.join(PKL_FILES_DIR, 'vocab.pkl')) else os.path.join(PREPROCESSING_DIR, 'vocab.pkl')
DATASET_PKL_PATH = os.path.join(PKL_FILES_DIR, 'preprocessed_dataset_with_tokens.pkl') if os.path.exists(os.path.join(PKL_FILES_DIR, 'preprocessed_dataset_with_tokens.pkl')) else os.path.join(PREPROCESSING_DIR, 'preprocessed_dataset_with_tokens.pkl')

# Also check for object tags in preprocessing folder
if not os.path.exists(OBJECT_TAGS_PATH):
    alt_path = os.path.join(PREPROCESSING_DIR, 'object_tags_precomputed.json')
    if os.path.exists(alt_path):
        OBJECT_TAGS_PATH = alt_path
    else:
        alt_path2 = os.path.join(PKL_FILES_DIR, 'object_tags_precomputed.json')
        if os.path.exists(alt_path2):
            OBJECT_TAGS_PATH = alt_path2

# Results directories
RESULTS_DIR = os.path.join(PREPROCESSING_DIR, 'results_automated')
RESULTS_OPTIMIZED_VLM_DIR = os.path.join(PROJECT_ROOT, '..', 'results_optimized_vlm')

# Model checkpoints - try to find the best model
def find_best_model_checkpoint():
    """Find the best model checkpoint file."""
    search_paths = [
        os.path.join(RESULTS_DIR, 'best_model.pt'),
        os.path.join(OUTPUTS_DIR, 'best_model.pt'),
        os.path.join(MODELS_DIR, 'best_model.pt'),
    ]
    
    # Also look in subdirectories of results_automated
    if os.path.exists(RESULTS_DIR):
        for folder in os.listdir(RESULTS_DIR):
            folder_path = os.path.join(RESULTS_DIR, folder)
            if os.path.isdir(folder_path):
                model_path = os.path.join(folder_path, 'model.pt')
                if os.path.exists(model_path):
                    search_paths.append(model_path)
    
    for path in search_paths:
        if os.path.exists(path):
            return path
    
    return search_paths[0]  # Return default even if not exists

BEST_MODEL_PATH = find_best_model_checkpoint()


def print_paths():
    """Print all configured paths for debugging."""
    print("="*70)
    print("IML a3 Project Paths Configuration")
    print("="*70)
    print(f"PROJECT_ROOT:      {PROJECT_ROOT}")
    print(f"DATA_DIR:          {DATA_DIR}")
    print(f"MODELS_DIR:        {MODELS_DIR}")
    print(f"CODE_DIR:          {CODE_DIR}")
    print(f"PREPROCESSING_DIR: {PREPROCESSING_DIR}")
    print(f"IMAGE_FOLDER:      {IMAGE_FOLDER}")
    print(f"VOCAB_PATH:        {VOCAB_PATH}")
    print(f"DATASET_PKL_PATH:  {DATASET_PKL_PATH}")
    print(f"OBJECT_TAGS_PATH:  {OBJECT_TAGS_PATH}")
    print(f"RESULTS_DIR:       {RESULTS_DIR}")
    print(f"BEST_MODEL_PATH:   {BEST_MODEL_PATH}")
    print("="*70)


if __name__ == '__main__':
    print_paths()
