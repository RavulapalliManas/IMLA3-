# Models Package - Vision-Language Transformer and supporting modules
"""
Models subpackage for IML a3

Contains:
- model_transformer.py: OptimizedVLT - Vision-Language Transformer model
- cnn_lstm_model.py: CNN-LSTM baseline model
- dataloader.py: ArtEmisDataset - PyTorch dataset for ArtEmis data
- collate.py: Collate functions for batching with object tags
- train_analyze.py: Training loop with analysis
- evaluate_metrics.py: BLEU/ROUGE metric evaluation
- inference_advanced.py: Advanced inference with beam search
- visualize_attention.py: Attention visualization utilities
- vlm_model_manager.py: Model checkpoint management
- generate_report.py: Report generation utilities
- paths.py: Centralized path configuration
"""

import os

# Get the models directory path
MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(MODELS_DIR)

# Import main classes for easy access
try:
    from .model_transformer import OptimizedVLT, PositionalEncoding
    from .dataloader import ArtEmisDataset
    from .collate import artemis_collate_fn, load_object_tags_cache, set_vocab_cache, get_object_ids_for_image
    from .vlm_model_manager import VLMModelManager
except ImportError:
    # When running scripts directly from models folder
    pass

__all__ = [
    'OptimizedVLT',
    'PositionalEncoding', 
    'ArtEmisDataset',
    'artemis_collate_fn',
    'load_object_tags_cache',
    'set_vocab_cache',
    'get_object_ids_for_image',
    'VLMModelManager',
    'MODELS_DIR',
    'PROJECT_ROOT',
]
