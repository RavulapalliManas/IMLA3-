import torch
from torch.nn.utils.rnn import pad_sequence
import json
import os

# Global object tags cache (loaded once)
_object_tags_cache = None
_vocab_stoi_cache = None

def load_object_tags_cache(json_path):
    """Load object tags from JSON file into global cache."""
    global _object_tags_cache
    if _object_tags_cache is None and os.path.exists(json_path):
        with open(json_path, 'r') as f:
            _object_tags_cache = json.load(f)
    return _object_tags_cache

def set_vocab_cache(vocab_stoi, pad_id=0):
    """Set vocabulary string-to-index mapping for object token conversion."""
    global _vocab_stoi_cache
    _vocab_stoi_cache = {'stoi': vocab_stoi, 'pad_id': pad_id}

def get_object_ids_for_image(filename, max_objects=10):
    """
    Convert object tag strings to token IDs for a single image.
    
    Args:
        filename: Image filename (e.g., 'painting.jpg')
        max_objects: Maximum number of objects to include
    
    Returns:
        torch.Tensor of shape (max_objects,) with padding
    """
    global _object_tags_cache, _vocab_stoi_cache
    
    if _object_tags_cache is None or _vocab_stoi_cache is None:
        # Return padded zeros if caches not initialized
        return torch.zeros(max_objects, dtype=torch.long)
    
    pad_id = _vocab_stoi_cache['pad_id']
    stoi = _vocab_stoi_cache['stoi']
    
    # Get objects for this image
    objects = _object_tags_cache.get(filename, [])
    
    # Convert to token IDs
    token_ids = []
    for obj in objects[:max_objects]:
        if obj in stoi:
            token_ids.append(stoi[obj])
    
    # Pad to max_objects
    while len(token_ids) < max_objects:
        token_ids.append(pad_id)
    
    return torch.tensor(token_ids[:max_objects], dtype=torch.long)


def artemis_collate_fn(batch, max_objects=10):
    """
    Collate function for ArtEmis dataset with HYBRID OBJECT SUPPORT.
    
    Now returns object IDs for each image in the batch, enabling
    the model to learn from both visual features AND object concepts.
    
    Args:
        batch: List of (image, input_ids, target_ids, filename) tuples
        max_objects: Maximum number of object tags per image
    
    Returns:
        images: (B, 3, 224, 224)
        input_ids_padded: (B, max_seq_len)
        target_ids_padded: (B, max_seq_len)
        object_ids: (B, max_objects) - NEW! Padded object token IDs
        filenames: List of str (image filenames)
    """
    # Separate components
    images = torch.stack([b[0] for b in batch])
    input_ids = [b[1] for b in batch]
    target_ids = [b[2] for b in batch]
    filenames = [b[3] for b in batch] if len(batch[0]) == 4 else [None] * len(batch)
    
    # Pad sequences to the same length
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    target_ids_padded = pad_sequence(target_ids, batch_first=True, padding_value=0)
    
    # NEW: Get object IDs for each image
    object_ids_list = [get_object_ids_for_image(fn, max_objects) for fn in filenames]
    object_ids = torch.stack(object_ids_list)  # (B, max_objects)
    
    return images, input_ids_padded, target_ids_padded, object_ids, filenames
