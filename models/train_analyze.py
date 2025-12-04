import os
import argparse
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from PIL import Image
import json
import os
import sys
import cv2
import random

# Disable torch.compile due to Triton incompatibility on your GPU
torch._dynamo.config.suppress_errors = True
import torch._dynamo
torch._dynamo.config.suppress_errors = True

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# Use relative imports when running as part of the package
try:
    from .dataloader import ArtEmisDataset
    from .collate import artemis_collate_fn, load_object_tags_cache, set_vocab_cache
    from .model_transformer import OptimizedVLT
    from .vlm_model_manager import VLMModelManager
except ImportError:
    # Fallback for running script directly
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dataloader import ArtEmisDataset
    from collate import artemis_collate_fn, load_object_tags_cache, set_vocab_cache
    from model_transformer import OptimizedVLT
    from vlm_model_manager import VLMModelManager 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--save_dir', type=str, default='./results_automated')
    parser.add_argument('--vlm_optimized_dir', type=str, default='e:/A3/results_optimized_vlm',
                        help='Directory for optimized VLM models (with metric comparison)')
    
    # Training epochs - 50 is good for testing, 80 for full training
    parser.add_argument('--epochs', type=int, default=50)
    
    # Early stopping patience - wait 5 epochs before stopping
    parser.add_argument('--patience', type=int, default=5)
    
    # Batch size - Use 256 for 6GB GPU (you tested this safely)
    parser.add_argument('--batch_size', type=int, default=64)
    
    # Learning rate - 3e-4 is safer than 5e-4 (Karpathy constant)
    parser.add_argument('--lr', type=float, default=3e-4)
    
    # Unfreeze encoder at epoch 5 (let decoder learn basics first)
    parser.add_argument('--unfreeze_epoch', type=int, default=5)
    
    # Data loading workers - use 4 for Windows (8 causes shared memory errors)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Gradient accumulation - simulate larger batch size
    parser.add_argument('--grad_accum_steps', type=int, default=1,
                        help='Accumulate gradients over N steps (effective_batch = batch_size * N)')
    
    # Use Exponential Moving Average for stable predictions
    parser.add_argument('--use_ema', action='store_true', default=False,
                        help='Use EMA model for validation (smoother predictions)')
    
    # MODEL CAPACITY - Critical for fixing underfitting
    # embed_size: Token embedding dimension (512 for rich style/emotion representations)
    parser.add_argument('--embed_size', type=int, default=512)
    
    # hidden_size: Transformer decoder hidden dimension (512 for complex patterns)
    parser.add_argument('--hidden_size', type=int, default=512)
    
    # num_heads: Multi-head attention heads (8 is standard for 512-dim models)
    parser.add_argument('--num_heads', type=int, default=8)
    
    # num_layers: Transformer decoder depth (4 layers for good capacity)
    parser.add_argument('--num_layers', type=int, default=4)
    
    # dropout: Regularization (0.2 is less aggressive than 0.3)
    parser.add_argument('--dropout', type=float, default=0.2)
    
    return parser.parse_args()

# --- HELPER: Decode ---
def decode_caption(token_ids, vocab_itos, vocab):
    words = []
    for tid in token_ids:
        tid = int(tid)
        if tid == vocab['start_id']: continue
        if tid == vocab['end_id']: break
        if tid == vocab['pad_id']: continue
        words.append(vocab_itos[tid])
    return " ".join(words)

# --- HELPER: Load Object Tags ---
def load_object_tags(json_path):
    """
    Load pre-computed object tags from JSON file.
    
    CRITICAL PERFORMANCE NOTE:
    Object detection (FasterRCNN/DETR) is GPU-intensive (~200ms per image).
    Running it inside the training loop would reduce throughput from ~500 samples/sec
    to ~5 samples/sec (100x slowdown). ALWAYS pre-compute offline and save to disk.
    
    Returns:
        dict: {image_filename: ["object1", "object2", ...], ...}
    """
    with open(json_path, 'r') as f:
        return json.load(f)

def get_object_token_ids(image_filename, object_tags_dict, vocab_stoi, max_objects=10, pad_id=0):
    """
    Convert object tag strings to token IDs with padding.
    
    Args:
        image_filename: str - Name of the image file
        object_tags_dict: dict - Loaded from object_tags_precomputed.json
        vocab_stoi: dict - String to index mapping
        max_objects: int - Fixed size for batching (pad if fewer objects)
        pad_id: int - Padding token ID
    
    Returns:
        torch.Tensor: (max_objects,) - Padded object token IDs
    """
    import torch
    
    # Get object tags for this image (empty list if not found)
    objects = object_tags_dict.get(image_filename, [])
    
    # Convert to token IDs (only if in vocabulary, else skip)
    token_ids = []
    for obj in objects[:max_objects]:  # Limit to max_objects
        if obj in vocab_stoi:
            token_ids.append(vocab_stoi[obj])
    
    # Pad to max_objects length
    while len(token_ids) < max_objects:
        token_ids.append(pad_id)
    
    return torch.tensor(token_ids[:max_objects], dtype=torch.long)

# --- 1. TRAINING FUNCTIONS ---
def train_one_epoch(loader, model, optimizer, criterion, scaler, scheduler, device, grad_accum_steps=1, ema_model=None):
    """
    Train one epoch with HYBRID OBJECT-AWARE TRAINING.
    
    The model now learns: "When I see Object ID 452 (person), I should output 'person'"
    This bridges the gap between visual features and conceptual understanding.
    """
    model.train()
    running_loss = 0
    optimizer.zero_grad(set_to_none=True)
    
    for batch_idx, batch_data in enumerate(tqdm(loader, desc="Train", leave=False)):
        # NEW: Unpack 5 values from updated collate_fn (images, input_ids, target_ids, object_ids, filenames)
        if len(batch_data) == 5:
            images, input_ids, target_ids, object_ids, filenames = batch_data
            object_ids = object_ids.to(device, non_blocking=True)  # NEW: Object IDs to GPU
        elif len(batch_data) == 4:
            images, input_ids, target_ids, filenames = batch_data
            object_ids = None
        else:
            images, input_ids, target_ids = batch_data
            object_ids = None
            
        images = images.to(device, non_blocking=True)
        input_ids = input_ids.to(device, non_blocking=True)
        target_ids = target_ids.to(device, non_blocking=True)
        
        with autocast('cuda', dtype=torch.float16):
            # NEW: Pass object_ids to model for hybrid multi-modal learning
            logits = model(images, input_ids, object_input_ids=object_ids)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), target_ids.reshape(-1))
            loss = loss / grad_accum_steps  # Scale loss for gradient accumulation
        
        scaler.scale(loss).backward()
        
        # Only update weights every grad_accum_steps batches
        if (batch_idx + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            
            # Update EMA model if enabled
            if ema_model is not None:
                ema_model.update_parameters(model)
        
        running_loss += loss.item() * grad_accum_steps  # Undo scaling for logging
        
        if batch_idx % 100 == 0:
            mem_allocated = torch.cuda.memory_allocated() / 1e9
            mem_reserved = torch.cuda.memory_reserved() / 1e9
            obj_info = f" | Objects: {object_ids.shape[1] if object_ids is not None else 0}"
            tqdm.write(f"Batch {batch_idx} | Loss: {loss.item():.4f} | GPU: {mem_allocated:.2f}GB/{mem_reserved:.2f}GB{obj_info}")
    
    return running_loss / len(loader)

def validate(loader, model, criterion, device):
    """Validate with HYBRID OBJECT-AWARE inference."""
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for batch_data in tqdm(loader, desc="Val", leave=False):
            # NEW: Unpack 5 values from updated collate_fn
            if len(batch_data) == 5:
                images, input_ids, target_ids, object_ids, filenames = batch_data
                object_ids = object_ids.to(device, non_blocking=True)
            elif len(batch_data) == 4:
                images, input_ids, target_ids, filenames = batch_data
                object_ids = None
            else:
                images, input_ids, target_ids = batch_data
                object_ids = None
                
            images = images.to(device, non_blocking=True)
            input_ids = input_ids.to(device, non_blocking=True)
            target_ids = target_ids.to(device, non_blocking=True)
            
            with autocast('cuda', dtype=torch.float16):
                # NEW: Pass object_ids to model
                logits = model(images, input_ids, object_input_ids=object_ids)
                loss = criterion(logits.reshape(-1, logits.shape[-1]), target_ids.reshape(-1))
            
            running_loss += loss.item()
    return running_loss / len(loader)

# --- 2. PLOTTING FUNCTIONS ---
def plot_losses(train_losses, val_losses, save_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training vs Validation Loss')
    plt.savefig(os.path.join(save_dir, 'vlt_training_loss.png'))
    plt.close()
    print("✅ Loss Graphs Saved.")

# --- 3. METRICS FUNCTIONS (BLEU/ROUGE) ---
def calculate_metrics(model, loader, vocab, device, save_dir, object_tags_dict=None, vocab_stoi=None):
    """
    Calculate BLEU/ROUGE metrics using HYBRID OBJECT-AWARE generation.
    """
    print("📊 Computing BLEU & ROUGE with Hybrid Model...")
    model.eval()
    refs = []
    hyps = []
    
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l_scores = []

    with torch.no_grad():
        for batch_data in tqdm(loader, desc="Metrics"):
            # NEW: Unpack 5 values from updated collate_fn
            if len(batch_data) == 5:
                images, _, target_ids, object_ids, filenames = batch_data
                object_ids = object_ids.to(device)
            elif len(batch_data) == 4:
                images, _, target_ids, filenames = batch_data
                object_ids = None
            else:
                images, _, target_ids = batch_data
                filenames = None
                object_ids = None
                
            images = images.to(device)
            
            # If object_ids not in batch but we have object_tags_dict, build them manually
            if object_ids is None and object_tags_dict is not None and vocab_stoi is not None and filenames is not None:
                batch_objects = []
                for fname in filenames:
                    obj_ids = get_object_token_ids(fname, object_tags_dict, vocab_stoi, 
                                                   max_objects=10, pad_id=model.pad_token_id)
                    batch_objects.append(obj_ids)
                object_ids = torch.stack(batch_objects).to(device)
            
            # Beam Search with Beam=3 for best results (now with object_ids)
            generated = model.generate_beam(images, object_ids, beam_size=3, max_len=40)
            
            for i in range(len(generated)):
                ref_text = decode_caption(target_ids[i], vocab['itos'], vocab)
                pred_text = decode_caption(generated[i], vocab['itos'], vocab)
                
                refs.append([ref_text.split()])
                hyps.append(pred_text.split())
                
                # ROUGE
                scores = scorer.score(ref_text, pred_text)
                rouge_l_scores.append(scores['rougeL'])

    # BLEU
    cc = SmoothingFunction().method1
    b1 = corpus_bleu(refs, hyps, weights=(1.0, 0, 0, 0), smoothing_function=cc)
    b2 = corpus_bleu(refs, hyps, weights=(0.5, 0.5, 0, 0), smoothing_function=cc)
    b4 = corpus_bleu(refs, hyps, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=cc)

    # Average ROUGE
    r_precision = np.mean([s.precision for s in rouge_l_scores])
    r_recall = np.mean([s.recall for s in rouge_l_scores])
    r_f1 = np.mean([s.fmeasure for s in rouge_l_scores])

    results = {
        "BLEU-1": b1, "BLEU-2": b2, "BLEU-4": b4,
        "ROUGE-L-P": r_precision, "ROUGE-L-R": r_recall, "ROUGE-L-F1": r_f1
    }
    
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"✅ Metrics Saved: BLEU-4: {b4:.4f} | ROUGE-L-F1: {r_f1:.4f}")

# --- 4. ATTENTION VISUALIZATION ---
# Global hook storage
attn_weights = {}
def get_activation(name):
    def hook(model, input, output):
        # Output of MultiheadAttention is (attn_output, attn_output_weights)
        # We want weights: (Batch, Target_Len, Source_Len)
        if isinstance(output, tuple):
            attn_weights[name] = output[1].detach().cpu()
    return hook

def visualize_attention(model, dataset, vocab, device, save_dir):
    print("👁️ Generating Attention Maps...")
    model.eval()
    
    # Get one sample
    img, input_ids, _ = dataset[0]
    img_input = img.unsqueeze(0).to(device)
    input_ids = input_ids.unsqueeze(0).to(device)
    
    # Register hook BEFORE forward pass
    handle = model.decoder.layers[-1].multihead_attn.register_forward_hook(get_activation('cross_attn'))
    
    # Run TRAINING-style forward (not generate) to get attention
    with torch.no_grad():
        _ = model(img_input, input_ids)
    
    # Get caption text
    caption = decode_caption(input_ids[0], vocab['itos'], vocab).split()
    
    # Check if hook captured anything
    if 'cross_attn' not in attn_weights or attn_weights['cross_attn'] is None:
        print("⚠️ Attention weights not captured. Skipping visualization.")
        handle.remove()
        return
    
    # Get Weights
    weights = attn_weights['cross_attn'][0]
    
    # Unnormalize image for display
    disp_img = img.permute(1, 2, 0).numpy()
    disp_img = (disp_img * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
    disp_img = np.clip(disp_img, 0, 1)
    
    # Plot
    fig = plt.figure(figsize=(15, 8))
    num_words = min(10, len(caption))
    
    for i in range(num_words):
        if i >= weights.shape[0]:
            break
            
        w = weights[i, :].reshape(14, 14).numpy()
        w = cv2.resize(w, (224, 224))
        
        ax = fig.add_subplot(2, 5, i+1)
        ax.imshow(disp_img)
        ax.imshow(w, cmap='jet', alpha=0.5)
        ax.set_title(caption[i] if i < len(caption) else '')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "attention_map.png"))
    plt.close()
    print("✅ Attention Map Saved.")
    handle.remove()

# --- 5. ABLATION STUDY ---
def run_ablation(model, dataset, vocab, device, save_dir):
    print("🧠 Running Ablation Study...")
    model.eval()
    img, _, _ = dataset[random.randint(0, 100)] # Random image
    img_input = img.unsqueeze(0).to(device)
    
    # A. Beam Size Comparison
    f = open(os.path.join(save_dir, "ablation_results.txt"), "w")
    f.write("--- Beam Search Comparison ---\n")
    for k in [1, 3, 5]:
        out = model.generate_beam(img_input, beam_size=k)
        text = decode_caption(out[0], vocab['itos'], vocab)
        f.write(f"Beam {k}: {text}\n")
    
    # B. Emotion Injection (Manual Forcing)
    # This requires modifying generation to accept a forced start sequence
    # For now, we simulate it by checking if we can generate different styles
    # (Full injection requires changing the generate() loop, which is complex for this script)
    f.write("\n--- (Note: Emotion Injection requires manual start token forcing in generate()) ---\n")
    f.close()
    print("✅ Ablation Results Saved.")

# --- MAIN AUTOMATION ---
def main():
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 STARTING AUTOMATED RUN on {device}")

    # 1. Load Data - Use flexible path resolution
    # Get the project root (IML a3 folder)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Go up from models/ to IML a3/
    
    # Define paths relative to project root with fallbacks
    def find_file(possible_paths):
        """Find the first existing file from a list of paths."""
        for path in possible_paths:
            if os.path.exists(path):
                return path
        return possible_paths[0]  # Return first path even if not exists
    
    vocab_path = find_file([
        os.path.join(project_root, 'code', 'Pkl Files', 'vocab.pkl'),
        os.path.join(project_root, 'code', 'preprocessing', 'vocab.pkl'),
        r"E:\A3\IML a3\code\preprocessing\vocab.pkl",
    ])
    
    pkl_path = find_file([
        os.path.join(project_root, 'code', 'Pkl Files', 'preprocessed_dataset_with_tokens.pkl'),
        os.path.join(project_root, 'code', 'preprocessing', 'preprocessed_dataset_with_tokens.pkl'),
        r"E:\A3\IML a3\code\preprocessing\preprocessed_dataset_with_tokens.pkl",
    ])
    
    img_folder = find_file([
        os.path.join(project_root, 'data', 'final_data'),
        r"E:\A3\IML a3\data\final_data",
    ])
    
    object_tags_path = find_file([
        os.path.join(project_root, 'code', 'Pkl Files', 'object_tags_precomputed.json'),
        os.path.join(project_root, 'code', 'preprocessing', 'object_tags_precomputed.json'),
        os.path.join(project_root, 'data', 'object_tags_precomputed.json'),
        r"E:\A3\IML a3\code\preprocessing\object_tags_precomputed.json",
    ])
    
    print(f"📁 Using paths:")
    print(f"   Vocab:       {vocab_path}")
    print(f"   Dataset:     {pkl_path}")
    print(f"   Images:      {img_folder}")
    print(f"   Object Tags: {object_tags_path}")
    
    with open(vocab_path, 'rb') as f: vocab = pickle.load(f)
    with open(pkl_path, 'rb') as f: full_df = pickle.load(f)
    
    # Load pre-computed object tags
    print("📦 Loading pre-computed object tags...")
    object_tags_dict = load_object_tags(object_tags_path)
    print(f"   Loaded object tags for {len(object_tags_dict)} images")
    vocab_stoi = vocab['stoi']  # String-to-index mapping for object tokens
    
    # NEW: Initialize collate function caches for hybrid training
    # load_object_tags_cache and set_vocab_cache already imported at top
    print("🔧 Initializing hybrid object tag pipeline...")
    load_object_tags_cache(object_tags_path)
    set_vocab_cache(vocab_stoi, pad_id=vocab['pad_id'])
    print("   ✅ Object tags will be included in every training batch!")
    
    indices = list(range(len(full_df)))
    np.random.seed(42)
    np.random.shuffle(indices)
    
    # Split: 90% train, 5% val, 5% test
    val_split = int(0.05 * len(full_df))    # First 5%
    test_split = int(0.10 * len(full_df))   # Next 5% (total 10%)
    
    train_ds = Subset(ArtEmisDataset(pkl_path, img_folder, split='train'), indices[test_split:])
    val_ds = Subset(ArtEmisDataset(pkl_path, img_folder, split='val'), indices[:val_split])
    test_ds = Subset(ArtEmisDataset(pkl_path, img_folder, split='val'), indices[val_split:test_split])
    
    print(f"📊 Dataset splits (90/5/5):")
    print(f"   Train: {len(train_ds)} samples ({len(train_ds)/len(full_df)*100:.1f}%)")
    print(f"   Val:   {len(val_ds)} samples ({len(val_ds)/len(full_df)*100:.1f}%)")
    print(f"   Test:  {len(test_ds)} samples ({len(test_ds)/len(full_df)*100:.1f}%)")
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers if args.num_workers > 0 else 0,
        collate_fn=artemis_collate_fn,
        pin_memory=True,
        persistent_workers=args.num_workers > 1,  # Only if multiple workers
        prefetch_factor=2  # Reduced from 4
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers if args.num_workers > 0 else 0,
        collate_fn=artemis_collate_fn,
        pin_memory=True,
        persistent_workers=args.num_workers > 1,
        prefetch_factor=2
    )
    test_loader = DataLoader(
        test_ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers if args.num_workers > 0 else 0,
        collate_fn=artemis_collate_fn,
        pin_memory=True,
        persistent_workers=args.num_workers > 1,
        prefetch_factor=2
    )

    # 2. Model & Optimization
    print("Initializing Hybrid OptimizedVLT...")
    model = OptimizedVLT(
        vocab_size=vocab['vocab_size'],
        pad_token_id=vocab['pad_id'],
        start_token_id=vocab['start_id'],
        end_token_id=vocab['end_id'],
        max_objects=10,              # NEW: Max object tags per image
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout_rate=args.dropout,
        device=device
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['pad_id'], label_smoothing=0.1)
    
    # OPTIMIZED: Better AdamW hyperparameters (from GPT-3/Llama papers)
    try:
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=args.lr, 
            betas=(0.9, 0.95),      # More stable than default (0.9, 0.999)
            eps=1e-8,
            weight_decay=0.01,
            fused=True               # Faster fused kernel on CUDA
        )
        print("⚡ Using fused AdamW (faster on GPU)")
    except:
        # Fallback if fused not available
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=args.lr, 
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.01
        )
    
    # FIXED: Use ReduceLROnPlateau to break plateau
    # Reduces LR by 0.5x when val_loss doesn't improve for 2 epochs
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',           # Minimize validation loss
        factor=0.5,           # Multiply LR by 0.5 when triggered
        patience=2,           # Wait 2 epochs before reducing
        verbose=True,         # Print LR changes
        min_lr=1e-6          # Don't go below 1e-6
    )
    
    # Keep OneCycleLR for per-batch updates (warmup in early epochs)
    batch_scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=args.lr, 
        steps_per_epoch=len(train_loader), 
        epochs=args.epochs, 
        pct_start=0.1
    )
    scaler = GradScaler('cuda')
    
    # OPTIMIZATION: Exponential Moving Average (optional, use --use_ema flag)
    ema_model = None
    if args.use_ema:
        try:
            from torch.optim.swa_utils import AveragedModel
            ema_model = AveragedModel(
                model, 
                multi_avg_fn=lambda averaged_model_parameter, model_parameter, num_averaged: 
                    0.999 * averaged_model_parameter + 0.001 * model_parameter
            )
            print("📊 EMA enabled (decay=0.999) for smoother predictions")
        except Exception as e:
            print(f"⚠️ EMA not available: {e}")
    
    # OPTIMIZATION: torch.compile DISABLED
    # Your GPU doesn't support Triton (triton_key import error)
    # torch.compile would give 20-30% speedup but requires compatible hardware
    print("ℹ️  torch.compile disabled (Triton not compatible with your GPU)")


    # 3. Training Loop
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience = 0
    
    # Track multiple best models with different criteria
    best_models = {
        'best_val_loss': {'loss': float('inf'), 'epoch': 0},
        'best_train_loss': {'loss': float('inf'), 'epoch': 0},
        'best_bleu': {'score': 0.0, 'epoch': 0}
    }
    
    # Initialize VLM Model Manager for optimized model saving
    print(f"\n📦 Initializing VLM Model Manager...")
    vlm_manager = VLMModelManager(save_dir=args.vlm_optimized_dir)
    print(f"   Save directory: {args.vlm_optimized_dir}")
    print(f"   Models will be saved based on metric comparison")
    
    for epoch in range(1, args.epochs + 1):
        # Print current learning rate for debugging
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch}/{args.epochs} | Learning Rate: {current_lr:.2e}")
        print(f"{'='*60}")
        
        if epoch == args.unfreeze_epoch: 
            print("🔓 Unfreezing encoder last layers...")
            model.unfreeze_last_layers()
        
        t_loss = train_one_epoch(train_loader, model, optimizer, criterion, scaler, batch_scheduler, device, 
                                args.grad_accum_steps, ema_model)
        
        # Validate with EMA model if enabled, otherwise use regular model
        eval_model = ema_model.module if ema_model is not None else model
        v_loss = validate(val_loader, eval_model, criterion, device)
        
        # Update ReduceLROnPlateau based on validation loss
        plateau_scheduler.step(v_loss)
        
        train_losses.append(t_loss)
        val_losses.append(v_loss)
        print(f"\n📊 Epoch {epoch} Summary:")
        print(f"   Train Loss: {t_loss:.4f}")
        print(f"   Val Loss:   {v_loss:.4f}")
        print(f"   Best Val:   {best_val_loss:.4f}")
        
        # Save to VLM Manager (compares and keeps only best)
        vlm_metrics = {
            'train_loss': t_loss,
            'val_loss': v_loss,
            'learning_rate': current_lr
        }
        is_best = vlm_manager.save_if_best(
            model=model,
            metrics=vlm_metrics,
            epoch=epoch,
            criterion='val_loss',  # Can change to 'bleu4', 'rouge_l_f1', etc.
            additional_info={'model_type': 'OptimizedVLT'}
        )
        
        # Save best validation loss model
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            patience = 0
            
            # Create timestamped folder for this best model
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            model_dir = os.path.join(args.save_dir, f"best_val_loss_epoch{epoch}_{timestamp}")
            os.makedirs(model_dir, exist_ok=True)
            
            # Save model
            torch.save(model.state_dict(), os.path.join(model_dir, "model.pt"))
            
            # Save metrics
            metrics = {
                'epoch': epoch,
                'train_loss': t_loss,
                'val_loss': v_loss,
                'learning_rate': current_lr,
                'criterion': 'best_validation_loss',
                'timestamp': timestamp
            }
            with open(os.path.join(model_dir, "metrics.json"), 'w') as f:
                json.dump(metrics, f, indent=4)
            
            print(f"✅ Best validation model saved to: {model_dir}")
            best_models['best_val_loss'] = {'loss': v_loss, 'epoch': epoch, 'path': model_dir}
            
            # Also save to default location for backward compatibility
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pt"))
        else:
            patience += 1
            if patience >= args.patience:
                print("Early Stopping Reached.")
                break
        
        # Save best training loss model (every 5 epochs to avoid spam)
        if epoch % 5 == 0 and t_loss < best_models['best_train_loss']['loss']:
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            model_dir = os.path.join(args.save_dir, f"best_train_loss_epoch{epoch}_{timestamp}")
            os.makedirs(model_dir, exist_ok=True)
            
            torch.save(model.state_dict(), os.path.join(model_dir, "model.pt"))
            
            metrics = {
                'epoch': epoch,
                'train_loss': t_loss,
                'val_loss': v_loss,
                'learning_rate': current_lr,
                'criterion': 'best_training_loss',
                'timestamp': timestamp
            }
            with open(os.path.join(model_dir, "metrics.json"), 'w') as f:
                json.dump(metrics, f, indent=4)
            
            print(f"💾 Best training loss model saved to: {model_dir}")
            best_models['best_train_loss'] = {'loss': t_loss, 'epoch': epoch, 'path': model_dir}
                
    # 4. Save Logs
    with open(os.path.join(args.save_dir, "loss_history.pkl"), "wb") as f:
        pickle.dump({'train': train_losses, 'val': val_losses}, f)
    plot_losses(train_losses, val_losses, args.save_dir)
    
    # Print VLM Manager summary
    print("\n" + "="*70)
    vlm_manager.print_summary()
    print("="*70)
    
    # Save summary of all best models
    summary_path = os.path.join(args.save_dir, "best_models_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(best_models, f, indent=4)
    print(f"\n📋 Best models summary saved to: {summary_path}")

    # 5. FINAL ANALYSIS (The "While You Sleep" Part)
    print("\n🏁 Training Done. Starting Analysis...")
    
    # Reload Best Model (best validation loss)
    best_model_path = os.path.join(args.save_dir, "best_model.pt")
    model.load_state_dict(torch.load(best_model_path))
    print(f"📥 Loaded best model from epoch {best_models['best_val_loss']['epoch']}")
    
    # A. Evaluate on TEST SET (final unseen data)
    print("\n" + "="*70)
    print("🧪 EVALUATING ON TEST SET")
    print("="*70)
    test_loss = validate(test_loader, model, criterion, device)
    print(f"✅ Test Loss: {test_loss:.4f}")
    
    # Save test metrics
    test_metrics_path = os.path.join(args.save_dir, "test_metrics.json")
    with open(test_metrics_path, 'w') as f:
        json.dump({
            'test_loss': test_loss,
            'best_epoch': best_models['best_val_loss']['epoch'],
            'val_loss': best_models['best_val_loss']['loss']
        }, f, indent=4)
    print(f"💾 Test metrics saved to: {test_metrics_path}")
    
    # B. Calculate BLEU/ROUGE on test set
    print("\n📊 Computing BLEU/ROUGE on test set...")
    calculate_metrics(model, test_loader, vocab, device, args.save_dir, object_tags_dict, vocab_stoi)
    
    # Load the metrics and update VLM manager with BLEU/ROUGE scores
    metrics_file = os.path.join(args.save_dir, "metrics.json")
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            bleu_rouge_metrics = json.load(f)
        
        # Update VLM manager with final metrics (this won't save unless better)
        final_metrics = {
            'train_loss': train_losses[-1] if train_losses else None,
            'val_loss': val_losses[-1] if val_losses else None,
            'test_loss': test_loss,
            'bleu1': bleu_rouge_metrics.get('BLEU-1', 0),
            'bleu2': bleu_rouge_metrics.get('BLEU-2', 0),
            'bleu4': bleu_rouge_metrics.get('BLEU-4', 0),
            'rouge_l_f1': bleu_rouge_metrics.get('ROUGE-L-F1', 0),
            'rouge_l_p': bleu_rouge_metrics.get('ROUGE-L-P', 0),
            'rouge_l_r': bleu_rouge_metrics.get('ROUGE-L-R', 0)
        }
        
        # Check if this is better than best model based on BLEU-4
        print("\n🔍 Checking if final model (with BLEU/ROUGE) is better than best saved model...")
        vlm_manager.save_if_best(
            model=model,
            metrics=final_metrics,
            epoch=best_models['best_val_loss']['epoch'],
            criterion='bleu4',  # Use BLEU-4 as final criterion
            additional_info={'model_type': 'OptimizedVLT', 'phase': 'final_evaluation'}
        )
    
    # C. Visualize Attention (on validation set)
    visualize_attention(model, val_ds, vocab, device, args.save_dir)
    
    # D. Ablation (on validation set)
    run_ablation(model, val_ds, vocab, device, args.save_dir)
    
    print("\n" + "="*70)
    print("🎉 ALL TASKS COMPLETED!")
    print("="*70)
    print(f"\n📊 Final Results:")
    print(f"   • Val Loss:  {best_models['best_val_loss']['loss']:.4f}")
    print(f"   • Test Loss: {test_loss:.4f}")
    print(f"\n📁 Saved Models:")
    for criterion, info in best_models.items():
        if 'path' in info:
            print(f"   • {criterion}: Epoch {info['epoch']} → {info['path']}")
    
    # Final VLM Manager summary
    print("\n" + "="*70)
    print("🏆 OPTIMIZED VLM MODEL (Best of All Iterations)")
    print("="*70)
    vlm_manager.print_summary()
    
    print("\n💤 YOU CAN WAKE UP NOW.")
    print("="*70)

if __name__ == "__main__":
    import random # Needed for ablation
    main()