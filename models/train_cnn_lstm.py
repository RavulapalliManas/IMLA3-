"""
Training Script for CNN + LSTM Image Captioning Model
======================================================
Features:
    - Mixed precision training (AMP) for RTX 3060
    - Gradient clipping & accumulation
    - Teacher forcing with scheduled sampling
    - BLEU score tracking
    - Early stopping with patience
    - Checkpoint saving & resuming
    - Comprehensive logging

Optimized for RTX 3060 Laptop GPU (6GB VRAM)
"""

import os
import sys
import json
import time
import pickle
import random
import argparse
from datetime import datetime
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn_lstm_model import CNNLSTMCaptioner, create_model
from models.dataloader import ArtEmisDataset
from models.collate import artemis_collate_fn


# =============================================================================
# SECTION A: Configuration
# =============================================================================

DEFAULT_CONFIG = {
    # Model architecture
    'vocab_size': 5000,          # Will be updated from vocab
    'embed_dim': 256,
    'hidden_dim': 512,
    'num_layers': 2,
    'dropout': 0.3,
    'tfidf_dim': 300,
    'max_len': 33,
    
    # Training hyperparameters
    'batch_size': 16,            # Conservative for 6GB VRAM
    'learning_rate': 3e-4,
    'weight_decay': 1e-5,
    'epochs': 50,
    'patience': 7,               # Early stopping patience
    'grad_clip': 1.0,
    'grad_accumulation_steps': 2,  # Effective batch = 32
    
    # Scheduled sampling
    'teacher_forcing_start': 1.0,   # Start with 100% teacher forcing
    'teacher_forcing_end': 0.6,     # End with 60% teacher forcing
    'teacher_forcing_decay': 0.95,  # Decay per epoch
    
    # Mixed precision
    'use_amp': True,
    
    # Checkpointing
    'save_every': 5,             # Save every N epochs
    'log_every': 50,             # Log every N batches
    
    # Special tokens
    'pad_id': 0,
    'start_id': 2,
    'end_id': 3,
    'unk_id': 1
}


# =============================================================================
# SECTION B: Dataset with Emotion Labels
# =============================================================================

class ArtEmisDatasetWithEmotion(ArtEmisDataset):
    """
    Extended ArtEmis dataset that also returns emotion labels.
    """
    
    EMOTION_MAP = {
        'contentment': 0, 'awe': 1, 'something_else': 2, 'sadness': 3,
        'amusement': 4, 'fear': 5, 'excitement': 6, 'disgust': 7, 'anger': 8
    }
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Get base items from parent class
        image, input_ids, target_ids, img_name = super().__getitem__(idx)
        
        # Get emotion label
        emotion_str = row.get('emotion', 'something_else')
        if isinstance(emotion_str, str):
            emotion_str = emotion_str.lower().replace(' ', '_')
        emotion_id = self.EMOTION_MAP.get(emotion_str, 2)  # Default to 'something_else'
        
        return image, input_ids, target_ids, img_name, emotion_id


def collate_with_emotion(batch):
    """
    Collate function that handles emotion labels.
    
    Returns:
        images: (B, 3, 224, 224)
        input_ids: (B, max_seq_len)
        target_ids: (B, max_seq_len)
        emotion_ids: (B,)
        filenames: List[str]
    """
    from torch.nn.utils.rnn import pad_sequence
    
    images = torch.stack([b[0] for b in batch])
    input_ids = [b[1] for b in batch]
    target_ids = [b[2] for b in batch]
    filenames = [b[3] for b in batch]
    emotion_ids = torch.tensor([b[4] for b in batch], dtype=torch.long)
    
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    target_ids_padded = pad_sequence(target_ids, batch_first=True, padding_value=0)
    
    return images, input_ids_padded, target_ids_padded, emotion_ids, filenames


# =============================================================================
# SECTION C: BLEU Score Calculation
# =============================================================================

def calculate_bleu(references: List[List[str]], hypotheses: List[str], n: int = 4) -> Dict[str, float]:
    """
    Calculate BLEU-1 to BLEU-4 scores.
    
    Args:
        references: List of reference sentences (tokenized)
        hypotheses: List of hypothesis sentences (tokenized)
        n: Maximum n-gram order
        
    Returns:
        Dictionary with BLEU-1 to BLEU-n scores
    """
    from collections import Counter
    import math
    
    def ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    def modified_precision(refs, hyp, n):
        hyp_ngrams = Counter(ngrams(hyp, n))
        max_counts = Counter()
        
        for ref in refs:
            ref_ngrams = Counter(ngrams(ref, n))
            for ng in hyp_ngrams:
                max_counts[ng] = max(max_counts[ng], ref_ngrams[ng])
        
        clipped = sum(min(hyp_ngrams[ng], max_counts[ng]) for ng in hyp_ngrams)
        total = sum(hyp_ngrams.values())
        
        return clipped / max(total, 1)
    
    # Calculate brevity penalty
    hyp_len = sum(len(h) for h in hypotheses)
    ref_lens = []
    for h, refs in zip(hypotheses, references):
        closest = min(refs, key=lambda r: abs(len(r) - len(h)))
        ref_lens.append(len(closest))
    ref_len = sum(ref_lens)
    
    if hyp_len <= ref_len:
        bp = math.exp(1 - ref_len / max(hyp_len, 1))
    else:
        bp = 1.0
    
    # Calculate n-gram precisions
    precisions = []
    for i in range(1, n + 1):
        precs = []
        for h, refs in zip(hypotheses, references):
            precs.append(modified_precision(refs, h, i))
        precisions.append(sum(precs) / max(len(precs), 1))
    
    # Calculate BLEU scores
    results = {}
    for i in range(1, n + 1):
        if 0 in precisions[:i]:
            results[f'bleu_{i}'] = 0.0
        else:
            log_prec = sum(math.log(p) for p in precisions[:i]) / i
            results[f'bleu_{i}'] = bp * math.exp(log_prec)
    
    return results


# =============================================================================
# SECTION D: Training Functions
# =============================================================================

class Trainer:
    """
    Training manager for CNN-LSTM captioner.
    """
    
    def __init__(
        self,
        model: CNNLSTMCaptioner,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        vocab: Dict,
        device: str = 'cuda',
        output_dir: str = './outputs'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.vocab = vocab
        self.device = device
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Loss function (ignore padding)
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=config['pad_id'],
            label_smoothing=0.1  # Label smoothing for regularization
        )
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        
        # Mixed precision scaler
        self.scaler = GradScaler() if config['use_amp'] else None
        
        # Tracking
        self.best_val_loss = float('inf')
        self.best_bleu = 0.0
        self.patience_counter = 0
        self.current_epoch = 0
        self.teacher_forcing_ratio = config['teacher_forcing_start']
        
        # Logging
        self.train_losses = []
        self.val_losses = []
        self.bleu_scores = []
    
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        self.optimizer.zero_grad()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(pbar):
            images, input_ids, target_ids, emotion_ids, _ = batch
            
            # Move to device
            images = images.to(self.device)
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            emotion_ids = emotion_ids.to(self.device)
            
            # Forward pass with mixed precision
            if self.config['use_amp']:
                with autocast():
                    logits = self.model(images, input_ids, emotion_ids)
                    
                    # Reshape for loss: (B*S, vocab) vs (B*S,)
                    B, S, V = logits.shape
                    loss = self.criterion(
                        logits.view(B * S, V),
                        target_ids.view(B * S)
                    )
                    loss = loss / self.config['grad_accumulation_steps']
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.config['grad_accumulation_steps'] == 0:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['grad_clip']
                    )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                logits = self.model(images, input_ids, emotion_ids)
                
                B, S, V = logits.shape
                loss = self.criterion(
                    logits.view(B * S, V),
                    target_ids.view(B * S)
                )
                loss = loss / self.config['grad_accumulation_steps']
                
                loss.backward()
                
                if (batch_idx + 1) % self.config['grad_accumulation_steps'] == 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['grad_clip']
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Track loss
            total_loss += loss.item() * self.config['grad_accumulation_steps']
            num_batches += 1
            
            # Update progress bar
            if batch_idx % self.config['log_every'] == 0:
                pbar.set_postfix({
                    'loss': f"{total_loss / num_batches:.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}",
                    'tf': f"{self.teacher_forcing_ratio:.2f}"
                })
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, Dict[str, float]]:
        """
        Validate the model.
        
        Returns:
            Tuple of (average loss, BLEU scores dict)
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_references = []
        all_hypotheses = []
        
        itos = self.vocab.get('itos', {})
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            images, input_ids, target_ids, emotion_ids, _ = batch
            
            images = images.to(self.device)
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            emotion_ids = emotion_ids.to(self.device)
            
            # Calculate loss
            if self.config['use_amp']:
                with autocast():
                    logits = self.model(images, input_ids, emotion_ids)
                    B, S, V = logits.shape
                    loss = self.criterion(
                        logits.view(B * S, V),
                        target_ids.view(B * S)
                    )
            else:
                logits = self.model(images, input_ids, emotion_ids)
                B, S, V = logits.shape
                loss = self.criterion(
                    logits.view(B * S, V),
                    target_ids.view(B * S)
                )
            
            total_loss += loss.item()
            num_batches += 1
            
            # Generate captions for BLEU calculation
            generated = self.model.generate_greedy(images, emotion_ids)
            
            # Convert to text
            for i in range(images.size(0)):
                # Reference
                ref_tokens = []
                for tid in target_ids[i].cpu().tolist():
                    if tid == self.config['end_id']:
                        break
                    if tid not in [self.config['pad_id'], self.config['start_id']]:
                        token = itos.get(tid, '<unk>')
                        ref_tokens.append(token)
                
                # Hypothesis
                hyp_tokens = []
                for tid in generated[i].cpu().tolist():
                    if tid == self.config['end_id']:
                        break
                    if tid not in [self.config['pad_id'], self.config['start_id']]:
                        token = itos.get(tid, '<unk>')
                        hyp_tokens.append(token)
                
                all_references.append([ref_tokens])
                all_hypotheses.append(hyp_tokens)
        
        avg_loss = total_loss / num_batches
        
        # Calculate BLEU scores
        bleu_scores = calculate_bleu(all_references, all_hypotheses)
        
        return avg_loss, bleu_scores
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_bleu': self.best_bleu,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'bleu_scores': self.bleu_scores,
            'teacher_forcing_ratio': self.teacher_forcing_ratio
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        path = os.path.join(self.output_dir, filename)
        torch.save(checkpoint, path)
        print(f"  ✓ Saved checkpoint: {filename}")
        
        if is_best:
            best_path = os.path.join(self.output_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"  ✓ Saved best model")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = os.path.join(self.output_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_bleu = checkpoint.get('best_bleu', 0.0)
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.bleu_scores = checkpoint.get('bleu_scores', [])
        self.teacher_forcing_ratio = checkpoint.get(
            'teacher_forcing_ratio', 
            self.config['teacher_forcing_start']
        )
        
        print(f"  ✓ Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self):
        """
        Full training loop.
        """
        print("\n" + "=" * 60)
        print("Starting Training")
        print("=" * 60)
        print(f"  Device: {self.device}")
        print(f"  Batch size: {self.config['batch_size']}")
        print(f"  Effective batch: {self.config['batch_size'] * self.config['grad_accumulation_steps']}")
        print(f"  Learning rate: {self.config['learning_rate']}")
        print(f"  Epochs: {self.config['epochs']}")
        print(f"  Early stopping patience: {self.config['patience']}")
        print(f"  Mixed precision: {self.config['use_amp']}")
        print(f"  Output dir: {self.output_dir}")
        print("=" * 60 + "\n")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config['epochs']):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, bleu_scores = self.validate()
            self.val_losses.append(val_loss)
            self.bleu_scores.append(bleu_scores)
            
            # Update learning rate
            self.scheduler.step()
            
            # Update teacher forcing ratio
            self.teacher_forcing_ratio = max(
                self.config['teacher_forcing_end'],
                self.teacher_forcing_ratio * self.config['teacher_forcing_decay']
            )
            
            # Logging
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch + 1}/{self.config['epochs']} ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  BLEU-1: {bleu_scores['bleu_1']:.4f}")
            print(f"  BLEU-4: {bleu_scores['bleu_4']:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            print(f"  TF Ratio: {self.teacher_forcing_ratio:.2f}")
            
            # Check for improvement
            is_best = False
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                is_best = True
                print(f"  ★ New best validation loss!")
            else:
                self.patience_counter += 1
                print(f"  Patience: {self.patience_counter}/{self.config['patience']}")
            
            if bleu_scores['bleu_4'] > self.best_bleu:
                self.best_bleu = bleu_scores['bleu_4']
            
            # Save checkpoint
            if (epoch + 1) % self.config['save_every'] == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt')
            
            if is_best:
                self.save_checkpoint('best_model.pt', is_best=True)
            
            # Early stopping
            if self.patience_counter >= self.config['patience']:
                print(f"\n⚠ Early stopping triggered at epoch {epoch + 1}")
                break
        
        # Final save
        self.save_checkpoint('final_model.pt')
        
        # Training summary
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"  Total time: {total_time / 60:.1f} minutes")
        print(f"  Best val loss: {self.best_val_loss:.4f}")
        print(f"  Best BLEU-4: {self.best_bleu:.4f}")
        print(f"  Final epoch: {self.current_epoch + 1}")
        print("=" * 60)
        
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'bleu_scores': self.bleu_scores,
            'config': self.config
        }
        with open(os.path.join(self.output_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)


# =============================================================================
# SECTION E: Main Training Script
# =============================================================================

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train CNN-LSTM Captioner')
    
    # Data paths
    parser.add_argument('--data_pkl', type=str, required=True,
                        help='Path to preprocessed dataset pickle')
    parser.add_argument('--img_folder', type=str, required=True,
                        help='Path to image folder')
    parser.add_argument('--vocab_pkl', type=str, required=True,
                        help='Path to vocabulary pickle')
    parser.add_argument('--tfidf_pkl', type=str, default=None,
                        help='Path to TF-IDF vectors pickle')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./outputs/cnn_lstm',
                        help='Output directory for checkpoints')
    
    # Training params
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--patience', type=int, default=7)
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                        help='Checkpoint to resume from')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu', 'mps'])
    
    args = parser.parse_args()
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    # Load vocabulary
    print("Loading vocabulary...")
    with open(args.vocab_pkl, 'rb') as f:
        vocab = pickle.load(f)
    
    vocab_size = vocab.get('vocab_size', len(vocab.get('itos', {})))
    print(f"  Vocabulary size: {vocab_size}")
    
    # Update config
    config = DEFAULT_CONFIG.copy()
    config['vocab_size'] = vocab_size
    config['batch_size'] = args.batch_size
    config['epochs'] = args.epochs
    config['learning_rate'] = args.lr
    config['patience'] = args.patience
    config['pad_id'] = vocab.get('pad_id', 0)
    config['start_id'] = vocab.get('start_id', 2)
    config['end_id'] = vocab.get('end_id', 3)
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = ArtEmisDatasetWithEmotion(
        pkl_path=args.data_pkl,
        img_folder=args.img_folder,
        split='train'
    )
    
    val_dataset = ArtEmisDatasetWithEmotion(
        pkl_path=args.data_pkl,
        img_folder=args.img_folder,
        split='val'
    )
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_with_emotion,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_with_emotion
    )
    
    # Create model
    print("\nCreating model...")
    model = create_model(
        vocab_size=vocab_size,
        config=config,
        tfidf_path=args.tfidf_pkl
    )
    
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Memory estimate: {model.get_memory_footprint()}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        vocab=vocab,
        device=args.device,
        output_dir=args.output_dir
    )
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train!
    trainer.train()


if __name__ == '__main__':
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    
    main()
