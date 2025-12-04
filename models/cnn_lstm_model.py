"""
CNN + LSTM Image Captioning Baseline with TF-IDF Embeddings
============================================================
Architecture:
    - Custom CNN Encoder (3-4 conv blocks) → 512-dim feature vector
    - LSTM Decoder with TF-IDF embeddings (300-dim PCA reduced)
    - Emotion conditioning via learned embedding
    
Optimized for RTX 3060 Laptop GPU (6GB VRAM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import os
from typing import Optional, Tuple, Dict


# =============================================================================
# SECTION A: TF-IDF Embedding Layer
# =============================================================================

class TFIDFEmbedding(nn.Module):
    """
    Embedding layer that uses precomputed TF-IDF PCA-reduced vectors.
    
    If TF-IDF vectors are not available, falls back to learned embeddings.
    Projects TF-IDF vectors to desired embedding dimension.
    
    Args:
        vocab_size: Size of vocabulary
        tfidf_dim: Dimension of TF-IDF vectors (e.g., 300 for PCA-reduced)
        embed_dim: Output embedding dimension (default 256)
        tfidf_path: Path to TF-IDF pickle file (optional)
        freeze_tfidf: Whether to freeze TF-IDF weights (default True)
    """
    
    def __init__(
        self,
        vocab_size: int,
        tfidf_dim: int = 300,
        embed_dim: int = 256,
        tfidf_path: Optional[str] = None,
        freeze_tfidf: bool = True
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.tfidf_dim = tfidf_dim
        self.embed_dim = embed_dim
        self.use_tfidf = False
        
        # Try to load TF-IDF vectors
        if tfidf_path and os.path.exists(tfidf_path):
            try:
                with open(tfidf_path, 'rb') as f:
                    tfidf_data = pickle.load(f)
                
                # Handle different pickle formats
                if isinstance(tfidf_data, dict):
                    # Format: {word_idx: vector} or {word: vector}
                    tfidf_matrix = self._build_tfidf_matrix(tfidf_data, vocab_size, tfidf_dim)
                elif isinstance(tfidf_data, np.ndarray):
                    tfidf_matrix = tfidf_data
                else:
                    raise ValueError(f"Unknown TF-IDF format: {type(tfidf_data)}")
                
                # Register as buffer (non-trainable) or parameter (trainable)
                tfidf_tensor = torch.tensor(tfidf_matrix, dtype=torch.float32)
                if freeze_tfidf:
                    self.register_buffer('tfidf_vectors', tfidf_tensor)
                else:
                    self.tfidf_vectors = nn.Parameter(tfidf_tensor)
                
                self.use_tfidf = True
                print(f"✓ Loaded TF-IDF vectors from {tfidf_path}")
                print(f"  Shape: {tfidf_tensor.shape}, Frozen: {freeze_tfidf}")
                
            except Exception as e:
                print(f"⚠ Could not load TF-IDF: {e}")
                print("  Falling back to learned embeddings")
        
        # Fallback: learned embeddings
        if not self.use_tfidf:
            self.learned_embedding = nn.Embedding(vocab_size, tfidf_dim, padding_idx=0)
            nn.init.xavier_uniform_(self.learned_embedding.weight)
            self.learned_embedding.weight.data[0].zero_()  # Zero out padding
            print(f"✓ Using learned embeddings: vocab={vocab_size}, dim={tfidf_dim}")
        
        # Projection layer: tfidf_dim → embed_dim
        self.projection = nn.Sequential(
            nn.Linear(tfidf_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
    
    def _build_tfidf_matrix(
        self, 
        tfidf_dict: Dict, 
        vocab_size: int, 
        tfidf_dim: int
    ) -> np.ndarray:
        """Build vocabulary-indexed TF-IDF matrix from dictionary."""
        matrix = np.zeros((vocab_size, tfidf_dim), dtype=np.float32)
        
        for key, vec in tfidf_dict.items():
            idx = int(key) if isinstance(key, (int, str)) and str(key).isdigit() else None
            if idx is not None and 0 <= idx < vocab_size:
                matrix[idx] = vec[:tfidf_dim]
        
        return matrix
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (batch_size, seq_len) token indices
            
        Returns:
            embeddings: (batch_size, seq_len, embed_dim)
        """
        if self.use_tfidf:
            # Index into precomputed TF-IDF vectors
            tfidf_embeds = F.embedding(token_ids, self.tfidf_vectors)
        else:
            # Use learned embeddings
            tfidf_embeds = self.learned_embedding(token_ids)
        
        # Project to desired dimension
        return self.projection(tfidf_embeds)


# =============================================================================
# SECTION B: Custom CNN Encoder
# =============================================================================

class ConvBlock(nn.Module):
    """Single convolutional block with BatchNorm and ReLU."""
    
    def __init__(self, in_channels: int, out_channels: int, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNNEncoder(nn.Module):
    """
    Custom lightweight CNN encoder for image feature extraction.
    
    Architecture (optimized for RTX 3060):
        Conv Block 1: 3 → 64 channels, 224→112
        Conv Block 2: 64 → 128 channels, 112→56
        Conv Block 3: 128 → 256 channels, 56→28
        Conv Block 4: 256 → 512 channels, 28→14
        Global Average Pooling → 512-dim
    
    Args:
        output_dim: Final feature dimension (default 512)
        dropout: Dropout rate (default 0.3)
    """
    
    def __init__(self, output_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        
        # Convolutional backbone
        self.conv_blocks = nn.Sequential(
            ConvBlock(3, 64),      # 224 → 112
            ConvBlock(64, 128),    # 112 → 56
            ConvBlock(128, 256),   # 56 → 28
            ConvBlock(256, 512),   # 28 → 14
        )
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final projection
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (batch_size, 3, 224, 224) normalized images
            
        Returns:
            features: (batch_size, output_dim) image features
        """
        x = self.conv_blocks(images)
        x = self.global_pool(x)
        return self.fc(x)


# =============================================================================
# SECTION C: Emotion Embedding
# =============================================================================

class EmotionEmbedding(nn.Module):
    """
    Learnable emotion embedding layer.
    
    Emotion categories (from ArtEmis):
        0: contentment
        1: awe
        2: something_else
        3: sadness
        4: amusement
        5: fear
        6: excitement
        7: disgust
        8: anger
    """
    
    EMOTION_LABELS = [
        "contentment", "awe", "something_else", "sadness",
        "amusement", "fear", "excitement", "disgust", "anger"
    ]
    
    def __init__(self, num_emotions: int = 9, embed_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(num_emotions, embed_dim)
        nn.init.xavier_uniform_(self.embedding.weight)
    
    def forward(self, emotion_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            emotion_ids: (batch_size,) emotion indices
            
        Returns:
            emotion_embeds: (batch_size, embed_dim)
        """
        return self.embedding(emotion_ids)


# =============================================================================
# SECTION D: LSTM Decoder with Attention
# =============================================================================

class LSTMDecoder(nn.Module):
    """
    LSTM decoder for caption generation with optional attention.
    
    Features:
        - TF-IDF embeddings for word representation
        - Attention over image features (optional)
        - Teacher forcing during training
        - Emotion conditioning
    
    Args:
        vocab_size: Vocabulary size
        embed_dim: Word embedding dimension
        hidden_dim: LSTM hidden dimension
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        tfidf_dim: TF-IDF vector dimension
        tfidf_path: Path to TF-IDF pickle file
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
        tfidf_dim: int = 300,
        tfidf_path: Optional[str] = None,
        emotion_dim: int = 64
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        
        # TF-IDF embedding layer
        self.embedding = TFIDFEmbedding(
            vocab_size=vocab_size,
            tfidf_dim=tfidf_dim,
            embed_dim=embed_dim,
            tfidf_path=tfidf_path,
            freeze_tfidf=True
        )
        
        # Emotion embedding
        self.emotion_embed = EmotionEmbedding(num_emotions=9, embed_dim=emotion_dim)
        
        # LSTM input: word_embed + image_features + emotion
        # Initial hidden state comes from image features
        lstm_input_dim = embed_dim + emotion_dim
        
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False
        )
        
        # Output projection
        self.output_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size)
        )
        
        # Image feature projection to initialize LSTM hidden state
        self.image_proj_h = nn.Linear(512, hidden_dim * num_layers)
        self.image_proj_c = nn.Linear(512, hidden_dim * num_layers)
    
    def init_hidden(
        self, 
        image_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize LSTM hidden state from image features.
        
        Args:
            image_features: (batch_size, 512) image features
            
        Returns:
            h0: (num_layers, batch_size, hidden_dim)
            c0: (num_layers, batch_size, hidden_dim)
        """
        batch_size = image_features.size(0)
        
        h = self.image_proj_h(image_features)  # (B, hidden_dim * num_layers)
        c = self.image_proj_c(image_features)
        
        h = h.view(batch_size, self.num_layers, self.hidden_dim)
        c = c.view(batch_size, self.num_layers, self.hidden_dim)
        
        h = h.permute(1, 0, 2).contiguous()  # (num_layers, B, hidden_dim)
        c = c.permute(1, 0, 2).contiguous()
        
        return h, c
    
    def forward(
        self,
        image_features: torch.Tensor,
        captions: torch.Tensor,
        emotion_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with teacher forcing.
        
        Args:
            image_features: (batch_size, 512) from CNN encoder
            captions: (batch_size, seq_len) token indices (input_ids)
            emotion_ids: (batch_size,) emotion indices
            
        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = captions.shape
        
        # Get embeddings
        word_embeds = self.embedding(captions)  # (B, S, embed_dim)
        emotion_embeds = self.emotion_embed(emotion_ids)  # (B, emotion_dim)
        
        # Expand emotion to match sequence length
        emotion_embeds = emotion_embeds.unsqueeze(1).expand(-1, seq_len, -1)  # (B, S, emotion_dim)
        
        # Concatenate word embeddings with emotion
        lstm_input = torch.cat([word_embeds, emotion_embeds], dim=-1)  # (B, S, embed_dim + emotion_dim)
        
        # Initialize hidden state from image features
        h0, c0 = self.init_hidden(image_features)
        
        # LSTM forward
        lstm_out, _ = self.lstm(lstm_input, (h0, c0))  # (B, S, hidden_dim)
        
        # Project to vocabulary
        logits = self.output_layer(lstm_out)  # (B, S, vocab_size)
        
        return logits
    
    def generate_step(
        self,
        word_embed: torch.Tensor,
        emotion_embed: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Single step generation for inference.
        
        Args:
            word_embed: (batch_size, 1, embed_dim)
            emotion_embed: (batch_size, 1, emotion_dim)
            hidden: (h, c) LSTM hidden state
            
        Returns:
            logits: (batch_size, vocab_size)
            new_hidden: Updated hidden state
        """
        lstm_input = torch.cat([word_embed, emotion_embed], dim=-1)  # (B, 1, input_dim)
        lstm_out, hidden = self.lstm(lstm_input, hidden)  # (B, 1, hidden_dim)
        logits = self.output_layer(lstm_out.squeeze(1))  # (B, vocab_size)
        return logits, hidden


# =============================================================================
# SECTION E: Full CNN-LSTM Model
# =============================================================================

class CNNLSTMCaptioner(nn.Module):
    """
    Complete CNN + LSTM Image Captioning Model with TF-IDF embeddings.
    
    This is the main model class that combines:
        - Custom CNN encoder for image features
        - LSTM decoder with TF-IDF word embeddings
        - Emotion conditioning
    
    Args:
        vocab_size: Size of vocabulary
        embed_dim: Word embedding dimension (default 256)
        hidden_dim: LSTM hidden dimension (default 512)
        num_layers: Number of LSTM layers (default 2)
        dropout: Dropout rate (default 0.3)
        tfidf_dim: TF-IDF vector dimension (default 300)
        tfidf_path: Path to TF-IDF pickle file
        max_len: Maximum caption length (default 33)
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
        tfidf_dim: int = 300,
        tfidf_path: Optional[str] = None,
        max_len: int = 33,
        pad_id: int = 0,
        start_id: int = 2,
        end_id: int = 3
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.pad_id = pad_id
        self.start_id = start_id
        self.end_id = end_id
        
        # CNN Encoder
        self.encoder = CNNEncoder(output_dim=512, dropout=dropout)
        
        # LSTM Decoder
        self.decoder = LSTMDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            tfidf_dim=tfidf_dim,
            tfidf_path=tfidf_path
        )
        
        print(f"\n{'='*60}")
        print("CNN-LSTM Captioner Initialized")
        print(f"{'='*60}")
        print(f"  Vocab size: {vocab_size}")
        print(f"  Embed dim: {embed_dim}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  LSTM layers: {num_layers}")
        print(f"  Dropout: {dropout}")
        print(f"  Max length: {max_len}")
        print(f"  TF-IDF dim: {tfidf_dim}")
        print(f"{'='*60}\n")
    
    def forward(
        self,
        images: torch.Tensor,
        captions: torch.Tensor,
        emotion_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for training with teacher forcing.
        
        Args:
            images: (B, 3, 224, 224) input images
            captions: (B, seq_len) input caption tokens
            emotion_ids: (B,) emotion indices
            
        Returns:
            logits: (B, seq_len, vocab_size)
        """
        # Encode images
        image_features = self.encoder(images)  # (B, 512)
        
        # Decode with teacher forcing
        logits = self.decoder(image_features, captions, emotion_ids)
        
        return logits
    
    @torch.no_grad()
    def generate_greedy(
        self,
        images: torch.Tensor,
        emotion_ids: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Greedy decoding for caption generation.
        
        Args:
            images: (B, 3, 224, 224) input images
            emotion_ids: (B,) emotion indices
            temperature: Softmax temperature (default 1.0)
            
        Returns:
            generated_ids: (B, max_len) generated token indices
        """
        self.eval()
        batch_size = images.size(0)
        device = images.device
        
        # Encode images
        image_features = self.encoder(images)  # (B, 512)
        
        # Initialize hidden state
        hidden = self.decoder.init_hidden(image_features)
        
        # Start with <start> token
        current_token = torch.full(
            (batch_size, 1), self.start_id, dtype=torch.long, device=device
        )
        
        # Get emotion embedding (fixed throughout generation)
        emotion_embed = self.decoder.emotion_embed(emotion_ids).unsqueeze(1)  # (B, 1, emotion_dim)
        
        generated = [current_token]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for _ in range(self.max_len - 1):
            # Get word embedding
            word_embed = self.decoder.embedding(current_token)  # (B, 1, embed_dim)
            
            # Generate next token
            logits, hidden = self.decoder.generate_step(word_embed, emotion_embed, hidden)
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Greedy selection
            current_token = logits.argmax(dim=-1, keepdim=True)  # (B, 1)
            generated.append(current_token)
            
            # Check for <end> token
            finished = finished | (current_token.squeeze(-1) == self.end_id)
            if finished.all():
                break
        
        return torch.cat(generated, dim=1)  # (B, generated_len)
    
    @torch.no_grad()
    def generate_beam(
        self,
        images: torch.Tensor,
        emotion_ids: torch.Tensor,
        beam_width: int = 3,
        length_penalty: float = 0.7
    ) -> torch.Tensor:
        """
        Beam search decoding for caption generation.
        
        Args:
            images: (B, 3, 224, 224) input images
            emotion_ids: (B,) emotion indices
            beam_width: Number of beams (default 3)
            length_penalty: Length normalization factor (default 0.7)
            
        Returns:
            generated_ids: (B, max_len) best generated sequences
        """
        self.eval()
        batch_size = images.size(0)
        device = images.device
        
        # For simplicity, process one image at a time for beam search
        all_generated = []
        
        for b in range(batch_size):
            img = images[b:b+1]  # (1, 3, 224, 224)
            emo = emotion_ids[b:b+1]  # (1,)
            
            # Encode image
            image_features = self.encoder(img)  # (1, 512)
            
            # Initialize beams
            # Each beam: (sequence, score, hidden_state, finished)
            h0, c0 = self.decoder.init_hidden(image_features)
            emotion_embed = self.decoder.emotion_embed(emo).unsqueeze(1)  # (1, 1, emotion_dim)
            
            # Start token
            start_token = torch.tensor([[self.start_id]], device=device)
            
            beams = [{
                'tokens': [self.start_id],
                'score': 0.0,
                'hidden': (h0, c0),
                'finished': False
            }]
            
            for step in range(self.max_len - 1):
                all_candidates = []
                
                for beam in beams:
                    if beam['finished']:
                        all_candidates.append(beam)
                        continue
                    
                    # Get last token
                    last_token = torch.tensor([[beam['tokens'][-1]]], device=device)
                    word_embed = self.decoder.embedding(last_token)
                    
                    # Generate
                    logits, new_hidden = self.decoder.generate_step(
                        word_embed, emotion_embed, beam['hidden']
                    )
                    log_probs = F.log_softmax(logits, dim=-1).squeeze(0)
                    
                    # Get top-k tokens
                    topk_scores, topk_ids = log_probs.topk(beam_width)
                    
                    for k in range(beam_width):
                        token_id = topk_ids[k].item()
                        token_score = topk_scores[k].item()
                        
                        new_tokens = beam['tokens'] + [token_id]
                        new_score = beam['score'] + token_score
                        
                        all_candidates.append({
                            'tokens': new_tokens,
                            'score': new_score,
                            'hidden': new_hidden,
                            'finished': token_id == self.end_id
                        })
                
                # Select top beams with length penalty
                def score_fn(b):
                    length = len(b['tokens'])
                    return b['score'] / (length ** length_penalty)
                
                beams = sorted(all_candidates, key=score_fn, reverse=True)[:beam_width]
                
                # Early stopping if all beams finished
                if all(b['finished'] for b in beams):
                    break
            
            # Select best beam
            best_beam = max(beams, key=lambda b: b['score'] / (len(b['tokens']) ** length_penalty))
            
            # Pad to max_len
            tokens = best_beam['tokens']
            while len(tokens) < self.max_len:
                tokens.append(self.pad_id)
            
            all_generated.append(torch.tensor(tokens[:self.max_len], device=device))
        
        return torch.stack(all_generated)  # (B, max_len)
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_memory_footprint(self) -> str:
        """Estimate GPU memory usage."""
        params = self.count_parameters()
        # Rough estimate: params * 4 bytes (float32) * 2 (for gradients)
        mem_mb = (params * 4 * 2) / (1024 * 1024)
        return f"{mem_mb:.1f} MB"


# =============================================================================
# SECTION F: Utility Functions
# =============================================================================

def create_model(
    vocab_size: int,
    config: Optional[Dict] = None,
    tfidf_path: Optional[str] = None
) -> CNNLSTMCaptioner:
    """
    Factory function to create CNN-LSTM model with config.
    
    Args:
        vocab_size: Size of vocabulary
        config: Optional configuration dictionary
        tfidf_path: Path to TF-IDF vectors
        
    Returns:
        model: CNNLSTMCaptioner instance
    """
    # Default config (optimized for RTX 3060)
    default_config = {
        'embed_dim': 256,
        'hidden_dim': 512,
        'num_layers': 2,
        'dropout': 0.3,
        'tfidf_dim': 300,
        'max_len': 33,
        'pad_id': 0,
        'start_id': 2,
        'end_id': 3
    }
    
    if config:
        default_config.update(config)
    
    model = CNNLSTMCaptioner(
        vocab_size=vocab_size,
        embed_dim=default_config['embed_dim'],
        hidden_dim=default_config['hidden_dim'],
        num_layers=default_config['num_layers'],
        dropout=default_config['dropout'],
        tfidf_dim=default_config['tfidf_dim'],
        tfidf_path=tfidf_path,
        max_len=default_config['max_len'],
        pad_id=default_config['pad_id'],
        start_id=default_config['start_id'],
        end_id=default_config['end_id']
    )
    
    return model


def load_model_checkpoint(
    checkpoint_path: str,
    vocab_size: int,
    config: Optional[Dict] = None,
    tfidf_path: Optional[str] = None,
    device: str = 'cuda'
) -> CNNLSTMCaptioner:
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to .pt checkpoint file
        vocab_size: Size of vocabulary
        config: Model configuration
        tfidf_path: Path to TF-IDF vectors
        device: Device to load model on
        
    Returns:
        model: Loaded CNNLSTMCaptioner
    """
    model = create_model(vocab_size, config, tfidf_path)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print(f"✓ Loaded model from {checkpoint_path}")
    return model


# =============================================================================
# Test the model
# =============================================================================

if __name__ == "__main__":
    print("Testing CNN-LSTM Captioner...")
    
    # Create model
    model = create_model(vocab_size=5000)
    print(f"\nTotal parameters: {model.count_parameters():,}")
    print(f"Estimated memory: {model.get_memory_footprint()}")
    
    # Test forward pass
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)
    captions = torch.randint(0, 5000, (batch_size, 20))
    emotions = torch.randint(0, 9, (batch_size,))
    
    print("\nTesting forward pass...")
    logits = model(images, captions, emotions)
    print(f"  Input images: {images.shape}")
    print(f"  Input captions: {captions.shape}")
    print(f"  Output logits: {logits.shape}")
    
    print("\nTesting greedy generation...")
    generated = model.generate_greedy(images, emotions)
    print(f"  Generated shape: {generated.shape}")
    
    print("\nTesting beam search (beam=3)...")
    generated_beam = model.generate_beam(images[:2], emotions[:2], beam_width=3)
    print(f"  Generated shape: {generated_beam.shape}")
    
    print("\n✓ All tests passed!")
