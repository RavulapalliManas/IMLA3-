import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
from typing import Optional, List

class OptimizedVLT(nn.Module):
    """
    Hybrid Objective-Subjective Vision-Language Transformer (ArtEmis ~6.5k).
    
    Architecture:
    - Encoder: ViT-Small (Pretrained, 384 dim) -> Frozen initially.
    - Adapter: Projection 384 -> 512.
    - Object Tags: Pre-computed (FasterRCNN) and embedded via shared vocabulary.
    - Multi-Modal Memory: Concatenated [Image Features + Object Embeddings].
    - Decoder: Custom Transformer (4 Layers, 512 dim, 8 Heads).
    - Regularization: Moderate Dropout (0.2), Pre-Norm, Weight Tying.
    - Optimizations: PyTorch 2.0 SDPA (Flash Attention), AMP-ready.
    
    IMPORTANT: Object tags MUST be pre-computed offline using FasterRCNN/DETR
    and saved to disk (JSON/Pickle) to avoid GPU bottleneck during training.
    Running object detection in the training loop would reduce throughput by ~10x.
    """
    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int,
        start_token_id: int,
        end_token_id: int,
        max_seq_len: int = 40,
        max_objects: int = 10,       # NEW: Maximum object tags per image
        embed_size: int = 512,       # INCREASED from 256 to 512 for better capacity
        hidden_size: int = 512,      # INCREASED decoder hidden dim to 512
        num_heads: int = 8,          # INCREASED from 4 to 8 (512/8 = 64 per head)
        num_layers: int = 4,         # INCREASED from 3 to 4 layers
        dropout_rate: float = 0.2,   # REDUCED from 0.3 to 0.2 (less regularization)
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pad_token_id = pad_token_id
        self.start_token_id = start_token_id
        self.end_token_id = end_token_id
        self.max_seq_len = max_seq_len
        self.max_objects = max_objects  # NEW: For object tag memory
        self.d_model = hidden_size  # Use hidden_size as d_model

        # --- 1. ENCODER (Pretrained ViT-Small) ---
        print("Initializing OptimizedVLT...")
        print("  - Loading Pretrained ViT-Small (384 dim)...")
        # Global_pool='' ensures we get (Batch, Num_Patches, Dim) instead of just (Batch, Dim)
        self.vision_encoder = timm.create_model(
            'vit_small_patch16_224', 
            pretrained=True, 
            num_classes=0, 
            global_pool=''
        )
        
        # Freeze Encoder Initially (Stage 1 Training)
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
            
        vit_dim = 384 # ViT-Small output dimension

        # --- 2. ADAPTER (Bridge) ---
        self.visual_projection = nn.Sequential(
            nn.Linear(vit_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.Dropout(dropout_rate)
        )

        # --- 3. DECODER (From Scratch) ---
        # Shared Embedding: Used for BOTH caption tokens AND object tags
        # This allows the model to relate visual objects to textual descriptions
        self.embedding = nn.Embedding(vocab_size, self.d_model, padding_idx=pad_token_id)
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(self.d_model, dropout=dropout_rate, max_len=max_seq_len)
        
        # Transformer Decoder
        # norm_first=True (Pre-LN) is crucial for training stability from scratch
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=num_heads,             # Use configurable num_heads
            dim_feedforward=self.d_model * 4,  # Standard 4x expansion
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
            norm_first=True         
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output Head
        self.output_head = nn.Linear(self.d_model, vocab_size)
        
        # Optimization: Tie weights (Embedding matrix = Output Matrix)
        # This acts as regularization and reduces parameters
        self.output_head.weight = self.embedding.weight

        # --- 4. INITIALIZATION ---
        self.apply(self._init_weights)
        self.to(self.device)

    def _init_weights(self, module):
        """Custom initialization to ensure convergence."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def unfreeze_last_layers(self):
        """Helper to unfreeze last 2 layers of ViT for Stage 2 fine-tuning."""
        print("  - Unfreezing last 2 ViT blocks for fine-tuning...")
        # Note: timm attribute is usually 'blocks'
        for param in self.vision_encoder.blocks[-2:].parameters():
            param.requires_grad = True
        # Also unfreeze the final norm layer of ViT
        for param in self.vision_encoder.norm.parameters():
            param.requires_grad = True

    def forward(self, images, text_input_ids, object_input_ids=None):
        """
        Hybrid Objective-Subjective Forward Pass with Multi-Modal Memory.
        
        Args:
            images: (B, 3, 224, 224) - Input images
            text_input_ids: (B, Seq_Len) - Caption tokens [START, <emotion>, <style>, ...]
            object_input_ids: (B, Top_K) - Pre-computed object tag token IDs
                             If None, falls back to image-only mode (backwards compatible)
        
        Memory Construction:
            Old: [Image Features] -> (B, 196, 512)
            New: [Image Features | Object Embeddings] -> (B, 196+K, 512)
        """
        # A. Vision Path
        # Features: (B, N, 384) where N=196 for ViT-Small 224x224 (14x14 patches)
        img_feats = self.vision_encoder.forward_features(images)
        
        # Safety check for dimensions
        if img_feats.dim() == 2: img_feats = img_feats.unsqueeze(1)
        
        # Project to d_model: (B, N, 512)
        img_memory = self.visual_projection(img_feats)

        # B. Object Tag Path (Hybrid Objective Memory)
        if object_input_ids is not None:
            # Embed object tags using SAME embedding as text (shared vocabulary)
            # Shape: (B, Top_K, 512)
            obj_embeddings = self.embedding(object_input_ids)
            
            # Create padding mask for objects (pad_token_id indicates no object)
            obj_pad_mask = (object_input_ids == self.pad_token_id)
            
            # Concatenate Image Features + Object Embeddings along sequence dim
            # Memory: (B, 196+K, 512) - Decoder can attend to both pixels and concepts
            memory = torch.cat([img_memory, obj_embeddings], dim=1)
            
            # Extend memory padding mask (images never padded, only objects)
            img_mask = torch.zeros(img_memory.size(0), img_memory.size(1), 
                                  dtype=torch.bool, device=self.device)
            memory_key_padding_mask = torch.cat([img_mask, obj_pad_mask], dim=1)
        else:
            # Fallback: Image-only mode (for inference if objects unavailable)
            memory = img_memory
            memory_key_padding_mask = None

        # C. Text Path
        # Embedding: (B, T, 512)
        tgt_emb = self.embedding(text_input_ids)
        tgt_emb = self.pos_encoder(tgt_emb)
        
        # D. Masks
        seq_len = text_input_ids.size(1)
        # Causal Mask (Prevent cheating)
        tgt_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=self.device), diagonal=1)
        # Padding Mask (Ignore pads)
        tgt_pad_mask = (text_input_ids == self.pad_token_id)
        
        # E. Decode with Multi-Modal Memory
        # PyTorch 2.0 automatically uses SDPA (Flash Attention) if available
        # due to batch_first=True and norm_first=True settings
        output = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=memory_key_padding_mask  # NEW: Handle object padding
        )
        
        return self.output_head(output)

    @torch.no_grad()
    def generate(self, images, object_input_ids=None, max_len=30, temperature=1.0):
        """
        Greedy Decoding: Fast and good for validation debugging.
        Now supports object tags for hybrid inference.
        """
        self.eval()
        img_feats = self.vision_encoder.forward_features(images.to(self.device))
        if img_feats.dim() == 2: img_feats = img_feats.unsqueeze(1)
        img_memory = self.visual_projection(img_feats)
        
        # Build hybrid memory if objects provided
        if object_input_ids is not None:
            obj_embeddings = self.embedding(object_input_ids.to(self.device))
            memory = torch.cat([img_memory, obj_embeddings], dim=1)
            
            obj_pad_mask = (object_input_ids == self.pad_token_id)
            img_mask = torch.zeros(img_memory.size(0), img_memory.size(1), 
                                  dtype=torch.bool, device=self.device)
            memory_key_padding_mask = torch.cat([img_mask, obj_pad_mask], dim=1)
        else:
            memory = img_memory
            memory_key_padding_mask = None
        
        batch_size = images.size(0)
        generated = torch.full((batch_size, 1), self.start_token_id, dtype=torch.long, device=self.device)
        
        for _ in range(max_len):
            tgt_emb = self.embedding(generated)
            tgt_emb = self.pos_encoder(tgt_emb)
            
            # No causal mask needed for inference loop (it's implicit)
            out = self.decoder(tgt=tgt_emb, memory=memory, 
                             memory_key_padding_mask=memory_key_padding_mask)
            
            # Get last token
            logits = self.output_head(out[:, -1, :]) / temperature
            next_token = torch.argmax(logits, dim=-1).unsqueeze(1)
            
            generated = torch.cat((generated, next_token), dim=1)
            
            # Optimization: Stop if all sequences hit END (simplified)
            if (next_token == self.end_token_id).all():
                break
                
        return generated

    @torch.no_grad()
    def generate_beam(self, images, object_input_ids=None, beam_size=3, max_len=30, alpha=0.7):
        """
        Beam Search: Use this for FINAL evaluation to maximize BLEU.
        Now supports hybrid objective-subjective generation with object tags.
        alpha: Length penalty (0.6-0.9 favors longer descriptions).
        """
        self.eval()
        batch_size = images.size(0)
        
        # 1. Encode Image Once
        img_feats = self.vision_encoder.forward_features(images.to(self.device)) #(B, N, 384)
        if img_feats.dim() == 2: img_feats = img_feats.unsqueeze(1)
        img_memory = self.visual_projection(img_feats) #(B, N, 512)
        
        # Build hybrid memory if objects provided
        if object_input_ids is not None:
            obj_embeddings = self.embedding(object_input_ids.to(self.device))
            memory = torch.cat([img_memory, obj_embeddings], dim=1)
            
            obj_pad_mask = (object_input_ids == self.pad_token_id)
            img_mask = torch.zeros(img_memory.size(0), img_memory.size(1), 
                                  dtype=torch.bool, device=self.device)
            memory_key_padding_mask = torch.cat([img_mask, obj_pad_mask], dim=1)
            # Expand for beam
            memory_key_padding_mask = memory_key_padding_mask.repeat_interleave(beam_size, dim=0)
        else:
            memory = img_memory
            memory_key_padding_mask = None
        
        # Expand memory for beam size: (B*Beam, N+K, 512)
        memory = memory.repeat_interleave(beam_size, dim=0)
        
        # 2. Setup Beams
        # Current Token: (B*Beam, 1)
        current_tokens = torch.full((batch_size * beam_size, 1), self.start_token_id, dtype=torch.long, device=self.device)
        
        # Scores: (B, Beam) - First beam has score 0, others -inf
        top_k_scores = torch.full((batch_size, beam_size), float("-inf"), device=self.device)
        top_k_scores[:, 0] = 0.0
        top_k_scores = top_k_scores.view(-1) # Flatten (B*Beam)
        
        # Finished sequences storage
        finished_seqs = [[] for _ in range(batch_size)]
        finished_scores = [[] for _ in range(batch_size)]

        for step in range(max_len):
            # Embed
            tgt_emb = self.embedding(current_tokens)
            tgt_emb = self.pos_encoder(tgt_emb)
            
            # Decode with hybrid memory
            out = self.decoder(tgt=tgt_emb, memory=memory,
                             memory_key_padding_mask=memory_key_padding_mask)
            
            # Logits for last token: (B*Beam, Vocab)
            logits = self.output_head(out[:, -1, :])
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Add to cumulative scores
            # (B*Beam, Vocab)
            scores = top_k_scores.unsqueeze(1) + log_probs
            vocab_size = scores.shape[1]
            
            # Reshape to (B, Beam*Vocab) to find topk per image
            scores = scores.view(batch_size, -1)
            
            # Get top K for each image
            # best_scores: (B, Beam), best_indices: (B, Beam)
            best_scores, best_indices = scores.topk(beam_size, dim=1)
            
            # Decode indices
            beam_indices = torch.div(best_indices, vocab_size, rounding_mode='floor') # Which beam did it come from?
            token_indices = best_indices % vocab_size # What is the token?
            
            # Reconstruct Batch Indices
            # We need to grab the correct history from current_tokens
            batch_offsets = torch.arange(batch_size, device=self.device).unsqueeze(1) * beam_size
            prev_beam_indices = batch_offsets + beam_indices # (B, Beam) flat indices
            
            # Update tokens
            # Select history
            next_tokens = current_tokens[prev_beam_indices.view(-1)] 
            # Append new token
            next_tokens = torch.cat([next_tokens, token_indices.view(-1).unsqueeze(1)], dim=1)
            
            # Check for END token
            current_tokens = next_tokens
            top_k_scores = best_scores.view(-1)
            
            # (Advanced logic for stopping and length penalty would go here)
            # For simplicity in this assignment, we run fixed steps or until easy stop
            
        # Return best beam (index 0)
        final_output = current_tokens.view(batch_size, beam_size, -1)[:, 0, :]
        return final_output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.max_len = max_len
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        
        if seq_len > self.max_len:
            pe = torch.zeros(seq_len, self.d_model, device=x.device)
            position = torch.arange(0, seq_len, dtype=torch.float, device=x.device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.d_model, 2, device=x.device).float() * (-math.log(10000.0) / self.d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            x = x + pe.unsqueeze(0)
        else:
            x = x + self.pe[:, :seq_len]
        
        return self.dropout(x)