"""
Comprehensive Attention Visualization for Hybrid VLT
=====================================================

Visualizes cross-attention between generated tokens and:
- Image patches (196 = 14x14 grid from ViT)
- Object tag slots (10 slots)

Creates publication-ready heatmaps showing:
1. Full attention matrix (tokens → patches + objects)
2. Per-token attention overlays on image
3. Object attention analysis
"""

import os
import json
import pickle
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from PIL import Image
from torchvision import transforms
import cv2

# Use relative imports when running as part of the package
try:
    from .model_transformer import OptimizedVLT
    from .collate import load_object_tags_cache, set_vocab_cache, get_object_ids_for_image
except ImportError:
    # Fallback for running script directly
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model_transformer import OptimizedVLT
    from collate import load_object_tags_cache, set_vocab_cache, get_object_ids_for_image


# Global storage for attention weights
attention_weights_storage = {}

def get_cross_attention_hook(name):
    """Hook to capture cross-attention weights from decoder layers."""
    def hook(module, input, output):
        # MultiheadAttention returns (attn_output, attn_weights)
        # attn_weights shape: (batch, num_heads, tgt_len, src_len)
        # Note: attn_weights may be None if not computed
        if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
            attention_weights_storage[name] = output[1].detach().cpu()
        elif isinstance(output, tuple) and len(output) >= 1:
            # If attention weights aren't returned, we still got the output
            # Store empty marker so we know the hook was called
            attention_weights_storage[name] = None
    return hook


def enable_attention_capture(model):
    """
    Register hooks on all multihead attention modules in decoder.
    Directly hooks the functional call to capture attention weights.
    """
    hooks = []
    
    for name, module in model.named_modules():
        if 'decoder.layers' in name and 'self_attn' in name:
            # Skip self-attention, we only want cross-attention
            continue
        
        if isinstance(module, torch.nn.MultiheadAttention):
            original_forward = module.forward
            module_name = name
            
            # Create wrapper that captures attention
            def make_wrapper(orig_forward, name_str):
                def forward_wrapper(query, key, value, key_padding_mask=None, 
                                  need_weights=False, attn_mask=None, 
                                  average_attn_weights=True, is_causal=False):
                    # Call original with need_weights=True to get attention
                    attn_output, attn_weights = orig_forward(
                        query, key, value,
                        key_padding_mask=key_padding_mask,
                        need_weights=True,
                        attn_mask=attn_mask,
                        average_attn_weights=False,
                        is_causal=is_causal
                    )
                    # Store attention weights
                    attention_weights_storage[name_str] = attn_weights.detach().cpu()
                    return attn_output, attn_weights
                return forward_wrapper
            
            module.forward = make_wrapper(original_forward, module_name)
            hooks.append(module)
    
    return hooks


def visualize_cross_attention_heatmap(
    model,
    image_path,
    vocab,
    object_tags_dict,
    device,
    save_dir,
    max_objects=10,
    layer_idx=-1
):
    """
    Create cross-attention heatmap: tokens → (patches + object_slots).
    
    Args:
        model: Trained OptimizedVLT model
        image_path: Path to input image
        vocab: Vocabulary dictionary
        object_tags_dict: Pre-computed object tags
        device: torch device
        save_dir: Directory to save visualizations
        max_objects: Number of object slots
        layer_idx: Which decoder layer to visualize (-1 = last)
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    # Enable attention weight capture
    enable_attention_capture(model)
    
    # First, discover what attention modules we have
    print("   Inspecting model for attention modules...")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.MultiheadAttention):
            print(f"      Found: {name}")
    
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get object tags for this image
    filename = os.path.basename(image_path)
    objects = object_tags_dict.get(filename, [])
    object_ids = get_object_ids_for_image(filename, max_objects).unsqueeze(0).to(device)
    
    print(f"📷 Image: {filename}")
    print(f"🏷️  Objects: {objects}")
    
    # The attention capture is already enabled via enable_attention_capture()
    # We'll look for cross_attn layers in the decoder
    
    # Generate caption to get attention weights
    with torch.no_grad():
        # First, encode image and build memory
        img_feats = model.vision_encoder.forward_features(image_tensor)
        if img_feats.dim() == 2:
            img_feats = img_feats.unsqueeze(1)
        memory = model.visual_projection(img_feats)  # (1, 196, 512)
        
        # Add object embeddings
        obj_emb = model.embedding(object_ids)  # (1, 10, 512)
        full_memory = torch.cat([memory, obj_emb], dim=1)  # (1, 206, 512)
        
        # Create memory mask
        obj_pad_mask = (object_ids == model.pad_token_id)
        img_mask = torch.zeros(1, memory.size(1), dtype=torch.bool, device=device)
        memory_mask = torch.cat([img_mask, obj_pad_mask], dim=1)
        
        # Generate with attention capture
        generated = [vocab['start_id']]
        all_attentions = []
        first_step = True
        
        for step in range(40):  # Max 40 tokens
            tgt_input = torch.LongTensor([generated]).to(device)
            tgt_emb = model.embedding(tgt_input)
            tgt_emb = model.pos_encoder(tgt_emb)
            
            # Clear previous attention
            attention_weights_storage.clear()
            
            # Decode
            output = model.decoder(
                tgt=tgt_emb, 
                memory=full_memory,
                memory_key_padding_mask=memory_mask
            )
            
            # Debug on first step
            if first_step:
                print(f"   Debug - Storage keys after decoder: {list(attention_weights_storage.keys())[:3]}")
                first_step = False
            
            # Capture attention for this step - find the cross-attention layer
            # Look for decoder layer multihead attention in storage
            cross_attn = None
            
            # Try the specific layer index
            if layer_idx >= 0:
                cross_attn_key = f'decoder.layers.{layer_idx}.multihead_attn'
            else:
                # For negative indexing, find the actual layer count
                decoder_layers = [k for k in attention_weights_storage.keys() 
                                 if 'decoder.layers' in k and 'multihead_attn' in k]
                if decoder_layers:
                    # Get the last layer
                    decoder_layers.sort(key=lambda x: int(x.split('.')[2]))
                    cross_attn_key = decoder_layers[layer_idx]
                else:
                    cross_attn_key = None
            
            if cross_attn_key and cross_attn_key in attention_weights_storage:
                # Shape: (batch, num_heads, tgt_len, src_len)
                attn = attention_weights_storage[cross_attn_key]
                if attn is not None:
                    # Average over heads, take last token's attention
                    avg_attn = attn.mean(dim=1)[0, -1, :]  # (src_len,)
                    all_attentions.append(avg_attn.numpy())
            
            # Get next token
            logits = model.output_head(output[:, -1, :])
            next_token = logits.argmax(dim=-1).item()
            generated.append(next_token)
            
            if next_token == vocab['end_id']:
                break
    
    # Decode caption
    caption_tokens = []
    for tid in generated:
        if tid == vocab['start_id']:
            caption_tokens.append('<START>')
        elif tid == vocab['end_id']:
            caption_tokens.append('<END>')
            break
        elif tid == vocab['pad_id']:
            continue
        else:
            caption_tokens.append(vocab['itos'][tid])
    
    print(f"📝 Generated: {' '.join(caption_tokens[1:-1])}")
    
    # Convert to numpy array
    attention_matrix = np.array(all_attentions)  # (num_tokens, 206)
    
    # Check if we have attention data
    if attention_matrix.size == 0:
        print(f"⚠️  Warning: No attention weights captured. Storage keys: {list(attention_weights_storage.keys())}")
        print(f"   Skipping visualization for this image.")
        return None, caption_tokens, objects
    
    # =========================================
    # PLOT 1: Full Cross-Attention Heatmap
    # =========================================
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Create custom colormap
    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    
    # Plot heatmap
    im = ax.imshow(attention_matrix, aspect='auto', cmap=cmap)
    
    # Add vertical line separating patches from objects
    ax.axvline(x=195.5, color='blue', linewidth=2, linestyle='--', label='Image | Objects')
    
    # Labels
    ax.set_xlabel('Memory Position (196 Patches + 10 Object Slots)', fontsize=12)
    ax.set_ylabel('Generated Tokens', fontsize=12)
    ax.set_title(f'Cross-Attention Heatmap\n{filename}', fontsize=14)
    
    # Y-axis: token labels
    ax.set_yticks(range(len(caption_tokens)))
    ax.set_yticklabels(caption_tokens, fontsize=9)
    
    # X-axis: mark key positions
    x_positions = [0, 98, 195, 196, 205]
    x_labels = ['Patch 0', 'Patch 98', 'Patch 195', 'Obj 0', 'Obj 9']
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=9)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight', fontsize=11)
    
    # Add region annotations
    ax.annotate('Image Patches\n(14×14 = 196)', xy=(98, -0.5), 
                xytext=(98, -2), fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray'))
    ax.annotate('Object\nSlots', xy=(201, -0.5), 
                xytext=(201, -2), fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='blue'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cross_attention_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: cross_attention_heatmap.png")
    
    # =========================================
    # PLOT 2: Per-Token Attention on Image
    # =========================================
    # Unnormalize image for display
    disp_img = image_tensor[0].cpu().permute(1, 2, 0).numpy()
    disp_img = (disp_img * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
    disp_img = np.clip(disp_img, 0, 1)
    
    # Select interesting tokens (skip <START>, emotions, styles)
    interesting_tokens = []
    for i, tok in enumerate(caption_tokens):
        if not tok.startswith('<') and not tok.startswith('['):
            interesting_tokens.append((i, tok))
    
    num_show = min(10, len(interesting_tokens))
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for idx, (token_idx, token_text) in enumerate(interesting_tokens[:num_show]):
        ax = axes[idx]
        
        # Get attention for this token (just the patch part: 0-195)
        patch_attn = attention_matrix[token_idx, :196]
        
        # Reshape to 14x14 and resize to image size
        attn_map = patch_attn.reshape(14, 14)
        attn_map = cv2.resize(attn_map, (224, 224))
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        
        # Display
        ax.imshow(disp_img)
        ax.imshow(attn_map, cmap='jet', alpha=0.5)
        ax.set_title(f'"{token_text}"', fontsize=11)
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(num_show, 10):
        axes[idx].axis('off')
    
    plt.suptitle(f'Per-Token Attention Maps\n{filename}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'per_token_attention.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: per_token_attention.png")
    
    # =========================================
    # PLOT 3: Object Slot Attention Analysis
    # =========================================
    # Extract object attention (columns 196-205)
    object_attention = attention_matrix[:, 196:]  # (num_tokens, 10)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Heatmap of object attention
    object_labels = objects + ['<PAD>'] * (max_objects - len(objects))
    
    im1 = ax1.imshow(object_attention, aspect='auto', cmap='Blues')
    ax1.set_xlabel('Object Slots', fontsize=11)
    ax1.set_ylabel('Generated Tokens', fontsize=11)
    ax1.set_title('Attention to Object Tags', fontsize=12)
    ax1.set_xticks(range(len(object_labels)))
    ax1.set_xticklabels(object_labels, rotation=45, ha='right', fontsize=9)
    ax1.set_yticks(range(len(caption_tokens)))
    ax1.set_yticklabels(caption_tokens, fontsize=9)
    plt.colorbar(im1, ax=ax1, label='Attention')
    
    # Right: Bar chart of total attention per object
    total_obj_attn = object_attention.sum(axis=0)
    colors = ['steelblue' if i < len(objects) else 'lightgray' for i in range(max_objects)]
    ax2.bar(range(max_objects), total_obj_attn, color=colors)
    ax2.set_xlabel('Object Slot', fontsize=11)
    ax2.set_ylabel('Total Attention (summed over tokens)', fontsize=11)
    ax2.set_title('Object Tag Importance', fontsize=12)
    ax2.set_xticks(range(max_objects))
    ax2.set_xticklabels(object_labels, rotation=45, ha='right', fontsize=9)
    
    # Add legend
    real_patch = mpatches.Patch(color='steelblue', label='Detected Objects')
    pad_patch = mpatches.Patch(color='lightgray', label='Padding')
    ax2.legend(handles=[real_patch, pad_patch])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'object_attention_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: object_attention_analysis.png")
    
    # =========================================
    # PLOT 4: Attention Distribution (Image vs Objects)
    # =========================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate per-token split of attention
    image_attn_per_token = attention_matrix[:, :196].sum(axis=1)
    object_attn_per_token = attention_matrix[:, 196:].sum(axis=1)
    
    x = range(len(caption_tokens))
    width = 0.35
    
    bars1 = ax.bar([i - width/2 for i in x], image_attn_per_token, width, 
                   label='Image Patches', color='coral')
    bars2 = ax.bar([i + width/2 for i in x], object_attn_per_token, width, 
                   label='Object Slots', color='steelblue')
    
    ax.set_xlabel('Generated Token', fontsize=11)
    ax.set_ylabel('Total Attention Weight', fontsize=11)
    ax.set_title('Attention Distribution: Image vs Objects per Token', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(caption_tokens, rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'attention_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: attention_distribution.png")
    
    return attention_matrix, caption_tokens, objects


def visualize_multiple_samples(
    model,
    image_folder,
    vocab,
    object_tags_dict,
    device,
    save_dir,
    num_samples=5
):
    """Generate attention visualizations for multiple samples."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Get images that have object tags
    available_images = [f for f in object_tags_dict.keys() 
                       if os.path.exists(os.path.join(image_folder, f))]
    
    import random
    random.seed(42)
    selected = random.sample(available_images, min(num_samples, len(available_images)))
    
    print(f"\n📊 Generating attention maps for {len(selected)} samples...")
    
    for i, img_name in enumerate(selected):
        print(f"\n[{i+1}/{len(selected)}] Processing: {img_name}")
        sample_dir = os.path.join(save_dir, f"sample_{i+1}")
        
        visualize_cross_attention_heatmap(
            model=model,
            image_path=os.path.join(image_folder, img_name),
            vocab=vocab,
            object_tags_dict=object_tags_dict,
            device=device,
            save_dir=sample_dir
        )
    
    print(f"\n✅ All attention maps saved to: {save_dir}")


def main():
    """Main function to run attention visualization."""
    import argparse
    
    # Get the project root for default paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    def find_file(possible_paths):
        for path in possible_paths:
            if os.path.exists(path):
                return path
        return possible_paths[0]
    
    default_vocab = find_file([
        os.path.join(project_root, 'code', 'Pkl Files', 'vocab.pkl'),
        os.path.join(project_root, 'code', 'preprocessing', 'vocab.pkl'),
        r'E:\A3\IML a3\code\preprocessing\vocab.pkl',
    ])
    
    default_checkpoint = find_file([
        os.path.join(project_root, 'code', 'preprocessing', 'results_automated', 'best_model.pt'),
        os.path.join(project_root, 'outputs', 'best_model.pt'),
        r'E:\A3\IML a3\code\preprocessing\results_automated\best_model.pt',
    ])
    
    default_object_tags = find_file([
        os.path.join(project_root, 'code', 'Pkl Files', 'object_tags_precomputed.json'),
        os.path.join(project_root, 'code', 'preprocessing', 'object_tags_precomputed.json'),
        os.path.join(project_root, 'data', 'object_tags_precomputed.json'),
        r'E:\A3\IML a3\code\preprocessing\object_tags_precomputed.json',
    ])
    
    default_img_folder = find_file([
        os.path.join(project_root, 'data', 'final_data'),
        r'E:\A3\IML a3\data\final_data',
    ])
    
    default_save_dir = os.path.join(project_root, 'code', 'preprocessing', 'results_automated', 'attention_maps')
    
    parser = argparse.ArgumentParser(description='Visualize VLT Cross-Attention')
    parser.add_argument('--checkpoint', type=str,
                        default=default_checkpoint)
    parser.add_argument('--vocab', type=str,
                        default=default_vocab)
    parser.add_argument('--object_tags', type=str,
                        default=default_object_tags)
    parser.add_argument('--image', type=str, default=None,
                        help='Single image path (or use --multi for multiple)')
    parser.add_argument('--image_folder', type=str,
                        default=default_img_folder)
    parser.add_argument('--save_dir', type=str,
                        default=default_save_dir)
    parser.add_argument('--multi', action='store_true',
                        help='Generate for multiple random samples')
    parser.add_argument('--num_samples', type=int, default=5)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    
    # Load vocab
    print("📚 Loading vocabulary...")
    with open(args.vocab, 'rb') as f:
        vocab = pickle.load(f)
    
    # Load object tags
    print("🏷️  Loading object tags...")
    with open(args.object_tags, 'r') as f:
        object_tags_dict = json.load(f)
    print(f"   Found tags for {len(object_tags_dict)} images")
    
    # Initialize collate caches
    load_object_tags_cache(args.object_tags)
    set_vocab_cache(vocab['stoi'], vocab['pad_id'])
    
    # Load model
    print("🤖 Loading model...")
    model = OptimizedVLT(
        vocab_size=vocab['vocab_size'],
        pad_token_id=vocab['pad_id'],
        start_token_id=vocab['start_id'],
        end_token_id=vocab['end_id'],
        max_objects=10,
        embed_size=512,
        hidden_size=512,
        num_heads=8,
        num_layers=4,
        dropout_rate=0.2,
        device=device
    ).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print("   ✅ Model loaded")
    
    if args.multi:
        visualize_multiple_samples(
            model=model,
            image_folder=args.image_folder,
            vocab=vocab,
            object_tags_dict=object_tags_dict,
            device=device,
            save_dir=args.save_dir,
            num_samples=args.num_samples
        )
    elif args.image:
        visualize_cross_attention_heatmap(
            model=model,
            image_path=args.image,
            vocab=vocab,
            object_tags_dict=object_tags_dict,
            device=device,
            save_dir=args.save_dir
        )
    else:
        # Default: use first image with objects
        for img_name in object_tags_dict.keys():
            img_path = os.path.join(args.image_folder, img_name)
            if os.path.exists(img_path):
                visualize_cross_attention_heatmap(
                    model=model,
                    image_path=img_path,
                    vocab=vocab,
                    object_tags_dict=object_tags_dict,
                    device=device,
                    save_dir=args.save_dir
                )
                break


if __name__ == '__main__':
    main()
