"""
Advanced Inference Script for Image Captioning Model
Supports: Beam Search, Temperature Sampling, Nucleus Sampling

NOW WITH HYBRID OBJECT TAG SUPPORT:
- Loads pre-computed object tags from JSON
- Injects object tokens into multi-modal memory
- Produces more objective, noun-rich captions
"""

import torch
import pickle
import argparse
import json
import os
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

# Use relative imports when running as part of the package
try:
    from .model_transformer import OptimizedVLT
except ImportError:
    # Fallback for running script directly
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model_transformer import OptimizedVLT

def get_args():
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
        r'E:\A3\IML a3\code\preprocessing\results_automated\best_val_loss_epoch31_20251204_133002\model.pt',
    ])
    
    default_object_tags = find_file([
        os.path.join(project_root, 'code', 'Pkl Files', 'object_tags_precomputed.json'),
        os.path.join(project_root, 'code', 'preprocessing', 'object_tags_precomputed.json'),
        os.path.join(project_root, 'data', 'object_tags_precomputed.json'),
        r'E:\A3\IML a3\code\preprocessing\object_tags_precomputed.json',
    ])
    
    default_image = find_file([
        os.path.join(project_root, 'data', 'final_data_resized', 'adriaen-brouwer_peasants-smoking-and-drinking.jpg'),
        os.path.join(project_root, 'data', 'final_data', 'adriaen-brouwer_peasants-smoking-and-drinking.jpg'),
        r'E:\A3\IML a3\data\final_data_resized\adriaen-brouwer_peasants-smoking-and-drinking.jpg',
    ])
    
    parser = argparse.ArgumentParser(description='Generate top-3 captions for image with emotion tag')
    parser.add_argument('--image', type=str, 
                        default=default_image,
                        help='Path to input image')
    parser.add_argument('--emotion', type=str, 
                        default='awe',
                        help='Emotion tag (e.g., awe, fear, anger, sadness, joy, etc.)')
    parser.add_argument('--checkpoint', type=str, 
                        default=default_checkpoint,
                        help='Path to model checkpoint')
    parser.add_argument('--vocab', type=str,
                        default=default_vocab,
                        help='Path to vocabulary file')
    parser.add_argument('--object_tags', type=str,
                        default=default_object_tags,
                        help='Path to pre-computed object tags JSON')
    parser.add_argument('--max_len', type=int, default=30,
                        help='Maximum caption length')
    parser.add_argument('--beam_size', type=int, default=10,
                        help='Beam size for beam search (higher for more top-3 candidates)')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature (higher=more diverse)')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='Nucleus sampling threshold')
    parser.add_argument('--use_objects', action='store_true', default=True,
                        help='Use object tags for hybrid captioning')
    return parser.parse_args()


def load_object_tags(json_path):
    """Load pre-computed object tags from JSON."""
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)
    return {}


def get_object_token_ids(image_filename, object_tags_dict, vocab_stoi, max_objects=10, pad_id=0):
    """
    Convert object tag strings to token IDs with padding.
    
    Args:
        image_filename: Name of the image file (e.g., 'painting.jpg')
        object_tags_dict: Dict from load_object_tags
        vocab_stoi: String-to-index vocabulary
        max_objects: Fixed size for batching
        pad_id: Padding token ID
    
    Returns:
        torch.Tensor: (max_objects,) - Padded object token IDs
    """
    objects = object_tags_dict.get(image_filename, [])
    
    token_ids = []
    for obj in objects[:max_objects]:
        if obj in vocab_stoi:
            token_ids.append(vocab_stoi[obj])
    
    # Pad to max_objects
    while len(token_ids) < max_objects:
        token_ids.append(pad_id)
    
    return torch.tensor(token_ids[:max_objects], dtype=torch.long)


def decode_caption(token_ids, vocab):
    """Convert token IDs to readable text."""
    words = []
    for tid in token_ids:
        tid = int(tid)
        if tid == vocab['start_id']: 
            continue
        if tid == vocab['end_id']: 
            break
        if tid == vocab['pad_id']: 
            continue
        words.append(vocab['itos'][tid])
    return " ".join(words)


def greedy_decode(model, image_tensor, max_len, vocab, device, object_input_ids=None):
    """
    Greedy Decoding with Hybrid Object Memory.
    Always pick the most probable next token.
    """
    model.eval()
    with torch.no_grad():
        # Encode image
        memory = model.vision_encoder(image_tensor)
        if memory.dim() == 2:
            memory = memory.unsqueeze(1)
        memory = model.visual_projection(memory)
        
        # Build hybrid memory if objects provided
        memory_key_padding_mask = None
        if object_input_ids is not None:
            obj_embeddings = model.embedding(object_input_ids.to(device))
            
            # Create padding mask for objects
            obj_pad_mask = (object_input_ids == model.pad_token_id)
            img_mask = torch.zeros(memory.size(0), memory.size(1), 
                                  dtype=torch.bool, device=device)
            memory_key_padding_mask = torch.cat([img_mask, obj_pad_mask.to(device)], dim=1)
            
            # Concatenate image + object memory
            memory = torch.cat([memory, obj_embeddings], dim=1)
        
        # Start with <START> token
        generated = [vocab['start_id']]
        
        for _ in range(max_len):
            tgt_input = torch.LongTensor([generated]).to(device)
            tgt_emb = model.embedding(tgt_input)
            tgt_emb = model.pos_encoder(tgt_emb)
            
            # Decode with hybrid memory
            output = model.decoder(tgt=tgt_emb, memory=memory,
                                  memory_key_padding_mask=memory_key_padding_mask)
            logits = model.output_head(output[:, -1, :])
            
            next_token = logits.argmax(dim=-1).item()
            generated.append(next_token)
            
            if next_token == vocab['end_id']:
                break
                
    return generated


def beam_search_decode(model, image_tensor, max_len, beam_size, vocab, device, object_input_ids=None):
    """
    Beam Search with Hybrid Object Memory.
    Maintain top-K hypotheses at each step.
    """
    model.eval()
    with torch.no_grad():
        # Encode image
        memory = model.vision_encoder(image_tensor)
        if memory.dim() == 2:
            memory = memory.unsqueeze(1)
        memory = model.visual_projection(memory)
        
        # Build hybrid memory if objects provided
        memory_key_padding_mask = None
        if object_input_ids is not None:
            obj_embeddings = model.embedding(object_input_ids.to(device))
            
            obj_pad_mask = (object_input_ids == model.pad_token_id)
            img_mask = torch.zeros(memory.size(0), memory.size(1), 
                                  dtype=torch.bool, device=device)
            memory_key_padding_mask = torch.cat([img_mask, obj_pad_mask.to(device)], dim=1)
            
            memory = torch.cat([memory, obj_embeddings], dim=1)
        
        # Initialize beam
        beams = [([vocab['start_id']], 0.0)]
        
        for step in range(max_len):
            candidates = []
            
            for seq, score in beams:
                if seq[-1] == vocab['end_id']:
                    candidates.append((seq, score))
                    continue
                
                tgt_input = torch.LongTensor([seq]).to(device)
                tgt_emb = model.embedding(tgt_input)
                tgt_emb = model.pos_encoder(tgt_emb)
                
                output = model.decoder(tgt=tgt_emb, memory=memory,
                                       memory_key_padding_mask=memory_key_padding_mask)
                logits = model.output_head(output[:, -1, :])
                log_probs = F.log_softmax(logits, dim=-1)
                
                top_probs, top_indices = log_probs.topk(beam_size, dim=-1)
                
                for i in range(beam_size):
                    token = top_indices[0, i].item()
                    token_score = top_probs[0, i].item()
                    new_seq = seq + [token]
                    new_score = score + token_score
                    candidates.append((new_seq, new_score))
            
            beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]
            
            if all(seq[-1] == vocab['end_id'] for seq, _ in beams):
                break
        
        best_seq, best_score = beams[0]
        return best_seq


def temperature_sample_decode(model, image_tensor, max_len, temperature, vocab, device, object_input_ids=None):
    """
    Temperature Sampling with Hybrid Object Memory.
    Higher temperature = more randomness/creativity.
    """
    model.eval()
    with torch.no_grad():
        memory = model.vision_encoder(image_tensor)
        if memory.dim() == 2:
            memory = memory.unsqueeze(1)
        memory = model.visual_projection(memory)
        
        # Build hybrid memory if objects provided
        memory_key_padding_mask = None
        if object_input_ids is not None:
            obj_embeddings = model.embedding(object_input_ids.to(device))
            
            obj_pad_mask = (object_input_ids == model.pad_token_id)
            img_mask = torch.zeros(memory.size(0), memory.size(1), 
                                  dtype=torch.bool, device=device)
            memory_key_padding_mask = torch.cat([img_mask, obj_pad_mask.to(device)], dim=1)
            
            memory = torch.cat([memory, obj_embeddings], dim=1)
        
        generated = [vocab['start_id']]
        
        for _ in range(max_len):
            tgt_input = torch.LongTensor([generated]).to(device)
            tgt_emb = model.embedding(tgt_input)
            tgt_emb = model.pos_encoder(tgt_emb)
            
            output = model.decoder(tgt=tgt_emb, memory=memory,
                                  memory_key_padding_mask=memory_key_padding_mask)
            logits = model.output_head(output[:, -1, :])
            
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)
            
            if next_token == vocab['end_id']:
                break
                
    return generated


def nucleus_sample_decode(model, image_tensor, max_len, top_p, vocab, device, object_input_ids=None):
    """
    Nucleus (Top-P) Sampling with Hybrid Object Memory.
    More diverse than greedy, more controlled than pure temperature.
    """
    model.eval()
    with torch.no_grad():
        memory = model.vision_encoder(image_tensor)
        if memory.dim() == 2:
            memory = memory.unsqueeze(1)
        memory = model.visual_projection(memory)
        
        # Build hybrid memory if objects provided
        memory_key_padding_mask = None
        if object_input_ids is not None:
            obj_embeddings = model.embedding(object_input_ids.to(device))
            
            obj_pad_mask = (object_input_ids == model.pad_token_id)
            img_mask = torch.zeros(memory.size(0), memory.size(1), 
                                  dtype=torch.bool, device=device)
            memory_key_padding_mask = torch.cat([img_mask, obj_pad_mask.to(device)], dim=1)
            
            memory = torch.cat([memory, obj_embeddings], dim=1)
        
        generated = [vocab['start_id']]
        
        for _ in range(max_len):
            tgt_input = torch.LongTensor([generated]).to(device)
            tgt_emb = model.embedding(tgt_input)
            tgt_emb = model.pos_encoder(tgt_emb)
            
            output = model.decoder(tgt=tgt_emb, memory=memory,
                                  memory_key_padding_mask=memory_key_padding_mask)
            logits = model.output_head(output[:, -1, :])
            
            probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            nucleus_mask = cumulative_probs <= top_p
            nucleus_mask[0, 0] = True
            
            filtered_probs = sorted_probs.clone()
            filtered_probs[~nucleus_mask] = 0.0
            filtered_probs = filtered_probs / filtered_probs.sum()
            
            sample_idx = torch.multinomial(filtered_probs, num_samples=1)
            next_token = sorted_indices[0, sample_idx].item()
            generated.append(next_token)
            
            if next_token == vocab['end_id']:
                break
                
    return generated


def beam_search_top_k(model, image_tensor, max_len, beam_size, vocab, device, k=3, object_input_ids=None):
    """
    Beam Search that returns top-K sequences and their scores.
    
    Args:
        model: The transformer model
        image_tensor: Preprocessed image tensor
        max_len: Maximum caption length
        beam_size: Beam width (higher for better top-K)
        vocab: Vocabulary dictionary
        device: torch device
        k: Number of top sequences to return (default 3)
        object_input_ids: Optional object IDs for hybrid memory
    
    Returns:
        List of (sequence, score) tuples, sorted by score (descending)
    """
    model.eval()
    with torch.no_grad():
        # Encode image
        memory = model.vision_encoder(image_tensor)
        if memory.dim() == 2:
            memory = memory.unsqueeze(1)
        memory = model.visual_projection(memory)
        
        # Build hybrid memory if objects provided
        memory_key_padding_mask = None
        if object_input_ids is not None:
            obj_embeddings = model.embedding(object_input_ids.to(device))
            
            obj_pad_mask = (object_input_ids == model.pad_token_id)
            img_mask = torch.zeros(memory.size(0), memory.size(1), 
                                  dtype=torch.bool, device=device)
            memory_key_padding_mask = torch.cat([img_mask, obj_pad_mask.to(device)], dim=1)
            
            memory = torch.cat([memory, obj_embeddings], dim=1)
        
        # Initialize beam
        beams = [([vocab['start_id']], 0.0)]
        
        for step in range(max_len):
            candidates = []
            
            for seq, score in beams:
                if seq[-1] == vocab['end_id']:
                    candidates.append((seq, score))
                    continue
                
                tgt_input = torch.LongTensor([seq]).to(device)
                tgt_emb = model.embedding(tgt_input)
                tgt_emb = model.pos_encoder(tgt_emb)
                
                output = model.decoder(tgt=tgt_emb, memory=memory,
                                       memory_key_padding_mask=memory_key_padding_mask)
                logits = model.output_head(output[:, -1, :])
                log_probs = F.log_softmax(logits, dim=-1)
                
                top_probs, top_indices = log_probs.topk(beam_size, dim=-1)
                
                for i in range(beam_size):
                    token = top_indices[0, i].item()
                    token_score = top_probs[0, i].item()
                    new_seq = seq + [token]
                    new_score = score + token_score
                    candidates.append((new_seq, new_score))
            
            beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]
            
            if all(seq[-1] == vocab['end_id'] for seq, _ in beams):
                break
        
        # Return top-K beams
        return sorted(beams, key=lambda x: x[1], reverse=True)[:k]


def score_caption_objectivity(caption_text, object_tags):
    """
    Score caption for objectivity based on presence of object/noun terms.
    Higher score = more objective (contains detected objects).
    
    Args:
        caption_text: Caption string
        object_tags: List of detected object tag strings
    
    Returns:
        float: Objectivity score [0, 1]
    """
    caption_lower = caption_text.lower()
    num_objects_mentioned = sum(1 for obj in object_tags if obj.lower() in caption_lower)
    objectivity_score = num_objects_mentioned / max(len(object_tags), 1)
    return objectivity_score


def score_caption_subjectivity(caption_text, emotion_tag):
    """
    Score caption for subjectivity based on presence of emotion/feeling terms.
    Higher score = more subjective (contains emotion words).
    
    Args:
        caption_text: Caption string
        emotion_tag: The emotion tag that should be present
    
    Returns:
        float: Subjectivity score [0, 1]
    """
    caption_lower = caption_text.lower()
    emotion_lower = emotion_tag.lower()
    
    # Check if emotion tag is present in caption
    if emotion_lower in caption_lower:
        return 1.0
    
    # Check for synonyms/related emotional words
    emotion_synonyms = {
        'awe': ['amazing', 'wonderful', 'breathtaking', 'astonishing', 'impressed', 'amazed'],
        'fear': ['scary', 'frightening', 'terrifying', 'disturbing', 'afraid', 'frightened'],
        'anger': ['angry', 'furious', 'enraged', 'upset', 'aggressive', 'hostile'],
        'sadness': ['sad', 'sorrowful', 'melancholic', 'mournful', 'gloomy', 'depressing'],
        'joy': ['happy', 'joyful', 'cheerful', 'delighted', 'pleased', 'glad'],
        'disgust': ['disgusting', 'repulsive', 'revolting', 'gross', 'nasty'],
    }
    
    related_words = emotion_synonyms.get(emotion_lower, [])
    num_matches = sum(1 for word in related_words if word in caption_lower)
    
    return min(num_matches / max(len(related_words), 1), 1.0)


def score_caption_combined(caption_text, emotion_tag, object_tags, weights=(0.5, 0.5)):
    """
    Combine objectivity and subjectivity scores for balanced caption evaluation.
    
    Args:
        caption_text: Caption string
        emotion_tag: Emotion tag
        object_tags: List of detected objects
        weights: Tuple of (objectivity_weight, subjectivity_weight), must sum to 1
    
    Returns:
        dict: {'combined_score': float, 'objectivity': float, 'subjectivity': float}
    """
    obj_score = score_caption_objectivity(caption_text, object_tags)
    subj_score = score_caption_subjectivity(caption_text, emotion_tag)
    
    combined = weights[0] * obj_score + weights[1] * subj_score
    
    return {
        'combined_score': combined,
        'objectivity': obj_score,
        'subjectivity': subj_score
    }


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # =====================
    # 1. LOAD VOCABULARY
    # =====================
    print(f"📚 Loading vocabulary from: {args.vocab}")
    with open(args.vocab, 'rb') as f:
        vocab = pickle.load(f)
    print(f"   Vocab size: {vocab['vocab_size']}")
    
    # =====================
    # 2. VALIDATE EMOTION TAG
    # =====================
    print(f"\n😊 Emotion tag: {args.emotion}")
    if args.emotion not in vocab['stoi']:
        print(f"⚠️  WARNING: Emotion '{args.emotion}' not in vocabulary!")
        print(f"   Available emotions: See vocabulary")
    
    # =====================
    # 3. LOAD OBJECT TAGS
    # =====================
    object_tags_dict = {}
    detected_objects = []
    if args.use_objects:
        print(f"\n🏷️  Loading object tags from: {args.object_tags}")
        object_tags_dict = load_object_tags(args.object_tags)
        print(f"   Loaded tags for {len(object_tags_dict)} images")
    
    # =====================
    # 4. LOAD MODEL
    # =====================
    print(f"\n🤖 Loading model from: {args.checkpoint}")
    
    model = OptimizedVLT(
        vocab_size=vocab["vocab_size"],
        pad_token_id=vocab["pad_id"],
        start_token_id=vocab["start_id"],
        end_token_id=vocab["end_id"],
        max_objects=10,
        embed_size=512,
        hidden_size=512,
        num_heads=8,
        num_layers=4,
        dropout_rate=0.2,
        device=device
    ).to(device)
    
    # Load weights
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("   ✅ Model loaded successfully")
    
    # =====================
    # 5. LOAD & PREPROCESS IMAGE
    # =====================
    print(f"\n🖼️  Loading image: {args.image}")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(args.image).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # =====================
    # 6. GET OBJECT TAGS FOR THIS IMAGE
    # =====================
    object_input_ids = None
    image_filename = os.path.basename(args.image)
    
    if args.use_objects and image_filename in object_tags_dict:
        detected_objects = object_tags_dict[image_filename]
        print(f"\n🏷️  Detected objects: {detected_objects}")
        object_input_ids = get_object_token_ids(
            image_filename, 
            object_tags_dict, 
            vocab['stoi'], 
            max_objects=10, 
            pad_id=vocab['pad_id']
        ).unsqueeze(0).to(device)
    else:
        print(f"\n⚠️  No object tags found for: {image_filename}")
        print("   Running in image-only mode (subjective only)")
    
    # =====================
    # 7. GENERATE TOP-3 CAPTIONS
    # =====================
    print(f"\n🔮 Generating top-3 captions with emotion: '{args.emotion}'")
    print(f"   Using beam search with beam_size={args.beam_size}")
    
    # Generate top-K candidates using beam search
    top_sequences = beam_search_top_k(
        model, img_tensor, args.max_len, args.beam_size, 
        vocab, device, k=10, object_input_ids=object_input_ids
    )
    
    # Decode sequences and score them
    captions_with_scores = []
    for seq, beam_score in top_sequences:
        caption_text = decode_caption(seq, vocab)
        
        # Score the caption for objectivity and subjectivity
        scores = score_caption_combined(
            caption_text, 
            args.emotion, 
            detected_objects,
            weights=(0.5, 0.5)  # Equal weight to objective and subjective
        )
        
        captions_with_scores.append({
            'text': caption_text,
            'beam_score': beam_score,
            'combined_score': scores['combined_score'],
            'objectivity': scores['objectivity'],
            'subjectivity': scores['subjectivity']
        })
    
    # Sort by combined score (descending)
    captions_with_scores.sort(key=lambda x: x['combined_score'], reverse=True)
    
    # =====================
    # 8. DISPLAY TOP-3 RESULTS
    # =====================
    print("\n" + "="*90)
    print("🏆 TOP-3 CAPTIONS (Best Objective & Subjective Combination)")
    print("="*90)
    
    for rank, caption_info in enumerate(captions_with_scores[:3], 1):
        print(f"\n#{rank} CAPTION:")
        print(f"   Text: {caption_info['text']}")
        print(f"   └─ Combined Score: {caption_info['combined_score']:.4f}")
        print(f"      ├─ Objectivity  (has detected objects): {caption_info['objectivity']:.4f}")
        print(f"      └─ Subjectivity (captures emotion): {caption_info['subjectivity']:.4f}")
    
    print("\n" + "="*90)
    print("📊 SUMMARY:")
    print("="*90)
    print(f"Image:             {image_filename}")
    print(f"Emotion Tag:       {args.emotion}")
    if detected_objects:
        print(f"Detected Objects:  {', '.join(detected_objects)}")
    print(f"Best Caption:      {captions_with_scores[0]['text']}")
    print("="*90)


if __name__ == '__main__':
    main()
