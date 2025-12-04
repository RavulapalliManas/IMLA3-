"""
Evaluate BLEU and ROUGE metrics for trained image captioning model.
Compares model predictions against ground truth captions on test set.
"""

import torch
import pickle
import argparse
import json
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import numpy as np

# Use relative imports when running as part of the package
try:
    from .dataloader import ArtEmisDataset
    from .collate import artemis_collate_fn
    from .model_transformer import OptimizedVLT
except ImportError:
    # Fallback for running script directly
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dataloader import ArtEmisDataset
    from collate import artemis_collate_fn
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
    
    default_dataset = find_file([
        os.path.join(project_root, 'code', 'Pkl Files', 'preprocessed_dataset_with_tokens.pkl'),
        os.path.join(project_root, 'code', 'preprocessing', 'preprocessed_dataset_with_tokens.pkl'),
        r'E:\A3\IML a3\code\preprocessing\preprocessed_dataset_with_tokens.pkl',
    ])
    
    default_img = find_file([
        os.path.join(project_root, 'data', 'final_data'),
        r'E:\A3\IML a3\data\final_data',
    ])
    
    default_checkpoint = find_file([
        os.path.join(project_root, 'code', 'preprocessing', 'results_automated', 'best_model.pt'),
        os.path.join(project_root, 'outputs', 'best_model.pt'),
        r'E:\A3\results_automated\best_val_loss_epoch29_20251204_112027\model.pt',
    ])
    
    parser = argparse.ArgumentParser(description='Evaluate BLEU/ROUGE metrics')
    parser.add_argument('--checkpoint', type=str, 
                        default=default_checkpoint,
                        help='Path to model checkpoint')
    parser.add_argument('--vocab', type=str,
                        default=default_vocab,
                        help='Path to vocabulary file')
    parser.add_argument('--dataset', type=str,
                        default=default_dataset,
                        help='Path to dataset pickle')
    parser.add_argument('--img_folder', type=str,
                        default=default_img,
                        help='Path to images folder')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test', 'all'],
                        help='Which split to evaluate on')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--max_len', type=int, default=30,
                        help='Maximum caption length')
    parser.add_argument('--num_samples', type=int, default=-1,
                        help='Number of samples to evaluate (-1 for all)')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save predictions to JSON file')
    return parser.parse_args()


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


def generate_caption_greedy(model, image, max_len, vocab, device):
    """Generate caption using greedy decoding."""
    model.eval()
    with torch.no_grad():
        # Encode image
        memory = model.vision_encoder(image)
        memory = model.visual_projection(memory)
        
        # Start with <START> token
        generated = [vocab['start_id']]
        
        for _ in range(max_len):
            tgt_input = torch.LongTensor([generated]).to(device)
            tgt_emb = model.embedding(tgt_input)
            tgt_emb = model.pos_encoder(tgt_emb)
            
            output = model.decoder(tgt=tgt_emb, memory=memory)
            logits = model.output_head(output[:, -1, :])
            
            next_token = logits.argmax(dim=-1).item()
            generated.append(next_token)
            
            if next_token == vocab['end_id']:
                break
                
    return generated


def calculate_bleu_scores(references, hypotheses):
    """
    Calculate BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores.
    
    Args:
        references: List of reference captions (each is list of lists of words)
        hypotheses: List of predicted captions (each is list of words)
    
    Returns:
        Dict with BLEU scores
    """
    smoothing = SmoothingFunction().method1
    
    # Corpus-level BLEU (standard metric)
    bleu1 = corpus_bleu(references, hypotheses, weights=(1.0, 0, 0, 0), smoothing_function=smoothing)
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
    bleu3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
    
    # Average sentence-level BLEU (more lenient for short captions)
    sent_bleu1 = np.mean([sentence_bleu(ref, hyp, weights=(1.0, 0, 0, 0), smoothing_function=smoothing) 
                          for ref, hyp in zip(references, hypotheses)])
    sent_bleu2 = np.mean([sentence_bleu(ref, hyp, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing) 
                          for ref, hyp in zip(references, hypotheses)])
    sent_bleu4 = np.mean([sentence_bleu(ref, hyp, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing) 
                          for ref, hyp in zip(references, hypotheses)])
    
    return {
        'corpus_bleu1': bleu1,
        'corpus_bleu2': bleu2,
        'corpus_bleu3': bleu3,
        'corpus_bleu4': bleu4,
        'sentence_bleu1': sent_bleu1,
        'sentence_bleu2': sent_bleu2,
        'sentence_bleu4': sent_bleu4
    }


def calculate_rouge_scores(references, hypotheses):
    """
    Calculate ROUGE-L scores.
    
    Args:
        references: List of reference caption strings
        hypotheses: List of predicted caption strings
    
    Returns:
        Dict with ROUGE scores
    """
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    scores = []
    for ref, hyp in zip(references, hypotheses):
        score = scorer.score(ref, hyp)
        scores.append(score['rougeL'].fmeasure)
    
    return {
        'rouge_l_mean': np.mean(scores),
        'rouge_l_std': np.std(scores),
        'rouge_l_min': np.min(scores),
        'rouge_l_max': np.max(scores)
    }


def evaluate_model(model, dataloader, vocab, device, max_len, save_predictions=False):
    """
    Evaluate model on dataset and compute metrics.
    
    Returns:
        metrics: Dict with BLEU/ROUGE scores
        predictions: List of (image_id, ground_truth, prediction) tuples
    """
    model.eval()
    
    references_text = []  # For ROUGE (strings)
    references_words = []  # For BLEU (list of word lists)
    hypotheses_text = []   # For ROUGE (strings)
    hypotheses_words = []  # For BLEU (list of words)
    
    predictions = []
    
    with torch.no_grad():
        for batch_idx, (images, _, target_ids) in enumerate(tqdm(dataloader, desc="Evaluating")):
            images = images.to(device)
            batch_size = images.size(0)
            
            # Generate captions for batch
            for i in range(batch_size):
                img = images[i:i+1]
                
                # Generate prediction
                pred_ids = generate_caption_greedy(model, img, max_len, vocab, device)
                pred_text = decode_caption(pred_ids, vocab)
                
                # Get ground truth
                gt_ids = target_ids[i]
                gt_text = decode_caption(gt_ids, vocab)
                
                # Store for metrics
                references_text.append(gt_text)
                hypotheses_text.append(pred_text)
                
                # For BLEU: reference is list of lists (allowing multiple refs per image)
                # We only have 1 reference per image, so wrap it
                references_words.append([gt_text.split()])
                hypotheses_words.append(pred_text.split())
                
                # Store predictions
                if save_predictions:
                    predictions.append({
                        'index': batch_idx * dataloader.batch_size + i,
                        'ground_truth': gt_text,
                        'prediction': pred_text,
                        'gt_length': len(gt_text.split()),
                        'pred_length': len(pred_text.split())
                    })
    
    # Calculate metrics
    print("\n" + "="*70)
    print("COMPUTING METRICS...")
    print("="*70)
    
    bleu_scores = calculate_bleu_scores(references_words, hypotheses_words)
    rouge_scores = calculate_rouge_scores(references_text, hypotheses_text)
    
    # Combine all metrics
    metrics = {
        **bleu_scores,
        **rouge_scores,
        'num_samples': len(hypotheses_text),
        'avg_pred_length': np.mean([len(h.split()) for h in hypotheses_text]),
        'avg_ref_length': np.mean([len(r.split()) for r in references_text])
    }
    
    return metrics, predictions


def print_metrics(metrics):
    """Pretty print metrics."""
    print("\n" + "="*70)
    print("📊 EVALUATION RESULTS")
    print("="*70)
    
    print("\n🔵 BLEU Scores (Corpus-level):")
    print(f"   BLEU-1: {metrics['corpus_bleu1']:.4f}")
    print(f"   BLEU-2: {metrics['corpus_bleu2']:.4f}")
    print(f"   BLEU-3: {metrics['corpus_bleu3']:.4f}")
    print(f"   BLEU-4: {metrics['corpus_bleu4']:.4f}")
    
    print("\n🔵 BLEU Scores (Sentence-level average):")
    print(f"   BLEU-1: {metrics['sentence_bleu1']:.4f}")
    print(f"   BLEU-2: {metrics['sentence_bleu2']:.4f}")
    print(f"   BLEU-4: {metrics['sentence_bleu4']:.4f}")
    
    print("\n🔴 ROUGE-L Scores:")
    print(f"   Mean:   {metrics['rouge_l_mean']:.4f}")
    print(f"   Std:    {metrics['rouge_l_std']:.4f}")
    print(f"   Min:    {metrics['rouge_l_min']:.4f}")
    print(f"   Max:    {metrics['rouge_l_max']:.4f}")
    
    print("\n📏 Caption Statistics:")
    print(f"   Samples evaluated: {metrics['num_samples']}")
    print(f"   Avg prediction length: {metrics['avg_pred_length']:.1f} words")
    print(f"   Avg reference length:  {metrics['avg_ref_length']:.1f} words")
    
    print("\n" + "="*70)


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*70)
    print("🧪 IMAGE CAPTIONING MODEL EVALUATION")
    print("="*70)
    
    # =====================
    # 1. LOAD VOCABULARY
    # =====================
    print(f"\n📚 Loading vocabulary from: {args.vocab}")
    with open(args.vocab, 'rb') as f:
        vocab = pickle.load(f)
    print(f"   Vocab size: {vocab['vocab_size']}")
    
    # =====================
    # 2. LOAD MODEL
    # =====================
    print(f"\n🤖 Loading model from: {args.checkpoint}")
    
    model = OptimizedVLT(
        vocab_size=vocab["vocab_size"],
        pad_token_id=vocab["pad_id"],
        start_token_id=vocab["start_id"],
        end_token_id=vocab["end_id"],
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
    # 3. LOAD DATASET
    # =====================
    print(f"\n📊 Loading dataset from: {args.dataset}")
    with open(args.dataset, 'rb') as f:
        full_df = pickle.load(f)
    
    # Create split (same as training: 90/5/5)
    import random
    indices = list(range(len(full_df)))
    random.seed(42)
    random.shuffle(indices)
    
    val_split = int(0.05 * len(full_df))
    test_split = int(0.10 * len(full_df))
    
    if args.split == 'train':
        selected_indices = indices[test_split:]
        dataset = Subset(ArtEmisDataset(args.dataset, args.img_folder, split='train'), selected_indices)
    elif args.split == 'val':
        selected_indices = indices[:val_split]
        dataset = Subset(ArtEmisDataset(args.dataset, args.img_folder, split='val'), selected_indices)
    elif args.split == 'test':
        selected_indices = indices[val_split:test_split]
        dataset = Subset(ArtEmisDataset(args.dataset, args.img_folder, split='val'), selected_indices)
    else:  # all
        dataset = ArtEmisDataset(args.dataset, args.img_folder, split='train')
    
    # Limit samples if requested
    if args.num_samples > 0 and args.num_samples < len(dataset):
        dataset = Subset(dataset, list(range(args.num_samples)))
    
    print(f"   Evaluating on {args.split} split: {len(dataset)} samples")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # Single-threaded for safety
        collate_fn=artemis_collate_fn
    )
    
    # =====================
    # 4. EVALUATE
    # =====================
    metrics, predictions = evaluate_model(
        model, dataloader, vocab, device, args.max_len, 
        save_predictions=args.save_predictions
    )
    
    # =====================
    # 5. DISPLAY RESULTS
    # =====================
    print_metrics(metrics)
    
    # =====================
    # 6. SAVE RESULTS
    # =====================
    # Use flexible output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output_dir = os.path.join(project_root, 'code', 'preprocessing', 'results_automated')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    metrics_file = os.path.join(output_dir, f'metrics_{args.split}.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"\n💾 Metrics saved to: {metrics_file}")
    
    # Save predictions if requested
    if args.save_predictions:
        pred_file = f'{output_dir}/predictions_{args.split}.json'
        with open(pred_file, 'w') as f:
            json.dump(predictions, f, indent=2)
        print(f"💾 Predictions saved to: {pred_file}")
        
        # Print some examples
        print("\n" + "="*70)
        print("📝 SAMPLE PREDICTIONS (first 5):")
        print("="*70)
        for i, pred in enumerate(predictions[:5]):
            print(f"\n[{i+1}] Sample {pred['index']}:")
            print(f"   Ground Truth: {pred['ground_truth']}")
            print(f"   Prediction:   {pred['prediction']}")
            print(f"   Lengths: GT={pred['gt_length']}, Pred={pred['pred_length']}")


if __name__ == '__main__':
    main()
