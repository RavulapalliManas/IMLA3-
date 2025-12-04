"""
Inference Script for CNN + LSTM Image Captioning Model
=======================================================
Features:
    - Single image caption generation
    - Batch inference
    - Greedy and beam search decoding
    - Visualization with attention (if available)
    - Emotion-conditioned generation

Usage:
    python inference_cnn_lstm.py --image path/to/image.jpg --emotion awe
    python inference_cnn_lstm.py --image_folder path/to/images --emotion contentment
"""

import os
import sys
import json
import pickle
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn_lstm_model import CNNLSTMCaptioner, load_model_checkpoint, create_model


# =============================================================================
# SECTION A: Constants and Configuration
# =============================================================================

EMOTION_MAP = {
    'contentment': 0, 'awe': 1, 'something_else': 2, 'sadness': 3,
    'amusement': 4, 'fear': 5, 'excitement': 6, 'disgust': 7, 'anger': 8
}

EMOTION_NAMES = list(EMOTION_MAP.keys())

# Image preprocessing (must match training transforms)
INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =============================================================================
# SECTION B: Caption Generator Class
# =============================================================================

class CaptionGenerator:
    """
    Caption generator for CNN-LSTM model.
    
    Handles:
        - Model loading
        - Image preprocessing
        - Caption generation (greedy/beam)
        - Text decoding
    """
    
    def __init__(
        self,
        model_path: str,
        vocab_path: str,
        tfidf_path: Optional[str] = None,
        device: str = 'cuda',
        config: Optional[Dict] = None
    ):
        """
        Initialize caption generator.
        
        Args:
            model_path: Path to model checkpoint (.pt file)
            vocab_path: Path to vocabulary pickle
            tfidf_path: Path to TF-IDF vectors (optional)
            device: Device to run on
            config: Model configuration (optional, loaded from checkpoint if not provided)
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Load vocabulary
        print("Loading vocabulary...")
        with open(vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)
        
        self.itos = self.vocab.get('itos', {})
        self.stoi = self.vocab.get('stoi', {})
        self.vocab_size = len(self.itos)
        
        self.pad_id = self.vocab.get('pad_id', 0)
        self.start_id = self.vocab.get('start_id', 2)
        self.end_id = self.vocab.get('end_id', 3)
        self.unk_id = self.vocab.get('unk_id', 1)
        
        print(f"  Vocabulary size: {self.vocab_size}")
        
        # Load model
        print("Loading model...")
        self._load_model(model_path, config, tfidf_path)
        print(f"  Model loaded on {self.device}")
        
        # Image transform
        self.transform = INFERENCE_TRANSFORM
    
    def _load_model(
        self,
        model_path: str,
        config: Optional[Dict],
        tfidf_path: Optional[str]
    ):
        """Load model from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get config from checkpoint if not provided
        if config is None:
            config = checkpoint.get('config', {})
        
        # Create model
        self.model = create_model(
            vocab_size=self.vocab_size,
            config=config,
            tfidf_path=tfidf_path
        )
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Store config
        self.config = config
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image tensor (1, 3, 224, 224)
        """
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image)
        return image_tensor.unsqueeze(0).to(self.device)
    
    def preprocess_pil_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess PIL image directly.
        
        Args:
            image: PIL Image object
            
        Returns:
            Preprocessed image tensor (1, 3, 224, 224)
        """
        image = image.convert('RGB')
        image_tensor = self.transform(image)
        return image_tensor.unsqueeze(0).to(self.device)
    
    def decode_tokens(self, token_ids: List[int]) -> str:
        """
        Convert token IDs to text.
        
        Args:
            token_ids: List of token indices
            
        Returns:
            Decoded caption string
        """
        words = []
        for tid in token_ids:
            if tid == self.end_id:
                break
            if tid not in [self.pad_id, self.start_id]:
                word = self.itos.get(tid, '<unk>')
                # Skip emotion and style tokens in output
                if not word.startswith('<emotion_') and not word.startswith('<style_'):
                    words.append(word)
        
        return ' '.join(words)
    
    @torch.no_grad()
    def generate(
        self,
        image: torch.Tensor,
        emotion: str = 'something_else',
        method: str = 'greedy',
        beam_width: int = 3,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0
    ) -> Tuple[str, List[int]]:
        """
        Generate caption for image.
        
        Args:
            image: Preprocessed image tensor (1, 3, 224, 224)
            emotion: Emotion label string
            method: 'greedy', 'beam', or 'sample'
            beam_width: Beam width for beam search
            temperature: Sampling temperature
            top_k: Top-k sampling (0 = disabled)
            top_p: Nucleus sampling threshold
            
        Returns:
            Tuple of (caption_text, token_ids)
        """
        self.model.eval()
        
        # Get emotion ID
        emotion_lower = emotion.lower().replace(' ', '_')
        emotion_id = EMOTION_MAP.get(emotion_lower, 2)  # Default to 'something_else'
        emotion_tensor = torch.tensor([emotion_id], device=self.device)
        
        # Generate tokens
        if method == 'greedy':
            token_ids = self.model.generate_greedy(
                image, emotion_tensor, temperature
            )[0].cpu().tolist()
        
        elif method == 'beam':
            token_ids = self.model.generate_beam(
                image, emotion_tensor, beam_width
            )[0].cpu().tolist()
        
        elif method == 'sample':
            token_ids = self._generate_sample(
                image, emotion_tensor, temperature, top_k, top_p
            )
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Decode to text
        caption = self.decode_tokens(token_ids)
        
        return caption, token_ids
    
    def _generate_sample(
        self,
        image: torch.Tensor,
        emotion_id: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float
    ) -> List[int]:
        """
        Sampling-based generation with top-k and nucleus sampling.
        
        Args:
            image: (1, 3, 224, 224)
            emotion_id: (1,)
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            
        Returns:
            List of token IDs
        """
        # Encode image
        image_features = self.model.encoder(image)
        hidden = self.model.decoder.init_hidden(image_features)
        
        # Get emotion embedding
        emotion_embed = self.model.decoder.emotion_embed(emotion_id).unsqueeze(1)
        
        # Start token
        current_token = torch.tensor([[self.start_id]], device=self.device)
        generated = [self.start_id]
        
        max_len = self.config.get('max_len', 33)
        
        for _ in range(max_len - 1):
            word_embed = self.model.decoder.embedding(current_token)
            logits, hidden = self.model.decoder.generate_step(
                word_embed, emotion_embed, hidden
            )
            
            # Apply temperature
            logits = logits / temperature
            
            # Top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            token_id = next_token.item()
            generated.append(token_id)
            current_token = next_token
            
            if token_id == self.end_id:
                break
        
        return generated
    
    def generate_from_path(
        self,
        image_path: str,
        emotion: str = 'something_else',
        method: str = 'greedy',
        **kwargs
    ) -> str:
        """
        Generate caption from image path.
        
        Args:
            image_path: Path to image file
            emotion: Emotion label
            method: Generation method
            **kwargs: Additional generation parameters
            
        Returns:
            Caption text
        """
        image = self.preprocess_image(image_path)
        caption, _ = self.generate(image, emotion, method, **kwargs)
        return caption
    
    def generate_batch(
        self,
        images: List[str],
        emotions: List[str],
        method: str = 'greedy',
        **kwargs
    ) -> List[str]:
        """
        Generate captions for multiple images.
        
        Args:
            images: List of image paths
            emotions: List of emotion labels
            method: Generation method
            
        Returns:
            List of caption texts
        """
        captions = []
        for img_path, emotion in zip(images, emotions):
            caption = self.generate_from_path(img_path, emotion, method, **kwargs)
            captions.append(caption)
        return captions


# =============================================================================
# SECTION C: Visualization Functions
# =============================================================================

def visualize_caption(
    image_path: str,
    caption: str,
    emotion: str,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Visualize image with generated caption.
    
    Args:
        image_path: Path to image
        caption: Generated caption
        emotion: Emotion label
        save_path: Optional path to save figure
        show: Whether to display the figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    image = Image.open(image_path)
    ax.imshow(image)
    ax.axis('off')
    
    # Title with emotion and caption
    title = f"Emotion: {emotion}\n\nCaption: {caption}"
    ax.set_title(title, fontsize=12, wrap=True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    if show:
        plt.show()
    
    plt.close()


def visualize_multi_emotion(
    generator: CaptionGenerator,
    image_path: str,
    emotions: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Generate and visualize captions for multiple emotions.
    
    Args:
        generator: CaptionGenerator instance
        image_path: Path to image
        emotions: List of emotions to try (default: all)
        save_path: Optional path to save figure
        show: Whether to display
    """
    if emotions is None:
        emotions = EMOTION_NAMES
    
    n_emotions = len(emotions)
    fig, axes = plt.subplots(1, 1, figsize=(12, 10))
    
    # Display image
    image = Image.open(image_path)
    axes.imshow(image)
    axes.axis('off')
    
    # Generate captions for each emotion
    caption_text = f"Image: {os.path.basename(image_path)}\n\n"
    
    image_tensor = generator.preprocess_image(image_path)
    
    for emotion in emotions:
        caption, _ = generator.generate(image_tensor, emotion, method='greedy')
        caption_text += f"[{emotion}]: {caption}\n\n"
    
    axes.set_title(caption_text, fontsize=10, loc='left', wrap=True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved multi-emotion visualization to {save_path}")
    
    if show:
        plt.show()
    
    plt.close()


def batch_inference_folder(
    generator: CaptionGenerator,
    folder_path: str,
    emotion: str = 'something_else',
    output_json: Optional[str] = None,
    method: str = 'greedy',
    limit: int = 100
) -> Dict[str, str]:
    """
    Run inference on all images in a folder.
    
    Args:
        generator: CaptionGenerator instance
        folder_path: Path to folder with images
        emotion: Emotion label to use
        output_json: Optional path to save results as JSON
        method: Generation method
        limit: Maximum number of images to process
        
    Returns:
        Dictionary mapping filenames to captions
    """
    results = {}
    
    # Get image files
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    image_files = [
        f for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in extensions
    ][:limit]
    
    print(f"Processing {len(image_files)} images...")
    
    for i, filename in enumerate(image_files):
        image_path = os.path.join(folder_path, filename)
        
        try:
            caption = generator.generate_from_path(
                image_path, emotion, method
            )
            results[filename] = caption
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(image_files)}")
                
        except Exception as e:
            print(f"  Error processing {filename}: {e}")
            results[filename] = f"ERROR: {str(e)}"
    
    # Save results
    if output_json:
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {output_json}")
    
    return results


# =============================================================================
# SECTION D: Interactive Demo
# =============================================================================

def interactive_demo(generator: CaptionGenerator):
    """
    Interactive caption generation demo.
    """
    print("\n" + "=" * 60)
    print("CNN-LSTM Caption Generator - Interactive Demo")
    print("=" * 60)
    print("Commands:")
    print("  - Enter image path to generate caption")
    print("  - 'e <emotion>' to change emotion (default: something_else)")
    print("  - 'm <method>' to change method (greedy/beam/sample)")
    print("  - 'all' to show all emotions for current image")
    print("  - 'q' to quit")
    print("=" * 60 + "\n")
    
    current_emotion = 'something_else'
    current_method = 'greedy'
    last_image = None
    
    while True:
        try:
            user_input = input(f"[{current_emotion}/{current_method}] > ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'q':
                print("Goodbye!")
                break
            
            if user_input.lower().startswith('e '):
                new_emotion = user_input[2:].strip().lower()
                if new_emotion in EMOTION_MAP:
                    current_emotion = new_emotion
                    print(f"  Emotion set to: {current_emotion}")
                else:
                    print(f"  Unknown emotion. Available: {', '.join(EMOTION_NAMES)}")
                continue
            
            if user_input.lower().startswith('m '):
                new_method = user_input[2:].strip().lower()
                if new_method in ['greedy', 'beam', 'sample']:
                    current_method = new_method
                    print(f"  Method set to: {current_method}")
                else:
                    print("  Unknown method. Available: greedy, beam, sample")
                continue
            
            if user_input.lower() == 'all' and last_image:
                print("\nGenerating captions for all emotions:")
                for emotion in EMOTION_NAMES:
                    image = generator.preprocess_image(last_image)
                    caption, _ = generator.generate(image, emotion, current_method)
                    print(f"  [{emotion}]: {caption}")
                print()
                continue
            
            # Assume it's an image path
            if os.path.exists(user_input):
                last_image = user_input
                caption = generator.generate_from_path(
                    user_input, current_emotion, current_method
                )
                print(f"\n  Caption: {caption}\n")
            else:
                print(f"  File not found: {user_input}")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"  Error: {e}")


# =============================================================================
# SECTION E: Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='CNN-LSTM Caption Generator Inference'
    )
    
    # Required arguments
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--vocab', type=str, required=True,
                        help='Path to vocabulary pickle')
    
    # Optional arguments
    parser.add_argument('--tfidf', type=str, default=None,
                        help='Path to TF-IDF vectors pickle')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu', 'mps'])
    
    # Input options (mutually exclusive)
    parser.add_argument('--image', type=str, default=None,
                        help='Single image path')
    parser.add_argument('--image_folder', type=str, default=None,
                        help='Folder of images for batch inference')
    parser.add_argument('--interactive', action='store_true',
                        help='Run interactive demo')
    
    # Generation options
    parser.add_argument('--emotion', type=str, default='something_else',
                        help=f'Emotion label. Options: {", ".join(EMOTION_NAMES)}')
    parser.add_argument('--method', type=str, default='greedy',
                        choices=['greedy', 'beam', 'sample'])
    parser.add_argument('--beam_width', type=int, default=3,
                        help='Beam width for beam search')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=0,
                        help='Top-k sampling (0=disabled)')
    parser.add_argument('--top_p', type=float, default=1.0,
                        help='Nucleus sampling threshold')
    
    # Output options
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for batch results')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize results')
    parser.add_argument('--save_viz', type=str, default=None,
                        help='Save visualization to file')
    parser.add_argument('--all_emotions', action='store_true',
                        help='Generate for all emotions')
    
    args = parser.parse_args()
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Create generator
    generator = CaptionGenerator(
        model_path=args.model,
        vocab_path=args.vocab,
        tfidf_path=args.tfidf,
        device=args.device
    )
    
    # Interactive mode
    if args.interactive:
        interactive_demo(generator)
        return
    
    # Single image
    if args.image:
        if args.all_emotions:
            print(f"\nGenerating captions for all emotions:")
            print(f"Image: {args.image}\n")
            
            image = generator.preprocess_image(args.image)
            for emotion in EMOTION_NAMES:
                caption, _ = generator.generate(
                    image, emotion, args.method,
                    beam_width=args.beam_width,
                    temperature=args.temperature
                )
                print(f"  [{emotion}]: {caption}")
            
            if args.visualize or args.save_viz:
                visualize_multi_emotion(
                    generator, args.image,
                    save_path=args.save_viz,
                    show=args.visualize
                )
        else:
            caption = generator.generate_from_path(
                args.image, args.emotion, args.method,
                beam_width=args.beam_width,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p
            )
            
            print(f"\nImage: {args.image}")
            print(f"Emotion: {args.emotion}")
            print(f"Caption: {caption}\n")
            
            if args.visualize or args.save_viz:
                visualize_caption(
                    args.image, caption, args.emotion,
                    save_path=args.save_viz,
                    show=args.visualize
                )
        return
    
    # Batch inference
    if args.image_folder:
        results = batch_inference_folder(
            generator,
            args.image_folder,
            emotion=args.emotion,
            output_json=args.output,
            method=args.method
        )
        
        # Print sample results
        print("\nSample results:")
        for filename, caption in list(results.items())[:5]:
            print(f"  {filename}: {caption}")
        return
    
    # No input specified
    print("Please specify --image, --image_folder, or --interactive")
    parser.print_help()


if __name__ == '__main__':
    main()
