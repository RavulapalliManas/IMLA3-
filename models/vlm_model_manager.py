"""
VLM Model Manager - Save only the best model based on metrics comparison
Creates a dedicated folder for optimized VLM with:
- One best_model.pt file (updated when metrics improve)
- One metrics.json file (tracks best metrics)
- history.json (logs all iterations for analysis)
"""

import os
import json
import torch
import shutil
from datetime import datetime
from typing import Dict, Any, Optional

class VLMModelManager:
    """
    Manages VLM model checkpoints by keeping only the best model based on metrics.
    Compares metrics each iteration and updates if current model is better.
    """
    
    def __init__(self, save_dir: str = None):
        """
        Initialize the model manager.
        
        Args:
            save_dir: Directory to save the best model and metrics
        """
        if save_dir is None:
            # Use flexible path resolution
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            save_dir = os.path.join(project_root, '..', 'results_optimized_vlm')
        
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.best_model_path = os.path.join(save_dir, "best_model.pt")
        self.metrics_path = os.path.join(save_dir, "metrics.json")
        self.history_path = os.path.join(save_dir, "history.json")
        
        # Load existing best metrics if available
        self.best_metrics = self._load_best_metrics()
        self.history = self._load_history()
        
    def _load_best_metrics(self) -> Optional[Dict[str, Any]]:
        """Load the current best metrics from disk."""
        if os.path.exists(self.metrics_path):
            with open(self.metrics_path, 'r') as f:
                return json.load(f)
        return None
    
    def _load_history(self) -> list:
        """Load the history of all iterations."""
        if os.path.exists(self.history_path):
            with open(self.history_path, 'r') as f:
                return json.load(f)
        return []
    
    def _save_metrics(self, metrics: Dict[str, Any]):
        """Save metrics to disk."""
        with open(self.metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
    
    def _save_history(self):
        """Save history to disk."""
        with open(self.history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
    
    def _is_better(self, new_metrics: Dict[str, Any], criterion: str = 'val_loss') -> bool:
        """
        Compare new metrics with best metrics to determine if new model is better.
        
        Args:
            new_metrics: New metrics to compare
            criterion: Primary metric to compare ('val_loss', 'bleu4', 'rouge_l_f1')
        
        Returns:
            True if new metrics are better, False otherwise
        """
        if self.best_metrics is None:
            return True  # First model is always best
        
        # Define whether lower or higher is better for each metric
        lower_is_better = {
            'val_loss': True,
            'train_loss': True,
            'test_loss': True
        }
        higher_is_better = {
            'bleu1': True,
            'bleu2': True,
            'bleu4': True,
            'rouge_l_f1': True,
            'rouge_l_p': True,
            'rouge_l_r': True
        }
        
        # Get criterion value (handle case-insensitive keys)
        new_value = None
        best_value = None
        
        for key, value in new_metrics.items():
            if key.lower().replace('-', '_').replace(' ', '_') == criterion.lower():
                new_value = value
                break
        
        for key, value in self.best_metrics.items():
            if key.lower().replace('-', '_').replace(' ', '_') == criterion.lower():
                best_value = value
                break
        
        if new_value is None or best_value is None:
            print(f"⚠️ Warning: Criterion '{criterion}' not found in metrics")
            return False
        
        # Compare based on criterion type
        if criterion.lower() in lower_is_better:
            return new_value < best_value
        elif criterion.lower() in higher_is_better:
            return new_value > best_value
        else:
            # Default: assume lower is better
            return new_value < best_value
    
    def save_if_best(self, 
                     model: torch.nn.Module, 
                     metrics: Dict[str, Any], 
                     epoch: int,
                     criterion: str = 'val_loss',
                     additional_info: Optional[Dict[str, Any]] = None) -> bool:
        """
        Save model and metrics only if current metrics are better than previous best.
        
        Args:
            model: PyTorch model to save
            metrics: Dictionary of metrics (must include criterion)
            epoch: Current epoch number
            criterion: Primary metric to compare ('val_loss', 'bleu4', etc.)
            additional_info: Optional additional information to store
        
        Returns:
            True if model was saved (is new best), False otherwise
        """
        # Add metadata to metrics
        current_metrics = {
            **metrics,
            'epoch': epoch,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'criterion_used': criterion
        }
        
        if additional_info:
            current_metrics.update(additional_info)
        
        # Add to history regardless of whether it's best
        self.history.append(current_metrics.copy())
        self._save_history()
        
        # Check if this is the best model
        is_best = self._is_better(current_metrics, criterion)
        
        if is_best:
            # Save model
            torch.save(model.state_dict(), self.best_model_path)
            
            # Save metrics
            self.best_metrics = current_metrics
            self._save_metrics(current_metrics)
            
            # Print improvement info
            if len(self.history) > 1:
                print(f"\n{'='*70}")
                print(f"🎉 NEW BEST MODEL! (Improved {criterion})")
                print(f"{'='*70}")
                print(f"   Epoch: {epoch}")
                print(f"   Previous best {criterion}: {self.history[-2].get(criterion, 'N/A')}")
                print(f"   Current {criterion}:       {current_metrics.get(criterion, 'N/A')}")
                print(f"   Model saved to: {self.best_model_path}")
                print(f"{'='*70}\n")
            else:
                print(f"\n✅ First model saved as best (Epoch {epoch})")
            
            return True
        else:
            criterion_val = current_metrics.get(criterion, 'N/A')
            best_val = self.best_metrics.get(criterion, 'N/A')
            print(f"   Current {criterion}: {criterion_val} (Best: {best_val}) - Not saved")
            return False
    
    def load_best_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Load the best saved model.
        
        Args:
            model: Model instance to load weights into
        
        Returns:
            Model with loaded weights
        """
        if not os.path.exists(self.best_model_path):
            raise FileNotFoundError(f"No best model found at {self.best_model_path}")
        
        model.load_state_dict(torch.load(self.best_model_path))
        print(f"✅ Loaded best model from: {self.best_model_path}")
        
        if self.best_metrics:
            print(f"   Best {self.best_metrics.get('criterion_used', 'metric')}: "
                  f"{self.best_metrics.get(self.best_metrics.get('criterion_used', 'val_loss'), 'N/A')}")
            print(f"   From epoch: {self.best_metrics.get('epoch', 'N/A')}")
        
        return model
    
    def get_best_metrics(self) -> Optional[Dict[str, Any]]:
        """Get the current best metrics."""
        return self.best_metrics
    
    def get_history(self) -> list:
        """Get the full history of all iterations."""
        return self.history
    
    def print_summary(self):
        """Print a summary of the best model and training history."""
        print(f"\n{'='*70}")
        print("📊 VLM MODEL MANAGER SUMMARY")
        print(f"{'='*70}")
        print(f"Save Directory: {self.save_dir}")
        print(f"Total Iterations: {len(self.history)}")
        
        if self.best_metrics:
            print(f"\n🏆 Best Model:")
            print(f"   Epoch: {self.best_metrics.get('epoch', 'N/A')}")
            print(f"   Timestamp: {self.best_metrics.get('timestamp', 'N/A')}")
            print(f"   Criterion: {self.best_metrics.get('criterion_used', 'N/A')}")
            
            # Print key metrics
            key_metrics = ['val_loss', 'train_loss', 'bleu4', 'rouge_l_f1']
            print(f"\n   Metrics:")
            for metric in key_metrics:
                for key, value in self.best_metrics.items():
                    if key.lower().replace('-', '_').replace(' ', '_') == metric:
                        print(f"      {key}: {value:.4f}" if isinstance(value, float) else f"      {key}: {value}")
        else:
            print("\n⚠️ No models saved yet")
        
        print(f"{'='*70}\n")


def create_vlm_optimizer_with_manager(save_dir: str = "e:/A3/results_optimized_vlm") -> VLMModelManager:
    """
    Factory function to create a VLM model manager.
    
    Args:
        save_dir: Directory for saving models
    
    Returns:
        Configured VLMModelManager instance
    """
    manager = VLMModelManager(save_dir)
    print(f"📁 VLM Model Manager initialized at: {save_dir}")
    return manager


# Example usage:
if __name__ == "__main__":
    # Example of how to use the VLMModelManager
    print("VLM Model Manager - Example Usage\n")
    
    manager = create_vlm_optimizer_with_manager()
    
    # Simulate training iterations
    print("Simulating training iterations...\n")
    
    # Iteration 1
    print("Iteration 1:")
    metrics1 = {
        'val_loss': 2.5,
        'train_loss': 2.3,
        'bleu4': 0.15,
        'rouge_l_f1': 0.20
    }
    # manager.save_if_best(model, metrics1, epoch=1, criterion='val_loss')
    print(f"Metrics: {metrics1}\n")
    
    # Iteration 2 (improved)
    print("Iteration 2:")
    metrics2 = {
        'val_loss': 2.2,
        'train_loss': 2.1,
        'bleu4': 0.18,
        'rouge_l_f1': 0.22
    }
    print(f"Metrics: {metrics2}\n")
    
    # Iteration 3 (worse)
    print("Iteration 3:")
    metrics3 = {
        'val_loss': 2.4,
        'train_loss': 2.2,
        'bleu4': 0.16,
        'rouge_l_f1': 0.21
    }
    print(f"Metrics: {metrics3} - Would NOT be saved\n")
    
    manager.print_summary()
