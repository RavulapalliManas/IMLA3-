import os
import pickle
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ArtEmisDataset(Dataset):
    def __init__(self, pkl_path, img_folder, split='train', model_type='transformer'):
        """
        Args:
            pkl_path: Path to the .pkl file containing the dataframe.
            img_folder: Folder containing the images.
            split: 'train' for augmentation, 'val' or 'test' for deterministic resizing.
            model_type: 'transformer' (unused but kept for compatibility).
        """
        self.df = pickle.load(open(pkl_path, 'rb'))
        self.img_folder = img_folder
        self.model_type = model_type
        self.split = split
        
        # --- 1. Training Transforms (Aggressive Augmentation) ---
        # "Make 6.5k images look like 65k"
        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)),        # Resize slightly larger than target
            transforms.RandomCrop((224, 224)),    # Randomly crop (forces model to look at details)
            transforms.RandomHorizontalFlip(p=0.5), # Mirroring (Art composition often works mirrored)
            transforms.RandomRotation(degrees=10), # Slight tilt
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Style robustness
            transforms.ToTensor(),
            transforms.Normalize(                 # ImageNet stats
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # --- 2. Validation/Test Transforms (Deterministic) ---
        # No randomness allowed here!
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),        # Direct resize or CenterCrop
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img_name = row['painting']
        img_path = os.path.join(self.img_folder, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            # Apply transforms based on split
            if self.split == 'train':
                image = self.train_transform(image)
            else:
                image = self.val_transform(image)
                
        except Exception as e:
            # print(f"Error loading {img_path}: {e}") # Optional: Comment out to reduce console spam
            # Return blank normalized image on error to prevent crash
            image = torch.zeros(3, 224, 224)
        
        # Get caption tokens
        input_ids = torch.tensor(row['input_ids'], dtype=torch.long)
        target_ids = torch.tensor(row['target_ids'], dtype=torch.long)
        
        # Return filename for object tag lookup (NEW for hybrid model)
        return image, input_ids, target_ids, img_name