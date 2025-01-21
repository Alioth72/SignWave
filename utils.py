import torch
import numpy as np
from typing import List, Dict, Any, Optional
import logging
import json
from pathlib import Path
import yaml
from datetime import datetime

class SignLanguageUtils:
    """
    Utility functions for sign language translation
    """
    @staticmethod
    def setup_logging(log_dir: str = 'logs'):
        """Setup logging configuration"""
        Path(log_dir).mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = Path(log_dir) / f'training_{timestamp}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    @staticmethod
    def compute_ctc_loss(predictions: torch.Tensor, 
                        targets: torch.Tensor,
                        pred_lengths: torch.Tensor,
                        target_lengths: torch.Tensor) -> torch.Tensor:
        loss = torch.nn.CTCLoss(blank=0, reduction='mean')
        predictions = predictions.log_softmax(-1).permute(1, 0, 2)
        return loss(predictions, targets, pred_lengths, target_lengths)

    @staticmethod
    def decode_predictions(predictions: torch.Tensor,
                         idx_to_char: Dict[int, str],
                         blank_idx: int = 0) -> List[str]:
        batch_texts = []
        pred_indices = torch.argmax(predictions, dim=2).cpu().numpy()
        
        for sequence in pred_indices:
            text = []
            prev_char = None
            
            for idx in sequence:
                if idx != blank_idx and idx != prev_char:
                    text.append(idx_to_char[idx])
                prev_char = idx
                
            batch_texts.append(''.join(text))
            
        return batch_texts

    @staticmethod
    def save_checkpoint(model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       epoch: int,
                       loss: float,
                       save_dir: str,
                       name: Optional[str] = None):
        """Save model checkpoint"""
        Path(save_dir).mkdir(exist_ok=True)
        
        if name is None:
            name = f'checkpoint_epoch_{epoch}.pt'
            
        checkpoint_path = Path(save_dir) / name
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, checkpoint_path)
        
        logging.info(f'Saved checkpoint to {checkpoint_path}')

    @staticmethod
    def load_checkpoint(checkpoint_path: str,
                       model: torch.nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       device: str = 'cuda'):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        return checkpoint['epoch'], checkpoint['loss']
