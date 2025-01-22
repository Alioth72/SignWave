import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from tqdm import tqdm
import numpy as np

class SignLanguageTrainer:
    """
    Trainer class for sign language translation model
    """
    def __init__(self,
                 model: torch.nn.Module,
                 dataloaders: Dict[str, torch.utils.data.DataLoader],
                 criterion: torch.nn.Module,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 device: str = 'cuda'):
        self.model = model
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.optimizer = optimizer or Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = scheduler or ReduceLROnPlateau(self.optimizer, mode='min', patience=3)
        self.device = device
        self.model.to(self.device)
        
    def train_one_epoch(self, epoch: int) -> float:
        """Train the model for one epoch"""
        self.model.train()
        running_loss = 0.0
        
        for inputs, targets in tqdm(self.dataloaders['train'], desc=f"Epoch {epoch} [Train]"):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(self.dataloaders['train'].dataset)
        logging.info(f"Epoch {epoch} Train Loss: {epoch_loss:.4f}")
        return epoch_loss
    
    def validate_one_epoch(self, epoch: int) -> float:
        """Validate the model for one epoch"""
        self.model.eval()
        running_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.dataloaders['val'], desc=f"Epoch {epoch} [Validation]"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(self.dataloaders['val'].dataset)
        logging.info(f"Epoch {epoch} Validation Loss: {epoch_loss:.4f}")
        return epoch_loss
    
    def train(self, num_epochs: int, save_dir: str = 'checkpoints', save_best_only: bool = True):
        """Train the model for a given number of epochs"""
        Path(save_dir).mkdir(exist_ok=True)
        best_loss = float('inf')
        
        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_one_epoch(epoch)
            val_loss = self.validate_one_epoch(epoch)
            
            self.scheduler.step(val_loss)
            
            # Save model checkpoint
            if not save_best_only or val_loss < best_loss:
                best_loss = val_loss
                checkpoint_path = Path(save_dir) / f'model_epoch_{epoch}.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': val_loss,
                }, checkpoint_path)
                logging.info(f"Saved checkpoint: {checkpoint_path}")
    
    def evaluate(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, Any]:
        """Evaluate the model on a test dataset"""
        self.model.eval()
        total, correct = 0, 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, desc="Evaluation"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                predictions = torch.argmax(outputs, dim=1)
                total += targets.size(0)
                correct += (predictions == targets).sum().item()
        
        accuracy = correct / total
        logging.info(f"Evaluation Accuracy: {accuracy:.4f}")
        return {"accuracy": accuracy}
