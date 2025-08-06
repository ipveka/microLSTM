"""
Model trainer with progress tracking for MicroLSTM.

This module provides the ModelTrainer class that handles the complete training
process for the character-level language model. It includes loss calculation,
optimizer setup, backpropagation, progress reporting, and model persistence.

The trainer is designed to be educational and provides extensive logging and
progress tracking to help understand the training process.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Optional, Tuple, Callable
import time
import os
import json
from pathlib import Path

from .model import MicroLSTM
from .tokenizer import CharacterTokenizer
from .data_utils import create_data_loader, validate_sequence_data
from .exceptions import TrainingError, ModelConfigurationError, FileOperationError, CudaError


class ModelTrainer:
    """
    Comprehensive trainer for MicroLSTM with progress tracking.
    
    This class handles the complete training pipeline including:
    - Loss calculation using CrossEntropyLoss
    - Optimizer setup and backpropagation
    - Progress reporting with loss tracking per epoch
    - Model saving and loading functionality
    - Training statistics and monitoring
    
    The trainer is designed to be educational, providing detailed logging and
    progress information to help understand how neural language models are trained.
    
    Args:
        model (MicroLSTM): The language model to train
        tokenizer (CharacterTokenizer): Tokenizer for text processing
        device (torch.device, optional): Device to run training on (CPU/GPU)
        
    Attributes:
        model (MicroLSTM): The language model being trained
        tokenizer (CharacterTokenizer): Text tokenizer
        device (torch.device): Training device
        criterion (nn.CrossEntropyLoss): Loss function for training
        optimizer (torch.optim.Optimizer): Optimizer for parameter updates
        training_history (Dict): History of training metrics
        
    Example:
        >>> model = MicroLSTM(vocab_size=50, embedding_dim=128, hidden_dim=256, num_layers=2)
        >>> tokenizer = CharacterTokenizer("sample text corpus")
        >>> trainer = ModelTrainer(model, tokenizer)
        >>> history = trainer.train(data_loader, epochs=10, learning_rate=0.001)
    """
    
    def __init__(
        self,
        model: MicroLSTM,
        tokenizer: CharacterTokenizer,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the trainer with model and tokenizer.
        
        Sets up the training environment including device selection, loss function,
        and training history tracking. The trainer automatically detects CUDA
        availability and uses GPU if available.
        
        Args:
            model (MicroLSTM): The language model to train
            tokenizer (CharacterTokenizer): Tokenizer for text processing
            device (torch.device, optional): Device for training. If None, 
                                           automatically selects CUDA if available
        
        Raises:
            TypeError: If model or tokenizer have incorrect types
            ValueError: If model and tokenizer have incompatible vocabulary sizes
        """
        # Validate input types
        if not isinstance(model, MicroLSTM):
            raise ModelConfigurationError(
                f"model must be MicroLSTM instance, got {type(model)}",
                parameter="model",
                value=type(model)
            )
        
        if not isinstance(tokenizer, CharacterTokenizer):
            raise ModelConfigurationError(
                f"tokenizer must be CharacterTokenizer instance, got {type(tokenizer)}",
                parameter="tokenizer",
                value=type(tokenizer)
            )
        
        # Validate compatibility between model and tokenizer
        if model.vocab_size != tokenizer.vocab_size():
            raise ModelConfigurationError(
                f"Model vocab_size ({model.vocab_size}) must match tokenizer vocab_size "
                f"({tokenizer.vocab_size()})",
                parameter="vocab_size_mismatch",
                model_vocab_size=model.vocab_size,
                tokenizer_vocab_size=tokenizer.vocab_size()
            )
        
        self.model = model
        self.tokenizer = tokenizer
        
        # Set up device (GPU if available, otherwise CPU)
        if device is None:
            try:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            except Exception as e:
                raise CudaError(
                    f"Failed to determine available device: {e}",
                    suggestion="Try specifying device explicitly",
                    original_error=str(e)
                )
        else:
            if not isinstance(device, torch.device):
                try:
                    self.device = torch.device(device)
                except Exception as e:
                    raise CudaError(
                        f"Invalid device specification: {device}",
                        device=str(device),
                        original_error=str(e)
                    )
            else:
                self.device = device
        
        # Move model to the selected device
        try:
            self.model.to(self.device)
        except RuntimeError as e:
            if "CUDA" in str(e) or "out of memory" in str(e).lower():
                raise CudaError(
                    f"Failed to move model to device {self.device}: {e}",
                    device=str(self.device),
                    suggestion="Try using CPU or reducing model size",
                    original_error=str(e)
                )
            else:
                raise ModelConfigurationError(
                    f"Failed to move model to device: {e}",
                    parameter="device",
                    value=str(self.device),
                    original_error=str(e)
                )
        
        # Initialize loss function
        # CrossEntropyLoss is standard for classification tasks like next-character prediction
        # It combines LogSoftmax and NLLLoss for numerical stability
        self.criterion = nn.CrossEntropyLoss()
        
        # Initialize optimizer (will be set up in train method)
        self.optimizer: Optional[torch.optim.Optimizer] = None
        
        # Initialize training history for progress tracking
        self.training_history: Dict[str, List[float]] = {
            'train_loss': [],
            'epoch_times': [],
            'learning_rates': []
        }
        
        # Training state
        self.current_epoch = 0
        self.total_epochs = 0
        self.best_loss = float('inf')
        
        print(f"ModelTrainer initialized on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def prepare_data(
        self,
        text: str,
        sequence_length: int,
        batch_size: int = 32,
        stride: int = 1,
        shuffle: bool = True,
        validation_split: float = 0.0
    ) -> DataLoader:
        """
        Prepare training data from text corpus.
        
        This method converts raw text into a DataLoader suitable for training.
        It handles tokenization, sequence creation, and batching automatically.
        
        Args:
            text (str): Input text corpus for training
            sequence_length (int): Length of each training sequence
            batch_size (int): Number of sequences per batch (default: 32)
            stride (int): Step size for sliding window (default: 1)
            shuffle (bool): Whether to shuffle training data (default: True)
            validation_split (float): Fraction of data to use for validation (default: 0.0)
        
        Returns:
            DataLoader: PyTorch DataLoader ready for training
        
        Raises:
            ValueError: If parameters are invalid or text is too short
            TrainingError: If data preparation fails
        
        Example:
            >>> trainer = ModelTrainer(model, tokenizer)
            >>> data_loader = trainer.prepare_data("hello world", sequence_length=5, batch_size=16)
            >>> print(f"Batches: {len(data_loader)}")
        """
        try:
            # Validate input parameters
            if not isinstance(text, str) or len(text) == 0:
                raise TrainingError(
                    "Text must be a non-empty string",
                    text_length=len(text) if isinstance(text, str) else None
                )
            
            if not isinstance(sequence_length, int) or sequence_length <= 0:
                raise TrainingError(
                    f"sequence_length must be a positive integer, got {sequence_length}",
                    sequence_length=sequence_length
                )
            
            if not isinstance(batch_size, int) or batch_size <= 0:
                raise TrainingError(
                    f"batch_size must be a positive integer, got {batch_size}",
                    batch_size=batch_size
                )
            
            if not isinstance(validation_split, (int, float)) or not 0.0 <= validation_split < 1.0:
                raise TrainingError(
                    f"validation_split must be a number in [0.0, 1.0), got {validation_split}",
                    validation_split=validation_split
                )
            
            if len(text) <= sequence_length:
                raise TrainingError(
                    f"Text length ({len(text)}) must be greater than sequence_length ({sequence_length})",
                    text_length=len(text),
                    sequence_length=sequence_length
                )
            
            # Split data if validation is requested
            if validation_split > 0.0:
                split_idx = int(len(text) * (1 - validation_split))
                train_text = text[:split_idx]
                # Store validation text for later use
                self._validation_text = text[split_idx:]
                self._validation_params = {
                    'sequence_length': sequence_length,
                    'batch_size': batch_size,
                    'stride': stride
                }
            else:
                train_text = text
                self._validation_text = None
            
            # Create DataLoader using the utility function
            data_loader = create_data_loader(
                text=train_text,
                tokenizer=self.tokenizer,
                sequence_length=sequence_length,
                batch_size=batch_size,
                stride=stride,
                shuffle=shuffle,
                num_workers=0  # Keep simple for educational purposes
            )
            
            # Store data information for logging
            self._data_info = {
                'text_length': len(train_text),
                'sequence_length': sequence_length,
                'batch_size': batch_size,
                'num_batches': len(data_loader),
                'total_sequences': len(data_loader.dataset),
                'stride': stride
            }
            
            print(f"Data prepared: {len(data_loader)} batches, {len(data_loader.dataset)} sequences")
            print(f"Sequence length: {sequence_length}, Batch size: {batch_size}")
            
            return data_loader
            
        except TrainingError:
            # Re-raise TrainingError as-is to preserve context
            raise
        except Exception as e:
            raise TrainingError(f"Failed to prepare training data: {e}")
    
    def train(
        self,
        data_loader: DataLoader,
        epochs: int,
        learning_rate: float = 0.001,
        optimizer_type: str = 'adam',
        weight_decay: float = 0.0,
        gradient_clip_norm: Optional[float] = 1.0,
        save_every: int = 10,
        save_path: Optional[str] = None,
        progress_callback: Optional[Callable[[int, float, Dict], None]] = None
    ) -> Dict[str, List[float]]:
        """
        Train the language model with comprehensive progress tracking.
        
        This method implements the complete training loop including:
        - Forward pass through the model
        - Loss calculation using CrossEntropyLoss
        - Backpropagation and parameter updates
        - Progress reporting and logging
        - Periodic model saving
        - Training statistics collection
        
        The training process uses teacher forcing, where the model learns to predict
        the next character given the previous characters in the sequence.
        
        Args:
            data_loader (DataLoader): Training data loader
            epochs (int): Number of training epochs
            learning_rate (float): Learning rate for optimizer (default: 0.001)
            optimizer_type (str): Type of optimizer ('adam', 'sgd', 'rmsprop') (default: 'adam')
            weight_decay (float): L2 regularization strength (default: 0.0)
            gradient_clip_norm (float, optional): Gradient clipping norm (default: 1.0)
            save_every (int): Save model every N epochs (default: 10)
            save_path (str, optional): Directory to save models (default: None)
            progress_callback (Callable, optional): Callback for custom progress handling
        
        Returns:
            Dict[str, List[float]]: Training history with losses and metrics
        
        Raises:
            TrainingError: If training fails or encounters errors
            ValueError: If parameters are invalid
        
        Example:
            >>> history = trainer.train(
            ...     data_loader=train_loader,
            ...     epochs=50,
            ...     learning_rate=0.001,
            ...     save_every=10,
            ...     save_path="./models"
            ... )
            >>> print(f"Final loss: {history['train_loss'][-1]:.4f}")
        """
        try:
            # Validate training parameters
            if not isinstance(epochs, int) or epochs <= 0:
                raise TrainingError(
                    f"epochs must be a positive integer, got {epochs}",
                    epochs=epochs
                )
            
            if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
                raise TrainingError(
                    f"learning_rate must be a positive number, got {learning_rate}",
                    learning_rate=learning_rate
                )
            
            if not isinstance(weight_decay, (int, float)) or weight_decay < 0:
                raise TrainingError(
                    f"weight_decay must be a non-negative number, got {weight_decay}",
                    weight_decay=weight_decay
                )
            
            if not isinstance(save_every, int) or save_every <= 0:
                raise TrainingError(
                    f"save_every must be a positive integer, got {save_every}",
                    save_every=save_every
                )
            
            # Set up optimizer
            self.optimizer = self._setup_optimizer(
                optimizer_type, learning_rate, weight_decay
            )
            
            # Set up model saving directory
            if save_path:
                save_dir = Path(save_path)
                save_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize training state
            self.total_epochs = epochs
            self.model.train()  # Set model to training mode
            
            print(f"Starting training for {epochs} epochs...")
            print(f"Optimizer: {optimizer_type}, Learning rate: {learning_rate}")
            print(f"Device: {self.device}, Gradient clipping: {gradient_clip_norm}")
            print("-" * 60)
            
            # Main training loop
            for epoch in range(epochs):
                epoch_start_time = time.time()
                self.current_epoch = epoch + 1
                
                # Train one epoch
                epoch_loss = self._train_epoch(
                    data_loader, gradient_clip_norm, progress_callback
                )
                
                # Calculate epoch duration
                epoch_time = time.time() - epoch_start_time
                
                # Update training history
                self.training_history['train_loss'].append(epoch_loss)
                self.training_history['epoch_times'].append(epoch_time)
                self.training_history['learning_rates'].append(
                    self.optimizer.param_groups[0]['lr']
                )
                
                # Update best loss
                if epoch_loss < self.best_loss:
                    self.best_loss = epoch_loss
                
                # Print progress
                self._print_epoch_progress(epoch + 1, epoch_loss, epoch_time)
                
                # Save model periodically
                if save_path and (epoch + 1) % save_every == 0:
                    checkpoint_path = save_dir / f"model_epoch_{epoch + 1}.pt"
                    self.save_model(str(checkpoint_path))
                
                # Call progress callback if provided
                if progress_callback:
                    progress_callback(epoch + 1, epoch_loss, self.training_history)
            
            # Save final model
            if save_path:
                final_path = save_dir / "model_final.pt"
                self.save_model(str(final_path))
                print(f"Final model saved to: {final_path}")
            
            # Print training summary
            self._print_training_summary()
            
            return self.training_history.copy()
            
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            if save_path:
                interrupt_path = Path(save_path) / f"model_interrupted_epoch_{self.current_epoch}.pt"
                self.save_model(str(interrupt_path))
                print(f"Model saved to: {interrupt_path}")
            return self.training_history.copy()
            
        except TrainingError:
            # Re-raise TrainingError as-is to preserve context
            raise
        except Exception as e:
            raise TrainingError(f"Training failed at epoch {self.current_epoch}: {e}")
    
    def _setup_optimizer(
        self, 
        optimizer_type: str, 
        learning_rate: float, 
        weight_decay: float
    ) -> torch.optim.Optimizer:
        """
        Set up the optimizer for training.
        
        Args:
            optimizer_type (str): Type of optimizer ('adam', 'sgd', 'rmsprop')
            learning_rate (float): Learning rate
            weight_decay (float): L2 regularization strength
        
        Returns:
            torch.optim.Optimizer: Configured optimizer
        
        Raises:
            ValueError: If optimizer_type is not supported
        """
        optimizer_type = optimizer_type.lower()
        
        if optimizer_type == 'adam':
            # Adam optimizer: Adaptive learning rates with momentum
            # Good default choice for most neural networks
            return optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),  # Standard momentum parameters
                eps=1e-8
            )
        elif optimizer_type == 'sgd':
            # Stochastic Gradient Descent: Simple but effective
            # Often works well with proper learning rate scheduling
            return optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=0.9,  # Add momentum for better convergence
                nesterov=True  # Nesterov momentum for improved performance
            )
        elif optimizer_type == 'rmsprop':
            # RMSprop: Good for RNNs and sequence models
            # Adapts learning rates based on recent gradients
            return optim.RMSprop(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                alpha=0.99,  # Smoothing constant
                eps=1e-8,
                momentum=0.9
            )
        else:
            raise TrainingError(
                f"Unsupported optimizer type: {optimizer_type}. "
                f"Supported types: 'adam', 'sgd', 'rmsprop'",
                optimizer_type=optimizer_type
            )
    
    def _train_epoch(
        self,
        data_loader: DataLoader,
        gradient_clip_norm: Optional[float],
        progress_callback: Optional[Callable]
    ) -> float:
        """
        Train the model for one epoch.
        
        This method implements the core training loop for a single epoch:
        1. Iterate through all batches in the data loader
        2. Forward pass: compute model predictions
        3. Loss calculation: compare predictions with targets
        4. Backward pass: compute gradients
        5. Gradient clipping: prevent exploding gradients
        6. Parameter update: apply gradients to model weights
        
        Args:
            data_loader (DataLoader): Training data loader
            gradient_clip_norm (float, optional): Gradient clipping norm
            progress_callback (Callable, optional): Progress callback function
        
        Returns:
            float: Average loss for the epoch
        """
        total_loss = 0.0
        num_batches = len(data_loader)
        
        for batch_idx, (input_sequences, target_sequences) in enumerate(data_loader):
            try:
                # Move data to the training device (GPU/CPU)
                input_sequences = input_sequences.to(self.device)
                target_sequences = target_sequences.to(self.device)
            except RuntimeError as e:
                if "CUDA" in str(e) or "out of memory" in str(e).lower():
                    raise CudaError(
                        f"Failed to move batch {batch_idx} to device {self.device}: {e}",
                        device=str(self.device),
                        batch=batch_idx,
                        suggestion="Try reducing batch_size or using CPU",
                        original_error=str(e)
                    )
                else:
                    raise TrainingError(
                        f"Failed to move data to device: {e}",
                        batch=batch_idx,
                        original_error=str(e)
                    )
            
            # Clear gradients from previous iteration
            # PyTorch accumulates gradients, so we need to zero them
            self.optimizer.zero_grad()
            
            try:
                # Forward pass: compute model predictions
                # Input shape: (batch_size, sequence_length)
                # Output shape: (batch_size, sequence_length, vocab_size)
                logits = self.model(input_sequences)
                
                # Reshape for loss calculation
                # CrossEntropyLoss expects: (N, C) for input and (N,) for target
                # where N is batch_size * sequence_length, C is vocab_size
                batch_size, seq_length, vocab_size = logits.shape
                
                # Flatten logits: (batch_size * seq_length, vocab_size)
                logits_flat = logits.view(-1, vocab_size)
                
                # Flatten targets: (batch_size * seq_length,)
                targets_flat = target_sequences.view(-1)
                
                # Calculate loss using CrossEntropyLoss
                # This computes the negative log-likelihood of the correct character
                # at each position in the sequence
                loss = self.criterion(logits_flat, targets_flat)
                
            except RuntimeError as e:
                if "CUDA" in str(e) or "out of memory" in str(e).lower():
                    raise CudaError(
                        f"CUDA error during forward pass at batch {batch_idx}: {e}",
                        device=str(self.device),
                        batch=batch_idx,
                        suggestion="Try reducing batch_size, sequence_length, or model size",
                        original_error=str(e)
                    )
                else:
                    raise TrainingError(
                        f"Error during forward pass at batch {batch_idx}: {e}",
                        batch=batch_idx,
                        original_error=str(e)
                    )
            
            try:
                # Backward pass: compute gradients
                # This calculates how much each parameter should change
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                # This is especially important for RNNs and LSTMs
                if gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        gradient_clip_norm
                    )
                
                # Parameter update: apply gradients to model weights
                # The optimizer uses the computed gradients to update parameters
                self.optimizer.step()
                
            except RuntimeError as e:
                if "CUDA" in str(e) or "out of memory" in str(e).lower():
                    raise CudaError(
                        f"CUDA error during backward pass at batch {batch_idx}: {e}",
                        device=str(self.device),
                        batch=batch_idx,
                        suggestion="Try reducing batch_size or model size",
                        original_error=str(e)
                    )
                else:
                    raise TrainingError(
                        f"Error during backward pass at batch {batch_idx}: {e}",
                        batch=batch_idx,
                        original_error=str(e)
                    )
            
            # Check for NaN or infinite loss
            if torch.isnan(loss) or torch.isinf(loss):
                raise TrainingError(
                    f"Loss became {'NaN' if torch.isnan(loss) else 'infinite'} at batch {batch_idx}",
                    batch=batch_idx,
                    loss_value=loss.item(),
                    suggestion="Try reducing learning_rate or adding gradient clipping"
                )
            
            # Accumulate loss for epoch average
            total_loss += loss.item()
            
            # Optional: print batch progress for very long epochs
            if num_batches > 100 and (batch_idx + 1) % (num_batches // 10) == 0:
                batch_loss = loss.item()
                progress = (batch_idx + 1) / num_batches * 100
                print(f"  Batch {batch_idx + 1}/{num_batches} ({progress:.1f}%) - Loss: {batch_loss:.4f}")
        
        # Return average loss for the epoch
        return total_loss / num_batches
    
    def _print_epoch_progress(self, epoch: int, loss: float, epoch_time: float):
        """
        Print progress information for the current epoch.
        
        Args:
            epoch (int): Current epoch number
            loss (float): Average loss for the epoch
            epoch_time (float): Time taken for the epoch in seconds
        """
        # Calculate additional metrics
        avg_time = sum(self.training_history['epoch_times']) / len(self.training_history['epoch_times'])
        remaining_epochs = self.total_epochs - epoch
        estimated_remaining = remaining_epochs * avg_time
        
        # Format time strings
        epoch_time_str = self._format_time(epoch_time)
        remaining_time_str = self._format_time(estimated_remaining)
        
        # Print epoch information
        print(f"Epoch {epoch:3d}/{self.total_epochs} - "
              f"Loss: {loss:.6f} - "
              f"Time: {epoch_time_str} - "
              f"ETA: {remaining_time_str}")
        
        # Print best loss if this is a new best
        if loss <= self.best_loss:
            print(f"  → New best loss: {loss:.6f}")
    
    def _print_training_summary(self):
        """Print a summary of the training process."""
        total_time = sum(self.training_history['epoch_times'])
        avg_time_per_epoch = total_time / len(self.training_history['epoch_times'])
        
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        print(f"Total epochs: {len(self.training_history['train_loss'])}")
        print(f"Final loss: {self.training_history['train_loss'][-1]:.6f}")
        print(f"Best loss: {self.best_loss:.6f}")
        print(f"Total training time: {self._format_time(total_time)}")
        print(f"Average time per epoch: {self._format_time(avg_time_per_epoch)}")
        
        # Loss improvement
        if len(self.training_history['train_loss']) > 1:
            initial_loss = self.training_history['train_loss'][0]
            final_loss = self.training_history['train_loss'][-1]
            improvement = ((initial_loss - final_loss) / initial_loss) * 100
            print(f"Loss improvement: {improvement:.2f}%")
        
        print("=" * 60)
    
    def _format_time(self, seconds: float) -> str:
        """Format time in seconds to a readable string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.1f}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours}h {minutes}m {secs:.1f}s"
    
    def save_model(self, filepath: str, include_optimizer: bool = True) -> None:
        """
        Save the trained model and training state to disk.
        
        This method saves a comprehensive checkpoint that includes:
        - Model state dict (trained parameters)
        - Model configuration (architecture details)
        - Tokenizer vocabulary
        - Training history and statistics
        - Optimizer state (optional, for resuming training)
        
        Args:
            filepath (str): Path where the model should be saved
            include_optimizer (bool): Whether to save optimizer state (default: True)
        
        Raises:
            IOError: If the file cannot be written
            TrainingError: If saving fails
        
        Example:
            >>> trainer.save_model("./models/my_model.pt")
            >>> print("Model saved successfully")
        """
        try:
            # Prepare checkpoint data
            checkpoint = {
                # Model information
                'model_state_dict': self.model.state_dict(),
                'model_config': {
                    'vocab_size': self.model.vocab_size,
                    'embedding_dim': self.model.embedding_dim,
                    'hidden_dim': self.model.hidden_dim,
                    'num_layers': self.model.num_layers,
                    'dropout': self.model.dropout
                },
                
                # Tokenizer information
                'tokenizer_vocab': self.tokenizer.get_vocab(),
                
                # Training information
                'training_history': self.training_history,
                'current_epoch': self.current_epoch,
                'best_loss': self.best_loss,
                
                # Metadata
                'save_timestamp': time.time(),
                'device': str(self.device)
            }
            
            # Include optimizer state if requested and available
            if include_optimizer and self.optimizer is not None:
                checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
                checkpoint['optimizer_type'] = type(self.optimizer).__name__
            
            # Include data information if available
            if hasattr(self, '_data_info'):
                checkpoint['data_info'] = self._data_info
            
            # Validate filepath
            if not isinstance(filepath, str) or not filepath.strip():
                raise FileOperationError(
                    "Filepath must be a non-empty string",
                    filepath=filepath,
                    operation="save_model"
                )
            
            # Create directory if it doesn't exist
            try:
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
            except (OSError, PermissionError) as e:
                raise FileOperationError(
                    f"Failed to create directory for {filepath}: {e}",
                    filepath=filepath,
                    operation="create_directory",
                    original_error=str(e)
                )
            
            # Save checkpoint
            torch.save(checkpoint, filepath)
            
            print(f"Model saved to: {filepath}")
            
        except FileOperationError:
            # Re-raise file operation errors as-is
            raise
        except Exception as e:
            raise FileOperationError(
                f"Failed to save model to {filepath}: {e}",
                filepath=filepath,
                operation="save_model",
                original_error=str(e)
            )
    
    def load_model(self, filepath: str, load_optimizer: bool = True) -> None:
        """
        Load a trained model and training state from disk.
        
        This method loads a previously saved checkpoint and restores:
        - Model parameters (trained weights)
        - Training history and statistics
        - Optimizer state (optional, for resuming training)
        
        Args:
            filepath (str): Path to the saved model file
            load_optimizer (bool): Whether to load optimizer state (default: True)
        
        Raises:
            FileNotFoundError: If the model file doesn't exist
            TrainingError: If loading fails or model is incompatible
        
        Example:
            >>> trainer.load_model("./models/my_model.pt")
            >>> print("Model loaded successfully")
        """
        try:
            # Validate filepath
            if not isinstance(filepath, str) or not filepath.strip():
                raise FileOperationError(
                    "Filepath must be a non-empty string",
                    filepath=filepath,
                    operation="load_model"
                )
            
            if not os.path.exists(filepath):
                raise FileOperationError(
                    f"Model file not found: {filepath}",
                    filepath=filepath,
                    operation="load_model"
                )
            
            # Load checkpoint
            try:
                checkpoint = torch.load(filepath, map_location=self.device)
            except Exception as e:
                raise FileOperationError(
                    f"Failed to load checkpoint from {filepath}: {e}",
                    filepath=filepath,
                    operation="load_checkpoint",
                    original_error=str(e)
                )
            
            # Validate checkpoint format
            if not isinstance(checkpoint, dict):
                raise FileOperationError(
                    f"Invalid checkpoint format: expected dictionary, got {type(checkpoint)}",
                    filepath=filepath,
                    operation="validate_checkpoint"
                )
            
            required_keys = ['model_state_dict', 'model_config', 'tokenizer_vocab']
            for key in required_keys:
                if key not in checkpoint:
                    raise FileOperationError(
                        f"Invalid checkpoint format: missing '{key}'",
                        filepath=filepath,
                        operation="validate_checkpoint",
                        missing_key=key
                    )
            
            # Validate model compatibility
            model_config = checkpoint['model_config']
            if model_config['vocab_size'] != self.model.vocab_size:
                raise ModelConfigurationError(
                    f"Model vocab_size mismatch: expected {self.model.vocab_size}, "
                    f"got {model_config['vocab_size']}",
                    parameter="vocab_size",
                    expected=self.model.vocab_size,
                    actual=model_config['vocab_size']
                )
            
            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load training history if available
            if 'training_history' in checkpoint:
                self.training_history = checkpoint['training_history']
            
            # Load training state if available
            if 'current_epoch' in checkpoint:
                self.current_epoch = checkpoint['current_epoch']
            
            if 'best_loss' in checkpoint:
                self.best_loss = checkpoint['best_loss']
            
            # Load optimizer state if requested and available
            if load_optimizer and 'optimizer_state_dict' in checkpoint:
                if self.optimizer is not None:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                else:
                    print("Warning: Optimizer state found but no optimizer initialized")
            
            # Load data info if available
            if 'data_info' in checkpoint:
                self._data_info = checkpoint['data_info']
            
            print(f"Model loaded from: {filepath}")
            if 'save_timestamp' in checkpoint:
                save_time = time.ctime(checkpoint['save_timestamp'])
                print(f"Model was saved on: {save_time}")
            
        except (FileOperationError, ModelConfigurationError):
            # Re-raise file and configuration errors as-is
            raise
        except Exception as e:
            raise TrainingError(f"Failed to load model from {filepath}: {e}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive training statistics and information.
        
        Returns:
            Dict[str, Any]: Dictionary containing training statistics including:
                - loss_stats: Loss-related statistics
                - time_stats: Training time statistics
                - model_info: Model architecture information
                - training_progress: Progress indicators
        
        Example:
            >>> stats = trainer.get_training_stats()
            >>> print(f"Average loss: {stats['loss_stats']['average_loss']:.4f}")
        """
        if not self.training_history['train_loss']:
            return {'message': 'No training history available'}
        
        losses = self.training_history['train_loss']
        times = self.training_history['epoch_times']
        
        return {
            'loss_stats': {
                'current_loss': losses[-1] if losses else None,
                'best_loss': self.best_loss,
                'average_loss': sum(losses) / len(losses) if losses else None,
                'loss_std': torch.tensor(losses).std().item() if len(losses) > 1 else 0,
                'loss_trend': 'decreasing' if len(losses) > 1 and losses[-1] < losses[0] else 'stable'
            },
            'time_stats': {
                'total_training_time': sum(times),
                'average_epoch_time': sum(times) / len(times) if times else None,
                'fastest_epoch': min(times) if times else None,
                'slowest_epoch': max(times) if times else None
            },
            'model_info': self.model.get_model_info(),
            'training_progress': {
                'epochs_completed': len(losses),
                'current_epoch': self.current_epoch,
                'total_epochs': self.total_epochs
            },
            'data_info': getattr(self, '_data_info', {}),
            'device': str(self.device)
        }
    
    def reset_training_history(self) -> None:
        """
        Reset training history and statistics.
        
        This method clears all training history while keeping the model
        and optimizer intact. Useful for starting fresh training runs
        or clearing history after loading a model.
        """
        self.training_history = {
            'train_loss': [],
            'epoch_times': [],
            'learning_rates': []
        }
        self.current_epoch = 0
        self.total_epochs = 0
        self.best_loss = float('inf')
        
        print("Training history reset")
    
    def __repr__(self) -> str:
        """String representation of the trainer."""
        return (f"ModelTrainer(model={self.model.__class__.__name__}, "
                f"device={self.device}, "
                f"epochs_trained={len(self.training_history['train_loss'])})")