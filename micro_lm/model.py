"""
Core LSTM Language Model Implementation

This module contains the MicroLM class, which implements a simple character-level
language model using LSTM (Long Short-Term Memory) networks. The model is designed
for educational purposes with extensive comments explaining each component.

The architecture consists of:
1. Embedding layer: Converts character indices to dense vectors
2. LSTM layers: Process sequences and capture temporal dependencies
3. Output projection: Maps LSTM output to vocabulary probabilities
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any

from .exceptions import ModelConfigurationError, CudaError


class MicroLM(nn.Module):
    """
    Simple LSTM-based language model for character-level text generation.
    
    This model predicts the next character in a sequence given the previous characters.
    It uses an embedding layer to convert character indices to dense vectors, processes
    them through LSTM layers to capture sequential patterns, and projects the output
    to vocabulary probabilities.
    
    Architecture:
        Input (character indices) → Embedding → LSTM → Linear → Output (logits)
    
    Args:
        vocab_size (int): Size of the character vocabulary
        embedding_dim (int): Dimension of character embeddings
        hidden_dim (int): Hidden dimension of LSTM layers
        num_layers (int): Number of LSTM layers to stack
        dropout (float): Dropout probability for regularization (default: 0.2)
    
    Example:
        >>> model = MicroLM(vocab_size=50, embedding_dim=128, hidden_dim=256, num_layers=2)
        >>> input_seq = torch.randint(0, 50, (32, 100))  # batch_size=32, seq_len=100
        >>> output = model(input_seq)  # Shape: (32, 100, 50)
    """
    
    def __init__(
        self, 
        vocab_size: int, 
        embedding_dim: int, 
        hidden_dim: int, 
        num_layers: int,
        dropout: float = 0.2
    ):
        super(MicroLM, self).__init__()
        
        # Store model configuration for inspection
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Validate input parameters
        if not isinstance(vocab_size, int) or vocab_size <= 0:
            raise ModelConfigurationError(
                f"vocab_size must be a positive integer, got {vocab_size}",
                parameter="vocab_size",
                value=vocab_size
            )
        if not isinstance(embedding_dim, int) or embedding_dim <= 0:
            raise ModelConfigurationError(
                f"embedding_dim must be a positive integer, got {embedding_dim}",
                parameter="embedding_dim",
                value=embedding_dim
            )
        if not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ModelConfigurationError(
                f"hidden_dim must be a positive integer, got {hidden_dim}",
                parameter="hidden_dim",
                value=hidden_dim
            )
        if not isinstance(num_layers, int) or num_layers <= 0:
            raise ModelConfigurationError(
                f"num_layers must be a positive integer, got {num_layers}",
                parameter="num_layers",
                value=num_layers
            )
        if not isinstance(dropout, (int, float)) or not 0 <= dropout <= 1:
            raise ModelConfigurationError(
                f"dropout must be a number between 0 and 1, got {dropout}",
                parameter="dropout",
                value=dropout
            )
        
        # Embedding layer: Maps character indices to dense vectors
        # This allows the model to learn meaningful representations for each character
        # rather than treating them as arbitrary integers
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )
        
        # LSTM layers: Process sequences and capture temporal dependencies
        # LSTM is chosen over simple RNN because it can better handle long sequences
        # and avoid vanishing gradient problems
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,  # Only apply dropout between layers
            batch_first=True  # Input shape: (batch, seq, feature)
        )
        
        # Output projection layer: Maps LSTM hidden states to vocabulary logits
        # This linear layer produces unnormalized probabilities (logits) for each
        # character in the vocabulary
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        # Initialize weights using Xavier/Glorot initialization for better training
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize model weights using appropriate initialization schemes.
        
        - Embedding weights: Normal distribution with small variance
        - LSTM weights: Xavier uniform initialization
        - Linear weights: Xavier uniform initialization
        - Biases: Zero initialization
        """
        # Initialize embedding weights
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.1)
        
        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:  # Input-to-hidden weights
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:  # Hidden-to-hidden weights
                nn.init.orthogonal_(param)
            elif 'bias' in name:  # Biases
                nn.init.zeros_(param)
        
        # Initialize output projection weights
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
    
    def forward(self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input tensor of character indices
                             Shape: (batch_size, sequence_length)
            hidden (Tuple[torch.Tensor, torch.Tensor], optional): Initial hidden state
                   If None, will be initialized to zeros
        
        Returns:
            torch.Tensor: Output logits for each position and character
                         Shape: (batch_size, sequence_length, vocab_size)
        
        The forward pass follows these steps:
        1. Convert character indices to embeddings
        2. Process embeddings through LSTM layers
        3. Project LSTM output to vocabulary logits
        """
        # Validate input tensor
        if not isinstance(x, torch.Tensor):
            raise ModelConfigurationError(
                f"Input must be a torch.Tensor, got {type(x)}",
                parameter="input",
                value=type(x)
            )
        
        if x.dim() != 2:
            raise ModelConfigurationError(
                f"Input tensor must be 2-dimensional (batch_size, sequence_length), got shape {x.shape}",
                parameter="input_shape",
                value=x.shape
            )
        
        if x.dtype not in [torch.long, torch.int, torch.int32, torch.int64]:
            raise ModelConfigurationError(
                f"Input tensor must have integer dtype for character indices, got {x.dtype}",
                parameter="input_dtype",
                value=x.dtype
            )
        
        # Check for valid token indices
        if torch.any(x < 0) or torch.any(x >= self.vocab_size):
            invalid_indices = x[(x < 0) | (x >= self.vocab_size)]
            raise ModelConfigurationError(
                f"Input contains invalid token indices. Valid range: 0-{self.vocab_size-1}, "
                f"found: {invalid_indices.unique().tolist()}",
                parameter="token_indices",
                value=invalid_indices.unique().tolist()
            )
        
        batch_size, seq_length = x.shape
        
        # Step 1: Convert character indices to dense embeddings
        # Input shape: (batch_size, seq_length)
        # Output shape: (batch_size, seq_length, embedding_dim)
        embedded = self.embedding(x)
        
        # Step 2: Process embeddings through LSTM layers
        # The LSTM processes the sequence and outputs hidden states for each position
        # along with the final hidden and cell states
        if hidden is None:
            hidden = self.init_hidden(batch_size, device=x.device)
        
        # LSTM forward pass
        # lstm_out shape: (batch_size, seq_length, hidden_dim)
        # hidden shape: (num_layers, batch_size, hidden_dim) for both h and c
        lstm_out, hidden = self.lstm(embedded, hidden)
        
        # Step 3: Project LSTM output to vocabulary logits
        # We apply the linear layer to each position in the sequence
        # Input shape: (batch_size, seq_length, hidden_dim)
        # Output shape: (batch_size, seq_length, vocab_size)
        logits = self.output_projection(lstm_out)
        
        return logits
    
    def init_hidden(self, batch_size: int, device: torch.device = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize hidden and cell states for LSTM.
        
        LSTM requires two hidden states:
        - Hidden state (h): Contains short-term memory
        - Cell state (c): Contains long-term memory
        
        Args:
            batch_size (int): Size of the current batch
            device (torch.device, optional): Device to create tensors on
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (hidden_state, cell_state)
                Both have shape: (num_layers, batch_size, hidden_dim)
        """
        # Validate batch_size
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ModelConfigurationError(
                f"batch_size must be a positive integer, got {batch_size}",
                parameter="batch_size",
                value=batch_size
            )
        
        if device is None:
            try:
                device = next(self.parameters()).device
            except StopIteration:
                raise ModelConfigurationError(
                    "Model has no parameters to determine device from",
                    parameter="device"
                )
        
        # Validate device
        if not isinstance(device, torch.device):
            try:
                device = torch.device(device)
            except Exception as e:
                raise CudaError(
                    f"Invalid device specification: {device}",
                    device=str(device),
                    original_error=str(e)
                )
        
        # Initialize both hidden and cell states to zeros
        # Shape: (num_layers, batch_size, hidden_dim)
        try:
            hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
            cell_state = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        except RuntimeError as e:
            if "CUDA" in str(e) or "out of memory" in str(e).lower():
                raise CudaError(
                    f"Failed to create hidden states on device {device}: {e}",
                    device=str(device),
                    suggestion="Try reducing batch_size or using CPU",
                    batch_size=batch_size,
                    original_error=str(e)
                )
            else:
                raise ModelConfigurationError(
                    f"Failed to create hidden states: {e}",
                    parameter="hidden_state_creation",
                    original_error=str(e)
                )
        
        return (hidden_state, cell_state)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the model architecture and parameters.
        
        This method provides detailed information about the model structure,
        parameter counts, and memory usage for debugging and analysis purposes.
        
        Returns:
            Dict[str, Any]: Dictionary containing model information including:
                - architecture: Model configuration parameters
                - parameters: Parameter counts for each component
                - total_params: Total number of trainable parameters
                - model_size_mb: Approximate memory usage in MB
                - layer_info: Detailed information about each layer
        """
        # Count parameters for each component
        embedding_params = sum(p.numel() for p in self.embedding.parameters())
        lstm_params = sum(p.numel() for p in self.lstm.parameters())
        output_params = sum(p.numel() for p in self.output_projection.parameters())
        total_params = embedding_params + lstm_params + output_params
        
        # Estimate model size in MB (assuming float32 = 4 bytes per parameter)
        model_size_mb = (total_params * 4) / (1024 * 1024)
        
        return {
            'architecture': {
                'vocab_size': self.vocab_size,
                'embedding_dim': self.embedding_dim,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'dropout': self.dropout
            },
            'parameters': {
                'embedding': embedding_params,
                'lstm': lstm_params,
                'output_projection': output_params,
                'total': total_params
            },
            'model_size_mb': round(model_size_mb, 2),
            'layer_info': {
                'embedding': f"Embedding({self.vocab_size}, {self.embedding_dim})",
                'lstm': f"LSTM({self.embedding_dim}, {self.hidden_dim}, num_layers={self.num_layers})",
                'output': f"Linear({self.hidden_dim}, {self.vocab_size})"
            }
        }
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return (f"MicroLM(vocab_size={self.vocab_size}, "
                f"embedding_dim={self.embedding_dim}, "
                f"hidden_dim={self.hidden_dim}, "
                f"num_layers={self.num_layers}, "
                f"dropout={self.dropout})")