"""
Data preparation and batching utilities for MicroLSTM.

This module provides functions to convert raw text into training sequences suitable
for language model training. It implements a sliding window approach to generate
input-target pairs and provides utilities for creating PyTorch DataLoaders with
proper batching.

The main components include:
1. Text-to-sequence conversion using sliding windows
2. Input/target sequence preparation for next-character prediction
3. DataLoader creation with customizable batch sizes
4. Helper functions for data preprocessing and validation
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Iterator, Optional
import numpy as np
from .tokenizer import CharacterTokenizer


class TextSequenceDataset(Dataset):
    """
    PyTorch Dataset for character-level language modeling sequences.
    
    This dataset takes a text corpus and creates training examples using a sliding
    window approach. Each example consists of an input sequence and a target sequence,
    where the target is the input shifted by one position (next character prediction).
    
    For example, with sequence_length=5 and text "hello world":
    - Input:  [h, e, l, l, o]  →  Target: [e, l, l, o, ' ']
    - Input:  [e, l, l, o, ' '] →  Target: [l, l, o, ' ', w]
    - And so on...
    
    Args:
        text (str): The input text corpus to create sequences from
        tokenizer (CharacterTokenizer): Tokenizer to convert text to indices
        sequence_length (int): Length of each training sequence
        stride (int): Step size for sliding window (default: 1 for maximum overlap)
    
    Attributes:
        sequences (torch.Tensor): All input sequences as a tensor
        targets (torch.Tensor): All target sequences as a tensor
    """
    
    def __init__(
        self, 
        text: str, 
        tokenizer: CharacterTokenizer, 
        sequence_length: int,
        stride: int = 1
    ):
        """
        Initialize the dataset by creating sequences from the input text.
        
        The initialization process:
        1. Validates input parameters
        2. Tokenizes the entire text corpus
        3. Creates sliding window sequences
        4. Prepares input-target pairs for training
        """
        # Validate input parameters
        if not isinstance(text, str) or len(text) == 0:
            raise ValueError("Text must be a non-empty string")
        
        if not isinstance(tokenizer, CharacterTokenizer):
            raise TypeError("tokenizer must be a CharacterTokenizer instance")
        
        if sequence_length <= 0:
            raise ValueError(f"sequence_length must be positive, got {sequence_length}")
        
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        
        if len(text) <= sequence_length:
            raise ValueError(
                f"Text length ({len(text)}) must be greater than sequence_length ({sequence_length})"
            )
        
        self.text = text
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.stride = stride
        
        # Tokenize the entire text corpus
        # This converts all characters to their corresponding integer indices
        self.token_indices = tokenizer.encode(text)
        
        # Create sequences using sliding window approach
        self.sequences, self.targets = self._create_sequences()
    
    def _create_sequences(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create input and target sequences using a sliding window approach.
        
        This method implements the core logic for generating training examples:
        1. Slide a window of size sequence_length across the tokenized text
        2. For each window position, create an input sequence
        3. Create the corresponding target sequence (input shifted by 1)
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (input_sequences, target_sequences)
                Both tensors have shape (num_sequences, sequence_length)
        """
        input_sequences = []
        target_sequences = []
        
        # Slide window across the text with specified stride
        # We need sequence_length + 1 tokens to create both input and target
        for i in range(0, len(self.token_indices) - self.sequence_length, self.stride):
            # Extract sequence_length tokens for input
            input_seq = self.token_indices[i:i + self.sequence_length]
            
            # Extract sequence_length tokens starting from i+1 for target
            # This creates the "next character" prediction task
            target_seq = self.token_indices[i + 1:i + self.sequence_length + 1]
            
            input_sequences.append(input_seq)
            target_sequences.append(target_seq)
        
        # Convert lists to PyTorch tensors for efficient processing
        sequences_tensor = torch.tensor(input_sequences, dtype=torch.long)
        targets_tensor = torch.tensor(target_sequences, dtype=torch.long)
        
        return sequences_tensor, targets_tensor
    
    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single training example by index.
        
        Args:
            idx (int): Index of the sequence to retrieve
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (input_sequence, target_sequence)
        """
        if idx >= len(self.sequences):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.sequences)}")
        
        return self.sequences[idx], self.targets[idx]
    
    def get_sequence_info(self) -> dict:
        """
        Get information about the dataset structure.
        
        Returns:
            dict: Information about sequences, including counts and statistics
        """
        return {
            'num_sequences': len(self.sequences),
            'sequence_length': self.sequence_length,
            'stride': self.stride,
            'vocab_size': self.tokenizer.vocab_size(),
            'total_tokens': len(self.token_indices),
            'coverage_ratio': len(self.sequences) * self.stride / len(self.token_indices)
        }


def create_training_sequences(
    text: str, 
    tokenizer: CharacterTokenizer, 
    sequence_length: int,
    stride: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert text into training sequences using a sliding window approach.
    
    This function provides a simple interface for creating training data from text.
    It implements the sliding window technique where each training example consists
    of a sequence of characters and the corresponding target sequence (shifted by 1).
    
    The sliding window approach maximizes the use of training data by creating
    overlapping sequences. For example, with stride=1, every possible subsequence
    of the specified length becomes a training example.
    
    Args:
        text (str): Input text to convert into sequences
        tokenizer (CharacterTokenizer): Tokenizer for text-to-index conversion
        sequence_length (int): Length of each training sequence
        stride (int): Step size for sliding window (default: 1)
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (input_sequences, target_sequences)
            - input_sequences: Shape (num_sequences, sequence_length)
            - target_sequences: Shape (num_sequences, sequence_length)
    
    Example:
        >>> tokenizer = CharacterTokenizer("hello world")
        >>> inputs, targets = create_training_sequences("hello", tokenizer, 3)
        >>> print(inputs.shape)  # torch.Size([3, 3]) for "hel", "ell", "llo"
        >>> print(targets.shape) # torch.Size([3, 3]) for "ell", "llo", "lo?"
    """
    # Create dataset to handle the sequence generation
    dataset = TextSequenceDataset(text, tokenizer, sequence_length, stride)
    
    # Return the generated sequences
    return dataset.sequences, dataset.targets


def prepare_input_target_pairs(
    token_indices: List[int], 
    sequence_length: int
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Prepare input-target pairs from a list of token indices.
    
    This helper function takes a sequence of token indices and creates input-target
    pairs suitable for language model training. Each input sequence is paired with
    a target sequence that represents the "next character" at each position.
    
    Args:
        token_indices (List[int]): List of token indices from tokenized text
        sequence_length (int): Desired length of each sequence
    
    Returns:
        Tuple[List[List[int]], List[List[int]]]: (input_sequences, target_sequences)
    
    Raises:
        ValueError: If token_indices is too short or sequence_length is invalid
    
    Example:
        >>> indices = [1, 2, 3, 4, 5, 6]  # Representing "abcdef"
        >>> inputs, targets = prepare_input_target_pairs(indices, 3)
        >>> print(inputs)   # [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
        >>> print(targets)  # [[2, 3, 4], [3, 4, 5], [4, 5, 6]]
    """
    if not isinstance(token_indices, list):
        raise TypeError("token_indices must be a list")
    
    if len(token_indices) <= sequence_length:
        raise ValueError(
            f"token_indices length ({len(token_indices)}) must be greater than "
            f"sequence_length ({sequence_length})"
        )
    
    if sequence_length <= 0:
        raise ValueError(f"sequence_length must be positive, got {sequence_length}")
    
    input_sequences = []
    target_sequences = []
    
    # Create sequences with sliding window approach
    for i in range(len(token_indices) - sequence_length):
        # Input sequence: tokens from position i to i+sequence_length
        input_seq = token_indices[i:i + sequence_length]
        
        # Target sequence: tokens from position i+1 to i+sequence_length+1
        # This represents the "next character" for each position in input
        target_seq = token_indices[i + 1:i + sequence_length + 1]
        
        input_sequences.append(input_seq)
        target_sequences.append(target_seq)
    
    return input_sequences, target_sequences


def create_data_loader(
    text: str,
    tokenizer: CharacterTokenizer,
    sequence_length: int,
    batch_size: int,
    stride: int = 1,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """
    Create a PyTorch DataLoader for training the language model.
    
    This function provides a complete solution for creating a DataLoader that can
    be used directly in training loops. It handles all the data preparation steps
    and returns a DataLoader with proper batching and shuffling.
    
    The DataLoader yields batches of (input_sequences, target_sequences) where:
    - input_sequences: Shape (batch_size, sequence_length)
    - target_sequences: Shape (batch_size, sequence_length)
    
    Args:
        text (str): Input text corpus for training
        tokenizer (CharacterTokenizer): Tokenizer for text processing
        sequence_length (int): Length of each training sequence
        batch_size (int): Number of sequences per batch
        stride (int): Step size for sliding window (default: 1)
        shuffle (bool): Whether to shuffle the data (default: True)
        num_workers (int): Number of worker processes for data loading (default: 0)
    
    Returns:
        DataLoader: PyTorch DataLoader ready for training
    
    Raises:
        ValueError: If any parameter is invalid
    
    Example:
        >>> tokenizer = CharacterTokenizer("hello world example text")
        >>> data_loader = create_data_loader(
        ...     text="hello world",
        ...     tokenizer=tokenizer,
        ...     sequence_length=5,
        ...     batch_size=2
        ... )
        >>> for batch_inputs, batch_targets in data_loader:
        ...     print(batch_inputs.shape)  # torch.Size([2, 5])
        ...     break
    """
    # Validate parameters
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    
    if num_workers < 0:
        raise ValueError(f"num_workers must be non-negative, got {num_workers}")
    
    # Create dataset
    dataset = TextSequenceDataset(text, tokenizer, sequence_length, stride)
    
    # Create and return DataLoader
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),  # Pin memory if CUDA is available
        drop_last=False  # Keep the last batch even if it's smaller
    )
    
    return data_loader


def validate_sequence_data(
    input_sequences: torch.Tensor,
    target_sequences: torch.Tensor,
    vocab_size: int
) -> bool:
    """
    Validate that sequence data is properly formatted for training.
    
    This function performs comprehensive validation of sequence data to ensure
    it meets the requirements for language model training. It checks tensor
    shapes, data types, value ranges, and consistency between inputs and targets.
    
    Args:
        input_sequences (torch.Tensor): Input sequence tensor
        target_sequences (torch.Tensor): Target sequence tensor  
        vocab_size (int): Size of the vocabulary
    
    Returns:
        bool: True if validation passes
    
    Raises:
        ValueError: If validation fails with detailed error message
        TypeError: If inputs have wrong types
    
    Example:
        >>> inputs = torch.randint(0, 10, (5, 8))  # 5 sequences of length 8
        >>> targets = torch.randint(0, 10, (5, 8))
        >>> is_valid = validate_sequence_data(inputs, targets, vocab_size=10)
        >>> print(is_valid)  # True
    """
    # Type validation
    if not isinstance(input_sequences, torch.Tensor):
        raise TypeError(f"input_sequences must be torch.Tensor, got {type(input_sequences)}")
    
    if not isinstance(target_sequences, torch.Tensor):
        raise TypeError(f"target_sequences must be torch.Tensor, got {type(target_sequences)}")
    
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive integer, got {vocab_size}")
    
    # Shape validation
    if input_sequences.shape != target_sequences.shape:
        raise ValueError(
            f"Input and target shapes must match. "
            f"Got input: {input_sequences.shape}, target: {target_sequences.shape}"
        )
    
    if len(input_sequences.shape) != 2:
        raise ValueError(
            f"Sequences must be 2D tensors (num_sequences, sequence_length). "
            f"Got shape: {input_sequences.shape}"
        )
    
    # Data type validation
    if input_sequences.dtype != torch.long:
        raise ValueError(f"Input sequences must have dtype torch.long, got {input_sequences.dtype}")
    
    if target_sequences.dtype != torch.long:
        raise ValueError(f"Target sequences must have dtype torch.long, got {target_sequences.dtype}")
    
    # Value range validation
    if input_sequences.min() < 0 or input_sequences.max() >= vocab_size:
        raise ValueError(
            f"Input sequence values must be in range [0, {vocab_size-1}]. "
            f"Got range [{input_sequences.min()}, {input_sequences.max()}]"
        )
    
    if target_sequences.min() < 0 or target_sequences.max() >= vocab_size:
        raise ValueError(
            f"Target sequence values must be in range [0, {vocab_size-1}]. "
            f"Got range [{target_sequences.min()}, {target_sequences.max()}]"
        )
    
    return True


def get_data_statistics(
    text: str,
    tokenizer: CharacterTokenizer,
    sequence_length: int,
    stride: int = 1
) -> dict:
    """
    Get comprehensive statistics about the training data.
    
    This function analyzes the input text and provides detailed statistics
    that are useful for understanding the dataset characteristics and making
    informed decisions about training parameters.
    
    Args:
        text (str): Input text corpus
        tokenizer (CharacterTokenizer): Tokenizer for analysis
        sequence_length (int): Sequence length for training
        stride (int): Stride for sequence generation
    
    Returns:
        dict: Comprehensive statistics about the data including:
            - text_stats: Character and token counts
            - sequence_stats: Number of sequences and coverage
            - vocab_stats: Vocabulary size and character distribution
            - memory_stats: Estimated memory usage
    
    Example:
        >>> tokenizer = CharacterTokenizer("hello world")
        >>> stats = get_data_statistics("hello world", tokenizer, 5)
        >>> print(stats['text_stats']['total_chars'])  # 11
    """
    # Basic text statistics
    total_chars = len(text)
    unique_chars = len(set(text))
    token_indices = tokenizer.encode(text)
    
    # Calculate number of sequences that will be generated
    num_sequences = max(0, (len(token_indices) - sequence_length) // stride + 1)
    
    # Character frequency analysis
    char_counts = {}
    for char in text:
        char_counts[char] = char_counts.get(char, 0) + 1
    
    # Most and least common characters
    sorted_chars = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)
    most_common = sorted_chars[:5] if sorted_chars else []
    least_common = sorted_chars[-5:] if sorted_chars else []
    
    # Memory estimation (assuming float32 for model weights)
    sequences_memory_mb = (num_sequences * sequence_length * 8) / (1024 * 1024)  # 8 bytes per long
    
    return {
        'text_stats': {
            'total_chars': total_chars,
            'unique_chars': unique_chars,
            'total_tokens': len(token_indices),
            'avg_char_frequency': total_chars / unique_chars if unique_chars > 0 else 0
        },
        'sequence_stats': {
            'num_sequences': num_sequences,
            'sequence_length': sequence_length,
            'stride': stride,
            'coverage_ratio': (num_sequences * stride) / len(token_indices) if token_indices else 0,
            'total_training_tokens': num_sequences * sequence_length
        },
        'vocab_stats': {
            'vocab_size': tokenizer.vocab_size(),
            'most_common_chars': most_common,
            'least_common_chars': least_common,
            'char_distribution': char_counts
        },
        'memory_stats': {
            'sequences_memory_mb': round(sequences_memory_mb, 2),
            'estimated_batch_memory_mb': lambda batch_size: round(
                (batch_size * sequence_length * 8) / (1024 * 1024), 2
            )
        }
    }