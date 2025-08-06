#!/usr/bin/env python3
"""
Simplified Data Preparation Demo for MicroLSTM

This script demonstrates basic data preparation for character-level language models.
"""

import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

from micro_lstm import CharacterTokenizer, DataLoader


class TextDataset(Dataset):
    """Simple dataset for character-level language modeling."""
    
    def __init__(self, text: str, tokenizer: CharacterTokenizer, sequence_length: int):
        self.text = text
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        
        # Encode the text
        self.encoded_text = tokenizer.encode(text)
        self.num_sequences = len(self.encoded_text) - sequence_length
        
        print(f"Dataset: {len(text):,} chars, {self.num_sequences:,} sequences")
    
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        # Input and target sequences
        input_seq = self.encoded_text[idx:idx + self.sequence_length]
        target_seq = self.encoded_text[idx + 1:idx + self.sequence_length + 1]
        
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)


def get_sample_text() -> str:
    """Get sample text for demonstration."""
    # Load TinyStories dataset
    loader = DataLoader()
    text, tokenizer, info = loader.quick_setup("roneneldan/TinyStories", preprocess=True)
    return text


def clean_text(text: str) -> str:
    """Basic text cleaning."""
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text


def create_dataloader(text: str, sequence_length: int = 50, batch_size: int = 32) -> DataLoader:
    """Create a DataLoader for training."""
    # Clean text
    text = clean_text(text)
    
    # Create tokenizer
    tokenizer = CharacterTokenizer(text)
    
    # Create dataset
    dataset = TextDataset(text, tokenizer, sequence_length)
    
    # Create dataloader
    dataloader = TorchDataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader, tokenizer


def main():
    """Main demonstration function."""
    print("📊 DATA PREPARATION DEMO")
    print("=" * 40)
    
    # Get sample text
    text = get_sample_text()
    print(f"Sample text length: {len(text)} characters")
    
    # Create dataloader
    dataloader, tokenizer = create_dataloader(text, sequence_length=30, batch_size=4)
    
    print(f"\nVocabulary size: {tokenizer.vocab_size()}")
    print(f"Number of batches: {len(dataloader)}")
    
    # Show a few examples
    print("\nSample batches:")
    for i, (inputs, targets) in enumerate(dataloader):
        if i >= 2:  # Show first 2 batches
            break
        
        print(f"\nBatch {i+1}:")
        print(f"  Input shape: {inputs.shape}")
        print(f"  Target shape: {targets.shape}")
        
        # Decode first sequence
        input_text = tokenizer.decode(inputs[0].tolist())
        target_text = tokenizer.decode(targets[0].tolist())
        print(f"  Input:  '{input_text}'")
        print(f"  Target: '{target_text}'")
    
    print("\n✅ Data preparation complete!")


if __name__ == "__main__":
    main()