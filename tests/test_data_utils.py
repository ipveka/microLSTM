"""
Simplified unit tests for data utilities.

This module contains focused tests for data processing utilities, covering:
- Text preprocessing functionality
- Dataset creation and management
- Data loading and batching
"""

import pytest
import torch
from torch.utils.data import DataLoader
from micro_lstm.data_utils import create_data_loader, TextSequenceDataset
from micro_lstm.tokenizer import CharacterTokenizer


class TestTextSequenceDataset:
    """Test TextSequenceDataset functionality."""
    
    def test_dataset_creation(self):
        """Test basic dataset creation."""
        text = "hello world example text"
        tokenizer = CharacterTokenizer(text)
        sequence_length = 5
        dataset = TextSequenceDataset(text, tokenizer, sequence_length)
        
        assert len(dataset) > 0
        assert dataset.sequence_length == sequence_length
    
    def test_dataset_getitem(self):
        """Test dataset item retrieval."""
        text = "hello world"
        tokenizer = CharacterTokenizer(text)
        sequence_length = 3
        dataset = TextSequenceDataset(text, tokenizer, sequence_length)
        
        # Get first item
        input_seq, target_seq = dataset[0]
        
        assert isinstance(input_seq, torch.Tensor)
        assert isinstance(target_seq, torch.Tensor)
        assert input_seq.shape[0] == sequence_length
        assert target_seq.shape[0] == sequence_length
    
    def test_dataset_length(self):
        """Test dataset length calculation."""
        text = "hello world example"
        tokenizer = CharacterTokenizer(text)
        sequence_length = 5
        dataset = TextSequenceDataset(text, tokenizer, sequence_length)
        
        # Should have at least some sequences
        assert len(dataset) > 0
        assert dataset.sequence_length == sequence_length


class TestCreateDataLoader:
    """Test create_data_loader functionality."""
    
    def test_basic_data_loader_creation(self):
        """Test basic data loader creation."""
        text = "hello world example text for training"
        tokenizer = CharacterTokenizer(text)
        sequence_length = 5
        batch_size = 2
        
        data_loader = create_data_loader(
            text=text,
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            batch_size=batch_size,
            shuffle=True
        )
        
        assert isinstance(data_loader, DataLoader)
        assert data_loader.batch_size == batch_size
        
        # Test iteration
        for batch in data_loader:
            assert len(batch) == 2  # input and target
            assert batch[0].shape[0] <= batch_size
            assert batch[0].shape[1] == sequence_length
            break
    
    def test_data_loader_basic_functionality(self):
        """Test basic data loader functionality."""
        text = "hello world example text for training"
        tokenizer = CharacterTokenizer(text)
        sequence_length = 4
        batch_size = 2
        
        data_loader = create_data_loader(
            text=text,
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            batch_size=batch_size,
            shuffle=True
        )
        
        assert isinstance(data_loader, DataLoader)
        assert data_loader.batch_size == batch_size
        
        # Should have data
        batches = len(list(data_loader))
        assert batches > 0
    
    def test_data_loader_parameters(self):
        """Test data loader with different parameters."""
        text = "hello world"
        tokenizer = CharacterTokenizer(text)
        sequence_length = 3
        batch_size = 1
        
        # Test with shuffle=False
        data_loader = create_data_loader(
            text=text,
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            batch_size=batch_size,
            shuffle=False
        )
        
        assert isinstance(data_loader, DataLoader)
        
        # Test with different batch sizes
        for batch_size in [1, 2, 4]:
            if batch_size <= len(text) - sequence_length + 1:
                data_loader = create_data_loader(
                    text=text,
                    tokenizer=tokenizer,
                    sequence_length=sequence_length,
                    batch_size=batch_size
                )
                assert data_loader.batch_size == batch_size 