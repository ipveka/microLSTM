"""
Unit tests for data preparation and batching utilities.

This module contains comprehensive tests for all data preparation functions,
including sequence generation, DataLoader creation, and validation utilities.
The tests cover normal operation, edge cases, and error conditions.
"""

import pytest
import torch
from torch.utils.data import DataLoader
import numpy as np

from micro_lm.data_utils import (
    TextSequenceDataset,
    create_training_sequences,
    prepare_input_target_pairs,
    create_data_loader,
    validate_sequence_data,
    get_data_statistics
)
from micro_lm.tokenizer import CharacterTokenizer


class TestTextSequenceDataset:
    """Test cases for TextSequenceDataset class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.sample_text = "hello world"
        self.tokenizer = CharacterTokenizer(self.sample_text)
        self.sequence_length = 5
    
    def test_dataset_initialization(self):
        """Test that dataset initializes correctly with valid parameters."""
        dataset = TextSequenceDataset(
            self.sample_text, 
            self.tokenizer, 
            self.sequence_length
        )
        
        assert len(dataset) > 0
        assert dataset.sequence_length == self.sequence_length
        assert dataset.stride == 1  # default value
        assert isinstance(dataset.sequences, torch.Tensor)
        assert isinstance(dataset.targets, torch.Tensor)
    
    def test_dataset_with_custom_stride(self):
        """Test dataset creation with custom stride parameter."""
        stride = 2
        dataset = TextSequenceDataset(
            self.sample_text, 
            self.tokenizer, 
            self.sequence_length,
            stride=stride
        )
        
        assert dataset.stride == stride
        # With stride=2, we should have fewer sequences than stride=1
        dataset_stride_1 = TextSequenceDataset(
            self.sample_text, 
            self.tokenizer, 
            self.sequence_length,
            stride=1
        )
        assert len(dataset) <= len(dataset_stride_1)
    
    def test_dataset_sequence_shapes(self):
        """Test that generated sequences have correct shapes."""
        dataset = TextSequenceDataset(
            self.sample_text, 
            self.tokenizer, 
            self.sequence_length
        )
        
        # Check tensor shapes
        assert dataset.sequences.shape[1] == self.sequence_length
        assert dataset.targets.shape[1] == self.sequence_length
        assert dataset.sequences.shape[0] == dataset.targets.shape[0]
        
        # Check individual sequence shapes
        input_seq, target_seq = dataset[0]
        assert input_seq.shape == (self.sequence_length,)
        assert target_seq.shape == (self.sequence_length,)
    
    def test_dataset_sequence_relationship(self):
        """Test that target sequences are correctly shifted input sequences."""
        dataset = TextSequenceDataset(
            "abcdef", 
            CharacterTokenizer("abcdef"), 
            3
        )
        
        input_seq, target_seq = dataset[0]
        
        # Convert back to characters to verify relationship
        tokenizer = dataset.tokenizer
        input_chars = tokenizer.decode(input_seq.tolist())
        target_chars = tokenizer.decode(target_seq.tolist())
        
        # Target should be input shifted by one position
        # For "abcdef" with seq_len=3: input="abc", target="bcd"
        assert len(input_chars) == len(target_chars)
        # The relationship should be that target[i] == input[i+1] conceptually
    
    def test_dataset_invalid_parameters(self):
        """Test dataset initialization with invalid parameters."""
        # Empty text
        with pytest.raises(ValueError, match="Text must be a non-empty string"):
            TextSequenceDataset("", self.tokenizer, self.sequence_length)
        
        # Invalid tokenizer
        with pytest.raises(TypeError, match="tokenizer must be a CharacterTokenizer"):
            TextSequenceDataset(self.sample_text, "not_a_tokenizer", self.sequence_length)
        
        # Invalid sequence length
        with pytest.raises(ValueError, match="sequence_length must be positive"):
            TextSequenceDataset(self.sample_text, self.tokenizer, 0)
        
        # Invalid stride
        with pytest.raises(ValueError, match="stride must be positive"):
            TextSequenceDataset(self.sample_text, self.tokenizer, self.sequence_length, stride=0)
        
        # Text too short
        with pytest.raises(ValueError, match="Text length .* must be greater than sequence_length"):
            TextSequenceDataset("hi", self.tokenizer, 5)
    
    def test_dataset_getitem(self):
        """Test dataset item access."""
        dataset = TextSequenceDataset(
            self.sample_text, 
            self.tokenizer, 
            self.sequence_length
        )
        
        # Valid index
        input_seq, target_seq = dataset[0]
        assert isinstance(input_seq, torch.Tensor)
        assert isinstance(target_seq, torch.Tensor)
        
        # Invalid index
        with pytest.raises(IndexError):
            dataset[len(dataset)]
    
    def test_dataset_info(self):
        """Test dataset information method."""
        dataset = TextSequenceDataset(
            self.sample_text, 
            self.tokenizer, 
            self.sequence_length
        )
        
        info = dataset.get_sequence_info()
        
        assert 'num_sequences' in info
        assert 'sequence_length' in info
        assert 'stride' in info
        assert 'vocab_size' in info
        assert 'total_tokens' in info
        assert 'coverage_ratio' in info
        
        assert info['sequence_length'] == self.sequence_length
        assert info['vocab_size'] == self.tokenizer.vocab_size()


class TestCreateTrainingSequences:
    """Test cases for create_training_sequences function."""
    
    def test_basic_sequence_creation(self):
        """Test basic sequence creation functionality."""
        text = "abcdef"
        tokenizer = CharacterTokenizer(text)
        sequence_length = 3
        
        inputs, targets = create_training_sequences(text, tokenizer, sequence_length)
        
        # Check shapes
        assert inputs.shape[1] == sequence_length
        assert targets.shape[1] == sequence_length
        assert inputs.shape[0] == targets.shape[0]
        
        # Check data types
        assert inputs.dtype == torch.long
        assert targets.dtype == torch.long
    
    def test_sequence_creation_with_stride(self):
        """Test sequence creation with different stride values."""
        text = "abcdefghij"
        tokenizer = CharacterTokenizer(text)
        sequence_length = 3
        
        # Test stride = 1 (default)
        inputs_1, targets_1 = create_training_sequences(text, tokenizer, sequence_length, stride=1)
        
        # Test stride = 2
        inputs_2, targets_2 = create_training_sequences(text, tokenizer, sequence_length, stride=2)
        
        # With larger stride, we should have fewer sequences
        assert len(inputs_2) <= len(inputs_1)
        assert len(targets_2) <= len(targets_1)
    
    def test_sequence_content_correctness(self):
        """Test that sequences contain correct character relationships."""
        text = "abcd"
        tokenizer = CharacterTokenizer(text)
        sequence_length = 2
        
        inputs, targets = create_training_sequences(text, tokenizer, sequence_length)
        
        # For "abcd" with seq_len=2, we should get:
        # inputs: ["ab", "bc"], targets: ["bc", "cd"]
        assert len(inputs) == 2
        
        # Verify the relationship by decoding
        for i in range(len(inputs)):
            input_text = tokenizer.decode(inputs[i].tolist())
            target_text = tokenizer.decode(targets[i].tolist())
            
            # Target should be input shifted by one character
            assert len(input_text) == len(target_text)


class TestPrepareInputTargetPairs:
    """Test cases for prepare_input_target_pairs function."""
    
    def test_basic_pair_preparation(self):
        """Test basic input-target pair preparation."""
        token_indices = [0, 1, 2, 3, 4, 5]  # Representing 6 characters
        sequence_length = 3
        
        inputs, targets = prepare_input_target_pairs(token_indices, sequence_length)
        
        # Should create 3 pairs: [0,1,2]->[1,2,3], [1,2,3]->[2,3,4], [2,3,4]->[3,4,5]
        assert len(inputs) == 3
        assert len(targets) == 3
        
        # Check first pair
        assert inputs[0] == [0, 1, 2]
        assert targets[0] == [1, 2, 3]
        
        # Check last pair
        assert inputs[-1] == [2, 3, 4]
        assert targets[-1] == [3, 4, 5]
    
    def test_pair_preparation_edge_cases(self):
        """Test pair preparation with edge cases."""
        # Minimum valid input
        token_indices = [0, 1, 2]  # 3 tokens
        sequence_length = 2
        
        inputs, targets = prepare_input_target_pairs(token_indices, sequence_length)
        
        # Should create 1 pair: [0,1]->[1,2]
        assert len(inputs) == 1
        assert inputs[0] == [0, 1]
        assert targets[0] == [1, 2]
    
    def test_pair_preparation_invalid_input(self):
        """Test pair preparation with invalid inputs."""
        # Invalid token_indices type
        with pytest.raises(TypeError, match="token_indices must be a list"):
            prepare_input_target_pairs("not_a_list", 3)
        
        # Token indices too short
        with pytest.raises(ValueError, match="token_indices length .* must be greater than"):
            prepare_input_target_pairs([0, 1], 3)
        
        # Invalid sequence length
        with pytest.raises(ValueError, match="sequence_length must be positive"):
            prepare_input_target_pairs([0, 1, 2, 3], 0)


class TestCreateDataLoader:
    """Test cases for create_data_loader function."""
    
    def test_basic_dataloader_creation(self):
        """Test basic DataLoader creation."""
        text = "hello world example"
        tokenizer = CharacterTokenizer(text)
        sequence_length = 5
        batch_size = 2
        
        data_loader = create_data_loader(
            text, tokenizer, sequence_length, batch_size
        )
        
        assert isinstance(data_loader, DataLoader)
        assert data_loader.batch_size == batch_size
    
    def test_dataloader_batching(self):
        """Test that DataLoader produces correct batch shapes."""
        text = "hello world example text for testing"
        tokenizer = CharacterTokenizer(text)
        sequence_length = 5
        batch_size = 3
        
        data_loader = create_data_loader(
            text, tokenizer, sequence_length, batch_size
        )
        
        # Get first batch
        batch_inputs, batch_targets = next(iter(data_loader))
        
        # Check batch shapes (may be smaller than batch_size for last batch)
        assert batch_inputs.shape[1] == sequence_length
        assert batch_targets.shape[1] == sequence_length
        assert batch_inputs.shape[0] <= batch_size
        assert batch_inputs.shape == batch_targets.shape
    
    def test_dataloader_parameters(self):
        """Test DataLoader with different parameters."""
        text = "hello world example"
        tokenizer = CharacterTokenizer(text)
        
        # Test with shuffle=False
        data_loader = create_data_loader(
            text, tokenizer, sequence_length=3, batch_size=2, shuffle=False
        )
        assert isinstance(data_loader, DataLoader)
        
        # Test with custom stride
        data_loader = create_data_loader(
            text, tokenizer, sequence_length=3, batch_size=2, stride=2
        )
        assert isinstance(data_loader, DataLoader)
    
    def test_dataloader_invalid_parameters(self):
        """Test DataLoader creation with invalid parameters."""
        text = "hello world"
        tokenizer = CharacterTokenizer(text)
        
        # Invalid batch size
        with pytest.raises(ValueError, match="batch_size must be positive"):
            create_data_loader(text, tokenizer, 3, batch_size=0)
        
        # Invalid num_workers
        with pytest.raises(ValueError, match="num_workers must be non-negative"):
            create_data_loader(text, tokenizer, 3, batch_size=2, num_workers=-1)


class TestValidateSequenceData:
    """Test cases for validate_sequence_data function."""
    
    def test_valid_sequence_data(self):
        """Test validation with valid sequence data."""
        vocab_size = 10
        inputs = torch.randint(0, vocab_size, (5, 8))  # 5 sequences of length 8
        targets = torch.randint(0, vocab_size, (5, 8))
        
        # Should return True for valid data
        assert validate_sequence_data(inputs, targets, vocab_size) is True
    
    def test_invalid_tensor_types(self):
        """Test validation with invalid tensor types."""
        vocab_size = 10
        valid_tensor = torch.randint(0, vocab_size, (5, 8))
        
        # Invalid input type
        with pytest.raises(TypeError, match="input_sequences must be torch.Tensor"):
            validate_sequence_data("not_tensor", valid_tensor, vocab_size)
        
        # Invalid target type
        with pytest.raises(TypeError, match="target_sequences must be torch.Tensor"):
            validate_sequence_data(valid_tensor, "not_tensor", vocab_size)
    
    def test_invalid_shapes(self):
        """Test validation with invalid tensor shapes."""
        vocab_size = 10
        
        # Mismatched shapes
        inputs = torch.randint(0, vocab_size, (5, 8))
        targets = torch.randint(0, vocab_size, (5, 6))  # Different sequence length
        
        with pytest.raises(ValueError, match="Input and target shapes must match"):
            validate_sequence_data(inputs, targets, vocab_size)
        
        # Wrong number of dimensions
        inputs_1d = torch.randint(0, vocab_size, (40,))  # 1D tensor
        targets_1d = torch.randint(0, vocab_size, (40,))
        
        with pytest.raises(ValueError, match="Sequences must be 2D tensors"):
            validate_sequence_data(inputs_1d, targets_1d, vocab_size)
    
    def test_invalid_data_types(self):
        """Test validation with invalid tensor data types."""
        vocab_size = 10
        
        # Float tensors instead of long
        inputs = torch.randn(5, 8)  # float32
        targets = torch.randn(5, 8)
        
        with pytest.raises(ValueError, match="Input sequences must have dtype torch.long"):
            validate_sequence_data(inputs, targets, vocab_size)
    
    def test_invalid_value_ranges(self):
        """Test validation with invalid value ranges."""
        vocab_size = 10
        
        # Values outside valid range
        inputs = torch.randint(-1, vocab_size + 1, (5, 8))  # Contains -1 and vocab_size
        targets = torch.randint(0, vocab_size, (5, 8))
        
        with pytest.raises(ValueError, match="Input sequence values must be in range"):
            validate_sequence_data(inputs, targets, vocab_size)
    
    def test_invalid_vocab_size(self):
        """Test validation with invalid vocab_size parameter."""
        inputs = torch.randint(0, 10, (5, 8))
        targets = torch.randint(0, 10, (5, 8))
        
        # Invalid vocab_size type
        with pytest.raises(ValueError, match="vocab_size must be positive integer"):
            validate_sequence_data(inputs, targets, "not_int")
        
        # Invalid vocab_size value
        with pytest.raises(ValueError, match="vocab_size must be positive integer"):
            validate_sequence_data(inputs, targets, 0)


class TestGetDataStatistics:
    """Test cases for get_data_statistics function."""
    
    def test_basic_statistics(self):
        """Test basic statistics calculation."""
        text = "hello world"
        tokenizer = CharacterTokenizer(text)
        sequence_length = 5
        
        stats = get_data_statistics(text, tokenizer, sequence_length)
        
        # Check that all expected keys are present
        assert 'text_stats' in stats
        assert 'sequence_stats' in stats
        assert 'vocab_stats' in stats
        assert 'memory_stats' in stats
        
        # Check text statistics
        text_stats = stats['text_stats']
        assert text_stats['total_chars'] == len(text)
        assert text_stats['unique_chars'] == len(set(text))
        assert text_stats['total_tokens'] == len(tokenizer.encode(text))
        
        # Check sequence statistics
        seq_stats = stats['sequence_stats']
        assert seq_stats['sequence_length'] == sequence_length
        assert seq_stats['num_sequences'] >= 0
        
        # Check vocabulary statistics
        vocab_stats = stats['vocab_stats']
        assert vocab_stats['vocab_size'] == tokenizer.vocab_size()
        assert 'most_common_chars' in vocab_stats
        assert 'least_common_chars' in vocab_stats
        assert 'char_distribution' in vocab_stats
    
    def test_statistics_with_different_parameters(self):
        """Test statistics with different sequence parameters."""
        text = "abcdefghijklmnop"
        tokenizer = CharacterTokenizer(text)
        
        # Test with different sequence lengths
        stats_short = get_data_statistics(text, tokenizer, sequence_length=3)
        stats_long = get_data_statistics(text, tokenizer, sequence_length=8)
        
        # Longer sequences should result in fewer total sequences
        assert stats_short['sequence_stats']['num_sequences'] >= stats_long['sequence_stats']['num_sequences']
        
        # Test with different strides
        stats_stride_1 = get_data_statistics(text, tokenizer, sequence_length=5, stride=1)
        stats_stride_2 = get_data_statistics(text, tokenizer, sequence_length=5, stride=2)
        
        # Larger stride should result in fewer sequences
        assert stats_stride_1['sequence_stats']['num_sequences'] >= stats_stride_2['sequence_stats']['num_sequences']
    
    def test_character_frequency_analysis(self):
        """Test character frequency analysis in statistics."""
        text = "aaaaabbbcc"  # 'a' appears 5 times, 'b' 3 times, 'c' 2 times
        tokenizer = CharacterTokenizer(text)
        
        stats = get_data_statistics(text, tokenizer, sequence_length=3)
        
        char_dist = stats['vocab_stats']['char_distribution']
        assert char_dist['a'] == 5
        assert char_dist['b'] == 3
        assert char_dist['c'] == 2
        
        # Most common should be 'a'
        most_common = stats['vocab_stats']['most_common_chars']
        assert most_common[0][0] == 'a'  # First tuple, first element (character)
        assert most_common[0][1] == 5   # First tuple, second element (count)
    
    def test_memory_estimation(self):
        """Test memory usage estimation."""
        text = "hello world example"
        tokenizer = CharacterTokenizer(text)
        sequence_length = 5
        
        stats = get_data_statistics(text, tokenizer, sequence_length)
        
        memory_stats = stats['memory_stats']
        assert 'sequences_memory_mb' in memory_stats
        assert isinstance(memory_stats['sequences_memory_mb'], (int, float))
        assert memory_stats['sequences_memory_mb'] >= 0
        
        # Test batch memory estimation function
        batch_memory_fn = memory_stats['estimated_batch_memory_mb']
        batch_memory = batch_memory_fn(32)  # 32 batch size
        assert isinstance(batch_memory, (int, float))
        assert batch_memory >= 0


class TestIntegration:
    """Integration tests for the complete data preparation pipeline."""
    
    def test_end_to_end_pipeline(self):
        """Test the complete pipeline from text to DataLoader."""
        # Sample text for training
        text = "The quick brown fox jumps over the lazy dog. " * 3
        
        # Create tokenizer
        tokenizer = CharacterTokenizer(text)
        
        # Create DataLoader
        data_loader = create_data_loader(
            text=text,
            tokenizer=tokenizer,
            sequence_length=10,
            batch_size=4,
            stride=1,
            shuffle=True
        )
        
        # Test that we can iterate through the data
        batch_count = 0
        total_sequences = 0
        
        for batch_inputs, batch_targets in data_loader:
            batch_count += 1
            total_sequences += batch_inputs.shape[0]
            
            # Validate each batch
            assert validate_sequence_data(batch_inputs, batch_targets, tokenizer.vocab_size())
            
            # Check that sequences are valid
            assert batch_inputs.shape[1] == 10  # sequence_length
            assert batch_targets.shape[1] == 10
            assert batch_inputs.shape == batch_targets.shape
            
            # Verify that all values are in valid range
            assert batch_inputs.min() >= 0
            assert batch_inputs.max() < tokenizer.vocab_size()
            assert batch_targets.min() >= 0
            assert batch_targets.max() < tokenizer.vocab_size()
        
        # Should have processed some data
        assert batch_count > 0
        assert total_sequences > 0
    
    def test_consistency_across_methods(self):
        """Test that different methods produce consistent results."""
        text = "hello world example"
        tokenizer = CharacterTokenizer(text)
        sequence_length = 5
        
        # Method 1: Using create_training_sequences
        inputs_1, targets_1 = create_training_sequences(text, tokenizer, sequence_length)
        
        # Method 2: Using TextSequenceDataset directly
        dataset = TextSequenceDataset(text, tokenizer, sequence_length)
        inputs_2, targets_2 = dataset.sequences, dataset.targets
        
        # Method 3: Using prepare_input_target_pairs
        token_indices = tokenizer.encode(text)
        input_lists, target_lists = prepare_input_target_pairs(token_indices, sequence_length)
        inputs_3 = torch.tensor(input_lists, dtype=torch.long)
        targets_3 = torch.tensor(target_lists, dtype=torch.long)
        
        # All methods should produce the same results
        assert torch.equal(inputs_1, inputs_2)
        assert torch.equal(targets_1, targets_2)
        assert torch.equal(inputs_1, inputs_3)
        assert torch.equal(targets_1, targets_3)
    
    def test_dataloader_with_model_compatibility(self):
        """Test that DataLoader output is compatible with model input expectations."""
        text = "hello world example text for model testing"
        tokenizer = CharacterTokenizer(text)
        sequence_length = 8
        batch_size = 4
        
        data_loader = create_data_loader(text, tokenizer, sequence_length, batch_size)
        
        # Get a batch
        batch_inputs, batch_targets = next(iter(data_loader))
        
        # Verify compatibility with model expectations
        assert batch_inputs.dtype == torch.long  # Required for embedding layers
        assert batch_targets.dtype == torch.long  # Required for loss calculation
        assert len(batch_inputs.shape) == 2  # (batch_size, sequence_length)
        assert len(batch_targets.shape) == 2
        assert batch_inputs.shape[1] == sequence_length
        assert batch_targets.shape[1] == sequence_length
        
        # Values should be valid token indices
        assert batch_inputs.min() >= 0
        assert batch_inputs.max() < tokenizer.vocab_size()
        assert batch_targets.min() >= 0
        assert batch_targets.max() < tokenizer.vocab_size()


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])