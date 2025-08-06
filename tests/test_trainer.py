"""
Simplified unit tests for the ModelTrainer implementation.

This module contains focused tests for the ModelTrainer class, covering:
- Basic trainer initialization and setup
- Data preparation functionality
- Training loop execution
- Model save and load operations
- Error handling for invalid inputs
"""

import pytest
import torch
import tempfile
import os
from micro_lstm.trainer import ModelTrainer
from micro_lstm.model import MicroLSTM
from micro_lstm.tokenizer import CharacterTokenizer
from micro_lstm.exceptions import ModelConfigurationError, TrainingError, FileOperationError


class TestModelTrainerBasic:
    """Test basic trainer functionality."""
    
    def test_basic_initialization(self):
        """Test basic trainer initialization."""
        tokenizer = CharacterTokenizer("hello world")
        model = MicroLSTM(
            vocab_size=tokenizer.vocab_size(),
            embedding_dim=32,
            hidden_dim=64,
            num_layers=1
        )
        
        trainer = ModelTrainer(model, tokenizer)
        
        assert trainer.model is model
        assert trainer.tokenizer is tokenizer
        assert isinstance(trainer.device, torch.device)
        assert trainer.current_epoch == 0
        assert trainer.best_loss == float('inf')
    
    def test_invalid_model_type(self):
        """Test initialization with invalid model type."""
        tokenizer = CharacterTokenizer("hello")
        
        with pytest.raises(ModelConfigurationError):
            ModelTrainer("not_a_model", tokenizer)
    
    def test_invalid_tokenizer_type(self):
        """Test initialization with invalid tokenizer type."""
        model = MicroLSTM(vocab_size=10, embedding_dim=16, hidden_dim=32, num_layers=1)
        
        with pytest.raises(ModelConfigurationError):
            ModelTrainer(model, "not_a_tokenizer")
    
    def test_vocab_size_mismatch(self):
        """Test initialization with mismatched vocabulary sizes."""
        tokenizer = CharacterTokenizer("hello")
        model = MicroLSTM(vocab_size=100, embedding_dim=16, hidden_dim=32, num_layers=1)
        
        with pytest.raises(ModelConfigurationError):
            ModelTrainer(model, tokenizer)


class TestDataPreparation:
    """Test data preparation functionality."""
    
    def setup_method(self):
        """Set up test components for each test."""
        self.tokenizer = CharacterTokenizer("hello world example text")
        self.model = MicroLSTM(
            vocab_size=self.tokenizer.vocab_size(),
            embedding_dim=32,
            hidden_dim=64,
            num_layers=1
        )
        self.trainer = ModelTrainer(self.model, self.tokenizer)
    
    def test_prepare_data_success(self):
        """Test successful data preparation."""
        text = "hello world"
        data_loader = self.trainer.prepare_data(
            text=text,
            sequence_length=3,
            batch_size=2,
            shuffle=True
        )
        
        assert data_loader is not None
        assert hasattr(data_loader, '__iter__')
        
        # Test that we can iterate through the data
        for batch in data_loader:
            assert len(batch) == 2  # input and target
            assert batch[0].shape[0] <= 2  # batch size
            assert batch[0].shape[1] == 3  # sequence length
            break
    
    def test_prepare_data_invalid_text(self):
        """Test data preparation with invalid text."""
        with pytest.raises(TrainingError):
            self.trainer.prepare_data(text="", sequence_length=3, batch_size=2)
    
    def test_prepare_data_invalid_sequence_length(self):
        """Test data preparation with invalid sequence length."""
        with pytest.raises(TrainingError):
            self.trainer.prepare_data(text="hello", sequence_length=0, batch_size=2)
    
    def test_prepare_data_invalid_batch_size(self):
        """Test data preparation with invalid batch size."""
        with pytest.raises(TrainingError):
            self.trainer.prepare_data(text="hello", sequence_length=3, batch_size=0)


class TestTrainingLoop:
    """Test training loop functionality."""
    
    def setup_method(self):
        """Set up test components for each test."""
        self.tokenizer = CharacterTokenizer("hello world example text for training")
        self.model = MicroLSTM(
            vocab_size=self.tokenizer.vocab_size(),
            embedding_dim=32,
            hidden_dim=64,
            num_layers=1
        )
        self.trainer = ModelTrainer(self.model, self.tokenizer)
    
    def test_basic_training(self):
        """Test basic training functionality."""
        text = "hello world"
        data_loader = self.trainer.prepare_data(
            text=text,
            sequence_length=3,
            batch_size=2,
            shuffle=True
        )
        
        # Train for a few epochs
        self.trainer.train(data_loader, epochs=2, learning_rate=0.01)
        
        # Check that training history was updated
        assert len(self.trainer.training_history['train_loss']) > 0
        assert self.trainer.current_epoch > 0
        assert self.trainer.best_loss < float('inf')
    
    def test_training_with_different_optimizers(self):
        """Test training with different optimizer types."""
        text = "hello world"
        data_loader = self.trainer.prepare_data(
            text=text,
            sequence_length=3,
            batch_size=2
        )
        
        # Test with SGD
        self.trainer.train(data_loader, epochs=1, learning_rate=0.01, optimizer_type='sgd')
        
        # Test with Adam
        self.trainer.train(data_loader, epochs=1, learning_rate=0.01, optimizer_type='adam')
    
    def test_training_invalid_epochs(self):
        """Test training with invalid number of epochs."""
        text = "hello world"
        data_loader = self.trainer.prepare_data(
            text=text,
            sequence_length=3,
            batch_size=2
        )
        
        with pytest.raises(TrainingError):
            self.trainer.train(data_loader, epochs=0, learning_rate=0.01)
    
    def test_training_invalid_learning_rate(self):
        """Test training with invalid learning rate."""
        text = "hello world"
        data_loader = self.trainer.prepare_data(
            text=text,
            sequence_length=3,
            batch_size=2
        )
        
        with pytest.raises(TrainingError):
            self.trainer.train(data_loader, epochs=1, learning_rate=-0.01)


class TestModelSaveLoad:
    """Test model save and load functionality."""
    
    def setup_method(self):
        """Set up test components for each test."""
        self.tokenizer = CharacterTokenizer("hello world")
        self.model = MicroLSTM(
            vocab_size=self.tokenizer.vocab_size(),
            embedding_dim=32,
            hidden_dim=64,
            num_layers=1
        )
        self.trainer = ModelTrainer(self.model, self.tokenizer)
    
    def test_save_load_model(self):
        """Test saving and loading model."""
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            model_path = f.name
        
        try:
            # Save model
            self.trainer.save_model(model_path)
            assert os.path.exists(model_path)
            
            # Load model
            loaded_trainer = ModelTrainer(self.model, self.tokenizer)
            loaded_trainer.load_model(model_path)
            
            # Verify model parameters are the same
            for param1, param2 in zip(self.model.parameters(), loaded_trainer.model.parameters()):
                assert torch.allclose(param1, param2)
                
        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)
    
    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file."""
        with pytest.raises(FileOperationError):
            self.trainer.load_model("non_existent_file.pt")
    
    def test_get_training_stats(self):
        """Test getting training statistics."""
        # Train a bit first
        text = "hello world"
        data_loader = self.trainer.prepare_data(
            text=text,
            sequence_length=3,
            batch_size=2
        )
        self.trainer.train(data_loader, epochs=1, learning_rate=0.01)
        
        # Get stats
        stats = self.trainer.get_training_stats()
        
        assert 'loss_stats' in stats
        assert 'time_stats' in stats
        assert 'model_info' in stats
        assert 'training_progress' in stats
        assert stats['loss_stats']['current_loss'] is not None
        assert stats['loss_stats']['best_loss'] < float('inf') 