"""
Comprehensive tests for the ModelTrainer class.

This module tests all aspects of the training functionality including:
- Trainer initialization and setup
- Data preparation and validation
- Training loop execution
- Loss calculation and optimization
- Progress tracking and reporting
- Model saving and loading
- Error handling and edge cases
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from micro_lm.trainer import ModelTrainer, TrainingError
from micro_lm.model import MicroLM
from micro_lm.tokenizer import CharacterTokenizer


class TestModelTrainerInitialization:
    """Test trainer initialization and setup."""
    
    def test_trainer_initialization_success(self):
        """Test successful trainer initialization."""
        # Create test components
        tokenizer = CharacterTokenizer("hello world")
        model = MicroLM(
            vocab_size=tokenizer.vocab_size(),
            embedding_dim=32,
            hidden_dim=64,
            num_layers=1
        )
        
        # Initialize trainer
        trainer = ModelTrainer(model, tokenizer)
        
        # Verify initialization
        assert trainer.model is model
        assert trainer.tokenizer is tokenizer
        assert isinstance(trainer.device, torch.device)
        assert isinstance(trainer.criterion, nn.CrossEntropyLoss)
        assert trainer.optimizer is None  # Not set until training
        assert trainer.current_epoch == 0
        assert trainer.best_loss == float('inf')
        assert 'train_loss' in trainer.training_history
        assert 'epoch_times' in trainer.training_history
        assert 'learning_rates' in trainer.training_history
    
    def test_trainer_initialization_with_device(self):
        """Test trainer initialization with specific device."""
        tokenizer = CharacterTokenizer("test")
        model = MicroLM(vocab_size=tokenizer.vocab_size(), embedding_dim=16, hidden_dim=32, num_layers=1)
        device = torch.device("cpu")
        
        trainer = ModelTrainer(model, tokenizer, device=device)
        
        assert trainer.device == device
        assert next(trainer.model.parameters()).device == device
    
    def test_trainer_initialization_invalid_model(self):
        """Test trainer initialization with invalid model type."""
        tokenizer = CharacterTokenizer("test")
        
        with pytest.raises(TypeError, match="model must be MicroLM instance"):
            ModelTrainer("not_a_model", tokenizer)
    
    def test_trainer_initialization_invalid_tokenizer(self):
        """Test trainer initialization with invalid tokenizer type."""
        model = MicroLM(vocab_size=10, embedding_dim=16, hidden_dim=32, num_layers=1)
        
        with pytest.raises(TypeError, match="tokenizer must be CharacterTokenizer instance"):
            ModelTrainer(model, "not_a_tokenizer")
    
    def test_trainer_initialization_vocab_size_mismatch(self):
        """Test trainer initialization with mismatched vocabulary sizes."""
        tokenizer = CharacterTokenizer("hello")
        model = MicroLM(vocab_size=100, embedding_dim=16, hidden_dim=32, num_layers=1)  # Wrong vocab size
        
        with pytest.raises(ValueError, match="Model vocab_size .* must match tokenizer vocab_size"):
            ModelTrainer(model, tokenizer)


class TestDataPreparation:
    """Test data preparation functionality."""
    
    def setup_method(self):
        """Set up test components for each test."""
        self.tokenizer = CharacterTokenizer("hello world example text")
        self.model = MicroLM(
            vocab_size=self.tokenizer.vocab_size(),
            embedding_dim=32,
            hidden_dim=64,
            num_layers=1
        )
        self.trainer = ModelTrainer(self.model, self.tokenizer)
    
    def test_prepare_data_success(self):
        """Test successful data preparation."""
        text = "hello world example"
        data_loader = self.trainer.prepare_data(
            text=text,
            sequence_length=5,
            batch_size=2
        )
        
        assert isinstance(data_loader, DataLoader)
        assert len(data_loader) > 0
        assert hasattr(self.trainer, '_data_info')
        assert self.trainer._data_info['sequence_length'] == 5
        assert self.trainer._data_info['batch_size'] == 2
        
        # Test that we can iterate through the data
        for batch_inputs, batch_targets in data_loader:
            assert batch_inputs.shape[1] == 5  # sequence_length
            assert batch_targets.shape[1] == 5  # sequence_length
            assert batch_inputs.dtype == torch.long
            assert batch_targets.dtype == torch.long
            break  # Just test first batch
    
    def test_prepare_data_with_validation_split(self):
        """Test data preparation with validation split."""
        text = "hello world example"  # Use only characters in tokenizer vocab
        data_loader = self.trainer.prepare_data(
            text=text,
            sequence_length=5,
            batch_size=2,
            validation_split=0.2
        )
        
        assert isinstance(data_loader, DataLoader)
        assert hasattr(self.trainer, '_validation_text')
        assert self.trainer._validation_text is not None
        assert len(self.trainer._validation_text) > 0
        assert hasattr(self.trainer, '_validation_params')
    
    def test_prepare_data_invalid_text(self):
        """Test data preparation with invalid text."""
        with pytest.raises(TrainingError, match="Text must be a non-empty string"):
            self.trainer.prepare_data("", sequence_length=5)
        
        with pytest.raises(TrainingError, match="Text must be a non-empty string"):
            self.trainer.prepare_data(None, sequence_length=5)
    
    def test_prepare_data_invalid_sequence_length(self):
        """Test data preparation with invalid sequence length."""
        with pytest.raises(TrainingError, match="sequence_length must be a positive integer"):
            self.trainer.prepare_data("hello", sequence_length=0)
        
        with pytest.raises(TrainingError, match="sequence_length must be a positive integer"):
            self.trainer.prepare_data("hello", sequence_length=-1)
    
    def test_prepare_data_text_too_short(self):
        """Test data preparation with text shorter than sequence length."""
        with pytest.raises(TrainingError, match="Text length .* must be greater than sequence_length"):
            self.trainer.prepare_data("hi", sequence_length=5)
    
    def test_prepare_data_invalid_batch_size(self):
        """Test data preparation with invalid batch size."""
        with pytest.raises(TrainingError, match="batch_size must be a positive integer"):
            self.trainer.prepare_data("hello world", sequence_length=3, batch_size=0)
    
    def test_prepare_data_invalid_validation_split(self):
        """Test data preparation with invalid validation split."""
        with pytest.raises(TrainingError, match="validation_split must be in"):
            self.trainer.prepare_data("hello world", sequence_length=3, validation_split=1.5)


class TestTrainingLoop:
    """Test the main training functionality."""
    
    def setup_method(self):
        """Set up test components for each test."""
        self.tokenizer = CharacterTokenizer("hello world example text for training")
        self.model = MicroLM(
            vocab_size=self.tokenizer.vocab_size(),
            embedding_dim=16,
            hidden_dim=32,
            num_layers=1
        )
        self.trainer = ModelTrainer(self.model, self.tokenizer)
        
        # Create test data loader
        self.data_loader = self.trainer.prepare_data(
            text="hello world example text for training",
            sequence_length=5,
            batch_size=2
        )
    
    def test_train_basic_functionality(self):
        """Test basic training functionality."""
        # Train for a few epochs
        history = self.trainer.train(
            data_loader=self.data_loader,
            epochs=3,
            learning_rate=0.01
        )
        
        # Verify training history
        assert isinstance(history, dict)
        assert 'train_loss' in history
        assert 'epoch_times' in history
        assert 'learning_rates' in history
        assert len(history['train_loss']) == 3
        assert len(history['epoch_times']) == 3
        assert len(history['learning_rates']) == 3
        
        # Verify trainer state
        assert self.trainer.current_epoch == 3
        assert self.trainer.optimizer is not None
        assert self.trainer.best_loss < float('inf')
        
        # Verify losses are reasonable (not NaN or infinite)
        for loss in history['train_loss']:
            assert not torch.isnan(torch.tensor(loss))
            assert not torch.isinf(torch.tensor(loss))
            assert loss > 0  # Loss should be positive
    
    def test_train_different_optimizers(self):
        """Test training with different optimizer types."""
        optimizers = ['adam', 'sgd', 'rmsprop']
        
        for opt_type in optimizers:
            # Reset trainer for each optimizer
            trainer = ModelTrainer(self.model, self.tokenizer)
            data_loader = trainer.prepare_data(
                text="hello world example",  # Use only characters in tokenizer vocab
                sequence_length=3,
                batch_size=2
            )
            
            history = trainer.train(
                data_loader=data_loader,
                epochs=2,
                learning_rate=0.01,
                optimizer_type=opt_type
            )
            
            assert len(history['train_loss']) == 2
            assert trainer.optimizer is not None
            assert opt_type.lower() in str(type(trainer.optimizer)).lower()
    
    def test_train_with_gradient_clipping(self):
        """Test training with gradient clipping."""
        history = self.trainer.train(
            data_loader=self.data_loader,
            epochs=2,
            learning_rate=0.01,
            gradient_clip_norm=0.5
        )
        
        assert len(history['train_loss']) == 2
        # Training should complete without errors
    
    def test_train_with_weight_decay(self):
        """Test training with weight decay (L2 regularization)."""
        history = self.trainer.train(
            data_loader=self.data_loader,
            epochs=2,
            learning_rate=0.01,
            weight_decay=0.01
        )
        
        assert len(history['train_loss']) == 2
        # Training should complete without errors
    
    def test_train_invalid_epochs(self):
        """Test training with invalid epoch count."""
        with pytest.raises(TrainingError, match="epochs must be a positive integer"):
            self.trainer.train(self.data_loader, epochs=0)
        
        with pytest.raises(TrainingError, match="epochs must be a positive integer"):
            self.trainer.train(self.data_loader, epochs=-1)
    
    def test_train_invalid_learning_rate(self):
        """Test training with invalid learning rate."""
        with pytest.raises(TrainingError, match="learning_rate must be a positive number"):
            self.trainer.train(self.data_loader, epochs=1, learning_rate=0)
        
        with pytest.raises(TrainingError, match="learning_rate must be a positive number"):
            self.trainer.train(self.data_loader, epochs=1, learning_rate=-0.01)
    
    def test_train_invalid_optimizer_type(self):
        """Test training with invalid optimizer type."""
        with pytest.raises(TrainingError, match="Unsupported optimizer type"):
            self.trainer.train(
                self.data_loader,
                epochs=1,
                optimizer_type="invalid_optimizer"
            )
    
    def test_train_with_progress_callback(self):
        """Test training with progress callback."""
        callback_calls = []
        
        def progress_callback(epoch, loss, history):
            callback_calls.append((epoch, loss, history))
        
        history = self.trainer.train(
            data_loader=self.data_loader,
            epochs=2,
            learning_rate=0.01,
            progress_callback=progress_callback
        )
        
        assert len(callback_calls) == 2
        assert callback_calls[0][0] == 1  # First epoch
        assert callback_calls[1][0] == 2  # Second epoch
        assert isinstance(callback_calls[0][1], float)  # Loss value
        assert isinstance(callback_calls[0][2], dict)  # History dict


class TestModelSaveLoad:
    """Test model saving and loading functionality."""
    
    def setup_method(self):
        """Set up test components for each test."""
        self.tokenizer = CharacterTokenizer("hello world test")
        self.model = MicroLM(
            vocab_size=self.tokenizer.vocab_size(),
            embedding_dim=16,
            hidden_dim=32,
            num_layers=1
        )
        self.trainer = ModelTrainer(self.model, self.tokenizer)
        
        # Train briefly to have some history
        data_loader = self.trainer.prepare_data("hello world test", sequence_length=3, batch_size=2)
        self.trainer.train(data_loader, epochs=2, learning_rate=0.01)
    
    def test_save_model_success(self):
        """Test successful model saving."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, "test_model.pt")
            
            # Save model
            self.trainer.save_model(filepath)
            
            # Verify file exists
            assert os.path.exists(filepath)
            
            # Verify file contents
            checkpoint = torch.load(filepath, map_location='cpu')
            assert 'model_state_dict' in checkpoint
            assert 'model_config' in checkpoint
            assert 'tokenizer_vocab' in checkpoint
            assert 'training_history' in checkpoint
            assert 'current_epoch' in checkpoint
            assert 'best_loss' in checkpoint
    
    def test_save_model_without_optimizer(self):
        """Test saving model without optimizer state."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, "test_model.pt")
            
            self.trainer.save_model(filepath, include_optimizer=False)
            
            checkpoint = torch.load(filepath, map_location='cpu')
            assert 'optimizer_state_dict' not in checkpoint
    
    def test_load_model_success(self):
        """Test successful model loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, "test_model.pt")
            
            # Save original state
            original_loss = self.trainer.best_loss
            original_epoch = self.trainer.current_epoch
            
            # Save model
            self.trainer.save_model(filepath)
            
            # Create new trainer and load model
            new_model = MicroLM(
                vocab_size=self.tokenizer.vocab_size(),
                embedding_dim=16,
                hidden_dim=32,
                num_layers=1
            )
            new_trainer = ModelTrainer(new_model, self.tokenizer)
            new_trainer.train(
                self.trainer.prepare_data("hello world test", sequence_length=3, batch_size=2),
                epochs=1,
                learning_rate=0.01
            )  # Initialize optimizer
            
            # Load model
            new_trainer.load_model(filepath)
            
            # Verify loaded state
            assert new_trainer.best_loss == original_loss
            assert new_trainer.current_epoch == original_epoch
            assert len(new_trainer.training_history['train_loss']) > 0
    
    def test_load_model_file_not_found(self):
        """Test loading model from non-existent file."""
        with pytest.raises(TrainingError, match="Model file not found"):
            self.trainer.load_model("non_existent_file.pt")
    
    def test_load_model_invalid_format(self):
        """Test loading model with invalid checkpoint format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, "invalid_model.pt")
            
            # Save invalid checkpoint
            torch.save({'invalid': 'data'}, filepath)
            
            with pytest.raises(TrainingError, match="Invalid checkpoint format"):
                self.trainer.load_model(filepath)
    
    def test_load_model_vocab_size_mismatch(self):
        """Test loading model with mismatched vocabulary size."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, "test_model.pt")
            
            # Create checkpoint with wrong vocab size
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'model_config': {
                    'vocab_size': 999,  # Wrong size
                    'embedding_dim': 16,
                    'hidden_dim': 32,
                    'num_layers': 1,
                    'dropout': 0.2
                },
                'tokenizer_vocab': self.tokenizer.get_vocab(),
                'training_history': {'train_loss': []},
                'current_epoch': 0,
                'best_loss': float('inf')
            }
            torch.save(checkpoint, filepath)
            
            with pytest.raises(TrainingError, match="Model vocab_size mismatch"):
                self.trainer.load_model(filepath)


class TestTrainingStatistics:
    """Test training statistics and monitoring functionality."""
    
    def setup_method(self):
        """Set up test components for each test."""
        self.tokenizer = CharacterTokenizer("hello world example")
        self.model = MicroLM(
            vocab_size=self.tokenizer.vocab_size(),
            embedding_dim=16,
            hidden_dim=32,
            num_layers=1
        )
        self.trainer = ModelTrainer(self.model, self.tokenizer)
    
    def test_get_training_stats_no_history(self):
        """Test getting training stats with no training history."""
        stats = self.trainer.get_training_stats()
        assert 'message' in stats
        assert stats['message'] == 'No training history available'
    
    def test_get_training_stats_with_history(self):
        """Test getting training stats after training."""
        # Train briefly
        data_loader = self.trainer.prepare_data("hello world example", sequence_length=3, batch_size=2)
        self.trainer.train(data_loader, epochs=3, learning_rate=0.01)
        
        stats = self.trainer.get_training_stats()
        
        # Verify stats structure
        assert 'loss_stats' in stats
        assert 'time_stats' in stats
        assert 'model_info' in stats
        assert 'training_progress' in stats
        assert 'device' in stats
        
        # Verify loss stats
        loss_stats = stats['loss_stats']
        assert 'current_loss' in loss_stats
        assert 'best_loss' in loss_stats
        assert 'average_loss' in loss_stats
        assert 'loss_trend' in loss_stats
        
        # Verify time stats
        time_stats = stats['time_stats']
        assert 'total_training_time' in time_stats
        assert 'average_epoch_time' in time_stats
        
        # Verify training progress
        progress = stats['training_progress']
        assert progress['epochs_completed'] == 3
        assert progress['current_epoch'] == 3
    
    def test_reset_training_history(self):
        """Test resetting training history."""
        # Train briefly
        data_loader = self.trainer.prepare_data("hello world", sequence_length=3, batch_size=2)
        self.trainer.train(data_loader, epochs=2, learning_rate=0.01)
        
        # Verify history exists
        assert len(self.trainer.training_history['train_loss']) == 2
        assert self.trainer.current_epoch == 2
        assert self.trainer.best_loss < float('inf')
        
        # Reset history
        self.trainer.reset_training_history()
        
        # Verify history is cleared
        assert len(self.trainer.training_history['train_loss']) == 0
        assert self.trainer.current_epoch == 0
        assert self.trainer.best_loss == float('inf')


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def setup_method(self):
        """Set up test components for each test."""
        self.tokenizer = CharacterTokenizer("hello world")
        self.model = MicroLM(
            vocab_size=self.tokenizer.vocab_size(),
            embedding_dim=16,
            hidden_dim=32,
            num_layers=1
        )
        self.trainer = ModelTrainer(self.model, self.tokenizer)
    
    def test_training_error_exception(self):
        """Test TrainingError exception."""
        error = TrainingError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)
    
    def test_train_with_empty_data_loader(self):
        """Test training with empty data loader."""
        # Create a very short text that results in no sequences
        with pytest.raises(TrainingError):
            self.trainer.prepare_data("hi", sequence_length=10)  # Text too short
    
    def test_save_model_invalid_path(self):
        """Test saving model to invalid path."""
        # Try to save to a directory that can't be created (permission issue simulation)
        with patch('os.makedirs', side_effect=PermissionError("Permission denied")):
            with pytest.raises(TrainingError, match="Failed to save model"):
                self.trainer.save_model("/invalid/path/model.pt")
    
    def test_model_forward_pass_error_handling(self):
        """Test handling of errors during model forward pass."""
        # Create data loader
        data_loader = self.trainer.prepare_data("hello world", sequence_length=3, batch_size=2)
        
        # Mock model to raise an error during forward pass
        with patch.object(self.model, 'forward', side_effect=RuntimeError("Forward pass error")):
            with pytest.raises(TrainingError, match="Training failed"):
                self.trainer.train(data_loader, epochs=1, learning_rate=0.01)


class TestTrainerStringRepresentation:
    """Test string representation and utility methods."""
    
    def test_trainer_repr(self):
        """Test trainer string representation."""
        tokenizer = CharacterTokenizer("test")
        model = MicroLM(vocab_size=tokenizer.vocab_size(), embedding_dim=16, hidden_dim=32, num_layers=1)
        trainer = ModelTrainer(model, tokenizer)
        
        repr_str = repr(trainer)
        assert "ModelTrainer" in repr_str
        assert "MicroLM" in repr_str
        assert "epochs_trained=0" in repr_str
    
    def test_format_time_method(self):
        """Test time formatting utility method."""
        tokenizer = CharacterTokenizer("test")
        model = MicroLM(vocab_size=tokenizer.vocab_size(), embedding_dim=16, hidden_dim=32, num_layers=1)
        trainer = ModelTrainer(model, tokenizer)
        
        # Test different time formats
        assert trainer._format_time(30) == "30.0s"
        assert trainer._format_time(90) == "1m 30.0s"
        assert trainer._format_time(3661) == "1h 1m 1.0s"


if __name__ == "__main__":
    pytest.main([__file__])