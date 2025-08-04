"""
Simplified unit tests for the MicroLM model implementation.

This module contains focused tests for the MicroLM class, covering:
- Basic model initialization and functionality
- Forward pass with correct shapes
- Error handling for invalid inputs
- Essential model properties
"""

import pytest
import torch
import torch.nn as nn
from micro_lstm.model import MicroLM
from micro_lstm.exceptions import ModelConfigurationError


class TestMicroLMBasic:
    """Test basic model functionality."""
    
    def test_basic_initialization(self):
        """Test basic model initialization with valid parameters."""
        model = MicroLM(
            vocab_size=50,
            embedding_dim=128,
            hidden_dim=256,
            num_layers=2
        )
        
        assert model.vocab_size == 50
        assert model.embedding_dim == 128
        assert model.hidden_dim == 256
        assert model.num_layers == 2
        assert model.dropout == 0.2  # default value
        
        # Check that all layers are properly initialized
        assert isinstance(model.embedding, nn.Embedding)
        assert isinstance(model.lstm, nn.LSTM)
        assert isinstance(model.output_projection, nn.Linear)
    
    def test_forward_pass_shapes(self):
        """Test forward pass produces correct output shapes."""
        model = MicroLM(
            vocab_size=30,
            embedding_dim=64,
            hidden_dim=128,
            num_layers=1
        )
        
        batch_size = 4
        seq_length = 10
        input_tensor = torch.randint(0, 30, (batch_size, seq_length))
        
        # Forward pass
        output = model(input_tensor)
        
        # Check output shape: (batch_size, seq_length, vocab_size)
        assert output.shape == (batch_size, seq_length, 30)
        assert not torch.isnan(output).any()
    
    def test_hidden_state_initialization(self):
        """Test hidden state initialization."""
        model = MicroLM(
            vocab_size=20,
            embedding_dim=32,
            hidden_dim=64,
            num_layers=2
        )
        
        batch_size = 3
        hidden = model.init_hidden(batch_size)
        
        # Check hidden state shapes
        h, c = hidden
        assert h.shape == (2, batch_size, 64)  # (num_layers, batch_size, hidden_dim)
        assert c.shape == (2, batch_size, 64)
        assert torch.all(h == 0)  # Should be zero-initialized
        assert torch.all(c == 0)


class TestMicroLMValidation:
    """Test input validation and error handling."""
    
    def test_invalid_vocab_size(self):
        """Test that invalid vocab_size raises ModelConfigurationError."""
        with pytest.raises(ModelConfigurationError):
            MicroLM(vocab_size=0, embedding_dim=64, hidden_dim=128, num_layers=1)
        
        with pytest.raises(ModelConfigurationError):
            MicroLM(vocab_size=-5, embedding_dim=64, hidden_dim=128, num_layers=1)
    
    def test_invalid_embedding_dim(self):
        """Test that invalid embedding_dim raises ModelConfigurationError."""
        with pytest.raises(ModelConfigurationError):
            MicroLM(vocab_size=50, embedding_dim=0, hidden_dim=128, num_layers=1)
    
    def test_invalid_hidden_dim(self):
        """Test that invalid hidden_dim raises ModelConfigurationError."""
        with pytest.raises(ModelConfigurationError):
            MicroLM(vocab_size=50, embedding_dim=64, hidden_dim=0, num_layers=1)
    
    def test_invalid_num_layers(self):
        """Test that invalid num_layers raises ModelConfigurationError."""
        with pytest.raises(ModelConfigurationError):
            MicroLM(vocab_size=50, embedding_dim=64, hidden_dim=128, num_layers=0)
    
    def test_invalid_dropout(self):
        """Test that invalid dropout raises ModelConfigurationError."""
        with pytest.raises(ModelConfigurationError):
            MicroLM(vocab_size=50, embedding_dim=64, hidden_dim=128, num_layers=1, dropout=-0.1)
        
        with pytest.raises(ModelConfigurationError):
            MicroLM(vocab_size=50, embedding_dim=64, hidden_dim=128, num_layers=1, dropout=1.5)


class TestMicroLMInfo:
    """Test model information methods."""
    
    def test_get_model_info(self):
        """Test model info structure."""
        model = MicroLM(
            vocab_size=25,
            embedding_dim=64,
            hidden_dim=128,
            num_layers=2,
            dropout=0.3
        )
        
        info = model.get_model_info()
        
        assert 'architecture' in info
        assert 'parameters' in info
        assert info['architecture']['vocab_size'] == 25
        assert info['architecture']['embedding_dim'] == 64
        assert info['architecture']['hidden_dim'] == 128
        assert info['architecture']['num_layers'] == 2
        assert info['architecture']['dropout'] == 0.3
        assert info['parameters']['total'] > 0
    
    def test_model_repr(self):
        """Test model string representation."""
        model = MicroLM(
            vocab_size=30,
            embedding_dim=64,
            hidden_dim=128,
            num_layers=1
        )
        
        repr_str = repr(model)
        assert "MicroLM" in repr_str
        assert "vocab_size=30" in repr_str
        assert "embedding_dim=64" in repr_str 