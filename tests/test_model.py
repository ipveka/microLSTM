"""
Unit tests for the MicroLM model implementation.

This module contains comprehensive tests for the MicroLM class, covering:
- Model initialization with various parameters
- Forward pass functionality and tensor shapes
- Hidden state initialization
- Model information and inspection utilities
- Error handling for invalid inputs
"""

import pytest
import torch
import torch.nn as nn
from micro_lm.model import MicroLM


class TestMicroLMInitialization:
    """Test model initialization with various parameters."""
    
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
    
    def test_initialization_with_custom_dropout(self):
        """Test initialization with custom dropout value."""
        model = MicroLM(
            vocab_size=30,
            embedding_dim=64,
            hidden_dim=128,
            num_layers=1,
            dropout=0.5
        )
        
        assert model.dropout == 0.5
    
    def test_single_layer_no_dropout(self):
        """Test that single layer LSTM doesn't use dropout."""
        model = MicroLM(
            vocab_size=30,
            embedding_dim=64,
            hidden_dim=128,
            num_layers=1,
            dropout=0.5
        )
        
        # Single layer LSTM should have dropout=0 regardless of input
        assert model.lstm.dropout == 0
    
    def test_multi_layer_with_dropout(self):
        """Test that multi-layer LSTM uses specified dropout."""
        model = MicroLM(
            vocab_size=30,
            embedding_dim=64,
            hidden_dim=128,
            num_layers=3,
            dropout=0.3
        )
        
        assert model.lstm.dropout == 0.3


class TestMicroLMValidation:
    """Test input validation and error handling."""
    
    def test_invalid_vocab_size(self):
        """Test that invalid vocab_size raises ValueError."""
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            MicroLM(vocab_size=0, embedding_dim=64, hidden_dim=128, num_layers=1)
        
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            MicroLM(vocab_size=-5, embedding_dim=64, hidden_dim=128, num_layers=1)
    
    def test_invalid_embedding_dim(self):
        """Test that invalid embedding_dim raises ValueError."""
        with pytest.raises(ValueError, match="embedding_dim must be positive"):
            MicroLM(vocab_size=50, embedding_dim=0, hidden_dim=128, num_layers=1)
        
        with pytest.raises(ValueError, match="embedding_dim must be positive"):
            MicroLM(vocab_size=50, embedding_dim=-10, hidden_dim=128, num_layers=1)
    
    def test_invalid_hidden_dim(self):
        """Test that invalid hidden_dim raises ValueError."""
        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            MicroLM(vocab_size=50, embedding_dim=64, hidden_dim=0, num_layers=1)
        
        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            MicroLM(vocab_size=50, embedding_dim=64, hidden_dim=-20, num_layers=1)
    
    def test_invalid_num_layers(self):
        """Test that invalid num_layers raises ValueError."""
        with pytest.raises(ValueError, match="num_layers must be positive"):
            MicroLM(vocab_size=50, embedding_dim=64, hidden_dim=128, num_layers=0)
        
        with pytest.raises(ValueError, match="num_layers must be positive"):
            MicroLM(vocab_size=50, embedding_dim=64, hidden_dim=128, num_layers=-1)
    
    def test_invalid_dropout(self):
        """Test that invalid dropout values raise ValueError."""
        with pytest.raises(ValueError, match="dropout must be between 0 and 1"):
            MicroLM(vocab_size=50, embedding_dim=64, hidden_dim=128, num_layers=1, dropout=-0.1)
        
        with pytest.raises(ValueError, match="dropout must be between 0 and 1"):
            MicroLM(vocab_size=50, embedding_dim=64, hidden_dim=128, num_layers=1, dropout=1.5)


class TestMicroLMForwardPass:
    """Test forward pass functionality and tensor shapes."""
    
    def test_forward_pass_shapes(self):
        """Test that forward pass produces correct output shapes."""
        model = MicroLM(vocab_size=50, embedding_dim=128, hidden_dim=256, num_layers=2)
        
        # Test with different batch sizes and sequence lengths
        test_cases = [
            (1, 10),    # Single sample, short sequence
            (32, 100),  # Standard batch, medium sequence
            (8, 500),   # Small batch, long sequence
        ]
        
        for batch_size, seq_length in test_cases:
            input_tensor = torch.randint(0, 50, (batch_size, seq_length))
            output = model(input_tensor)
            
            expected_shape = (batch_size, seq_length, 50)  # vocab_size = 50
            assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    
    def test_forward_pass_with_hidden_state(self):
        """Test forward pass with provided hidden state."""
        model = MicroLM(vocab_size=30, embedding_dim=64, hidden_dim=128, num_layers=2)
        batch_size, seq_length = 4, 20
        
        input_tensor = torch.randint(0, 30, (batch_size, seq_length))
        hidden = model.init_hidden(batch_size)
        
        output = model(input_tensor, hidden)
        
        expected_shape = (batch_size, seq_length, 30)
        assert output.shape == expected_shape
    
    def test_forward_pass_gradient_flow(self):
        """Test that gradients flow properly through the model."""
        model = MicroLM(vocab_size=20, embedding_dim=32, hidden_dim=64, num_layers=1)
        
        input_tensor = torch.randint(0, 20, (2, 10))
        target = torch.randint(0, 20, (2, 10))
        
        output = model(input_tensor)
        loss = nn.CrossEntropyLoss()(output.view(-1, 20), target.view(-1))
        loss.backward()
        
        # Check that gradients are computed for all parameters
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for parameter {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for parameter {name}"
    
    def test_forward_pass_deterministic(self):
        """Test that forward pass is deterministic with same input."""
        torch.manual_seed(42)
        model = MicroLM(vocab_size=25, embedding_dim=48, hidden_dim=96, num_layers=1)
        
        input_tensor = torch.randint(0, 25, (3, 15))
        
        # Run forward pass twice
        output1 = model(input_tensor)
        output2 = model(input_tensor)
        
        # Outputs should be identical
        assert torch.allclose(output1, output2), "Forward pass is not deterministic"


class TestMicroLMHiddenState:
    """Test hidden state initialization and management."""
    
    def test_init_hidden_shapes(self):
        """Test that init_hidden produces correct tensor shapes."""
        model = MicroLM(vocab_size=40, embedding_dim=80, hidden_dim=160, num_layers=3)
        
        batch_sizes = [1, 16, 64]
        
        for batch_size in batch_sizes:
            hidden_state, cell_state = model.init_hidden(batch_size)
            
            expected_shape = (3, batch_size, 160)  # (num_layers, batch_size, hidden_dim)
            assert hidden_state.shape == expected_shape
            assert cell_state.shape == expected_shape
    
    def test_init_hidden_zero_initialization(self):
        """Test that hidden states are initialized to zeros."""
        model = MicroLM(vocab_size=30, embedding_dim=60, hidden_dim=120, num_layers=2)
        
        hidden_state, cell_state = model.init_hidden(batch_size=5)
        
        assert torch.allclose(hidden_state, torch.zeros_like(hidden_state))
        assert torch.allclose(cell_state, torch.zeros_like(cell_state))
    
    def test_init_hidden_device_placement(self):
        """Test that hidden states are placed on the correct device."""
        model = MicroLM(vocab_size=25, embedding_dim=50, hidden_dim=100, num_layers=1)
        
        # Test with CPU (default)
        hidden_state, cell_state = model.init_hidden(batch_size=3)
        assert hidden_state.device.type == 'cpu'
        assert cell_state.device.type == 'cpu'
        
        # Test with explicit device specification
        device = torch.device('cpu')
        hidden_state, cell_state = model.init_hidden(batch_size=3, device=device)
        assert hidden_state.device == device
        assert cell_state.device == device


class TestMicroLMModelInfo:
    """Test model information and inspection utilities."""
    
    def test_get_model_info_structure(self):
        """Test that get_model_info returns properly structured information."""
        model = MicroLM(vocab_size=100, embedding_dim=128, hidden_dim=256, num_layers=2)
        
        info = model.get_model_info()
        
        # Check top-level keys
        expected_keys = ['architecture', 'parameters', 'model_size_mb', 'layer_info']
        assert all(key in info for key in expected_keys)
        
        # Check architecture info
        arch = info['architecture']
        assert arch['vocab_size'] == 100
        assert arch['embedding_dim'] == 128
        assert arch['hidden_dim'] == 256
        assert arch['num_layers'] == 2
        assert arch['dropout'] == 0.2
        
        # Check parameter counts
        params = info['parameters']
        assert 'embedding' in params
        assert 'lstm' in params
        assert 'output_projection' in params
        assert 'total' in params
        assert params['total'] == params['embedding'] + params['lstm'] + params['output_projection']
        
        # Check that model size is reasonable
        assert isinstance(info['model_size_mb'], (int, float))
        assert info['model_size_mb'] > 0
    
    def test_get_model_info_parameter_counts(self):
        """Test that parameter counts are calculated correctly."""
        model = MicroLM(vocab_size=50, embedding_dim=64, hidden_dim=128, num_layers=1)
        
        info = model.get_model_info()
        params = info['parameters']
        
        # Calculate expected parameter counts
        # Embedding: vocab_size * embedding_dim
        expected_embedding = 50 * 64
        
        # LSTM: 4 * (input_size * hidden_size + hidden_size * hidden_size + hidden_size) for each layer
        # For single layer: 4 * (64 * 128 + 128 * 128 + 128 + 128)
        expected_lstm = 4 * (64 * 128 + 128 * 128 + 128 + 128)
        
        # Output projection: hidden_dim * vocab_size + vocab_size
        expected_output = 128 * 50 + 50
        
        assert params['embedding'] == expected_embedding
        assert params['lstm'] == expected_lstm
        assert params['output_projection'] == expected_output
    
    def test_model_repr(self):
        """Test string representation of the model."""
        model = MicroLM(vocab_size=75, embedding_dim=96, hidden_dim=192, num_layers=3, dropout=0.1)
        
        repr_str = repr(model)
        
        # Check that all important parameters are in the representation
        assert 'MicroLM' in repr_str
        assert 'vocab_size=75' in repr_str
        assert 'embedding_dim=96' in repr_str
        assert 'hidden_dim=192' in repr_str
        assert 'num_layers=3' in repr_str
        assert 'dropout=0.1' in repr_str


class TestMicroLMIntegration:
    """Integration tests for complete model functionality."""
    
    def test_complete_training_step(self):
        """Test a complete training step with loss calculation."""
        model = MicroLM(vocab_size=30, embedding_dim=64, hidden_dim=128, num_layers=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Create sample data
        batch_size, seq_length = 4, 20
        input_seq = torch.randint(0, 30, (batch_size, seq_length))
        target_seq = torch.randint(0, 30, (batch_size, seq_length))
        
        # Forward pass
        output = model(input_seq)
        loss = criterion(output.view(-1, 30), target_seq.view(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check that loss is a valid number
        assert not torch.isnan(loss)
        assert loss.item() > 0
    
    def test_model_state_dict_save_load(self):
        """Test that model state can be saved and loaded correctly."""
        # Set seed for reproducible initialization
        torch.manual_seed(42)
        model1 = MicroLM(vocab_size=40, embedding_dim=80, hidden_dim=160, num_layers=2)
        
        # Get initial state
        state_dict = model1.state_dict()
        
        # Create new model and load state
        model2 = MicroLM(vocab_size=40, embedding_dim=80, hidden_dim=160, num_layers=2)
        model2.load_state_dict(state_dict)
        
        # Test that both models produce same output with same input
        torch.manual_seed(123)  # Set seed for input generation
        input_tensor = torch.randint(0, 40, (2, 15))
        
        # Set models to eval mode to ensure deterministic behavior
        model1.eval()
        model2.eval()
        
        output1 = model1(input_tensor)
        output2 = model2(input_tensor)
        
        assert torch.allclose(output1, output2), "Models don't produce same output after state loading"