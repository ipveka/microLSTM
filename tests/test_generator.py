"""
Comprehensive tests for the TextGenerator class.

This module tests all aspects of text generation including:
- Different sampling strategies (greedy, temperature, top-k, top-p)
- Parameter validation and error handling
- Prompt processing and continuation generation
- Length control and stopping criteria
- Edge cases and error conditions
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch
import warnings

from micro_lm.generator import TextGenerator, GenerationError
from micro_lm.model import MicroLM
from micro_lm.tokenizer import CharacterTokenizer


class TestTextGeneratorInitialization:
    """Test TextGenerator initialization and validation."""
    
    def test_valid_initialization(self):
        """Test successful initialization with valid model and tokenizer."""
        # Create test components
        tokenizer = CharacterTokenizer("hello world")
        model = MicroLM(
            vocab_size=tokenizer.vocab_size(),
            embedding_dim=32,
            hidden_dim=64,
            num_layers=1
        )
        
        # Initialize generator
        generator = TextGenerator(model, tokenizer)
        
        # Verify initialization
        assert generator.model is model
        assert generator.tokenizer is tokenizer
        assert generator.device == next(model.parameters()).device
        assert not generator.model.training  # Should be in eval mode
    
    def test_initialization_with_device(self):
        """Test initialization with specific device."""
        tokenizer = CharacterTokenizer("abc")
        model = MicroLM(vocab_size=3, embedding_dim=16, hidden_dim=32, num_layers=1)
        device = torch.device("cpu")
        
        generator = TextGenerator(model, tokenizer, device=device)
        
        assert generator.device == device
    
    def test_invalid_model_type(self):
        """Test initialization with invalid model type."""
        tokenizer = CharacterTokenizer("abc")
        invalid_model = nn.Linear(10, 5)  # Not a MicroLM
        
        with pytest.raises(TypeError, match="model must be MicroLM instance"):
            TextGenerator(invalid_model, tokenizer)
    
    def test_invalid_tokenizer_type(self):
        """Test initialization with invalid tokenizer type."""
        model = MicroLM(vocab_size=5, embedding_dim=16, hidden_dim=32, num_layers=1)
        invalid_tokenizer = "not a tokenizer"
        
        with pytest.raises(TypeError, match="tokenizer must be CharacterTokenizer instance"):
            TextGenerator(model, invalid_tokenizer)
    
    def test_vocab_size_mismatch(self):
        """Test initialization with mismatched vocabulary sizes."""
        tokenizer = CharacterTokenizer("abc")  # vocab_size = 3
        model = MicroLM(vocab_size=5, embedding_dim=16, hidden_dim=32, num_layers=1)  # Different size
        
        with pytest.raises(ValueError, match="Model vocab_size.*must match tokenizer vocab_size"):
            TextGenerator(model, tokenizer)
    
    def test_untrained_model_warning(self):
        """Test warning for potentially untrained model."""
        tokenizer = CharacterTokenizer("abc")
        model = MicroLM(vocab_size=3, embedding_dim=16, hidden_dim=32, num_layers=1)
        
        # Initialize model weights to very small values to trigger warning
        with torch.no_grad():
            for param in model.parameters():
                param.fill_(1e-8)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            TextGenerator(model, tokenizer)
            
            # Check if warning was raised
            assert len(w) > 0
            assert "very low variance" in str(w[0].message)


class TestParameterValidation:
    """Test parameter validation for generation methods."""
    
    @pytest.fixture
    def generator(self):
        """Create a test generator."""
        tokenizer = CharacterTokenizer("hello world test")
        model = MicroLM(
            vocab_size=tokenizer.vocab_size(),
            embedding_dim=32,
            hidden_dim=64,
            num_layers=1
        )
        return TextGenerator(model, tokenizer)
    
    def test_valid_parameters(self, generator):
        """Test generation with valid parameters."""
        # This should not raise any exceptions
        generator._validate_generation_params(
            prompt="hello",
            length=10,
            temperature=0.8,
            top_k=5,
            top_p=0.9,
            repetition_penalty=1.1
        )
    
    def test_invalid_prompt_type(self, generator):
        """Test validation with invalid prompt type."""
        with pytest.raises(ValueError, match="Prompt must be a string"):
            generator._validate_generation_params(
                prompt=123,  # Not a string
                length=10,
                temperature=0.8,
                top_k=None,
                top_p=None,
                repetition_penalty=1.0
            )
    
    def test_empty_prompt(self, generator):
        """Test validation with empty prompt."""
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            generator._validate_generation_params(
                prompt="",
                length=10,
                temperature=0.8,
                top_k=None,
                top_p=None,
                repetition_penalty=1.0
            )
    
    def test_invalid_length(self, generator):
        """Test validation with invalid length."""
        with pytest.raises(ValueError, match="Length must be positive"):
            generator._validate_generation_params(
                prompt="hello",
                length=0,  # Invalid length
                temperature=0.8,
                top_k=None,
                top_p=None,
                repetition_penalty=1.0
            )
    
    def test_invalid_temperature(self, generator):
        """Test validation with invalid temperature."""
        with pytest.raises(ValueError, match="Temperature must be non-negative"):
            generator._validate_generation_params(
                prompt="hello",
                length=10,
                temperature=-0.5,  # Negative temperature
                top_k=None,
                top_p=None,
                repetition_penalty=1.0
            )
    
    def test_invalid_top_k(self, generator):
        """Test validation with invalid top_k."""
        with pytest.raises(ValueError, match="top_k must be positive"):
            generator._validate_generation_params(
                prompt="hello",
                length=10,
                temperature=0.8,
                top_k=0,  # Invalid top_k
                top_p=None,
                repetition_penalty=1.0
            )
    
    def test_invalid_top_p(self, generator):
        """Test validation with invalid top_p."""
        with pytest.raises(ValueError, match="top_p must be between 0.0 and 1.0"):
            generator._validate_generation_params(
                prompt="hello",
                length=10,
                temperature=0.8,
                top_k=None,
                top_p=1.5,  # Invalid top_p
                repetition_penalty=1.0
            )
    
    def test_invalid_repetition_penalty(self, generator):
        """Test validation with invalid repetition penalty."""
        with pytest.raises(ValueError, match="repetition_penalty must be positive"):
            generator._validate_generation_params(
                prompt="hello",
                length=10,
                temperature=0.8,
                top_k=None,
                top_p=None,
                repetition_penalty=0.0  # Invalid penalty
            )


class TestPromptProcessing:
    """Test prompt processing functionality."""
    
    @pytest.fixture
    def generator(self):
        """Create a test generator."""
        tokenizer = CharacterTokenizer("hello world")
        model = MicroLM(
            vocab_size=tokenizer.vocab_size(),
            embedding_dim=32,
            hidden_dim=64,
            num_layers=1
        )
        return TextGenerator(model, tokenizer)
    
    def test_valid_prompt_processing(self, generator):
        """Test processing of valid prompt."""
        prompt = "hello"
        tokens = generator._process_prompt(prompt)
        
        # Verify tokens are generated
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(token, int) for token in tokens)
        
        # Verify round-trip conversion
        decoded = generator.tokenizer.decode(tokens)
        assert decoded == prompt
    
    def test_prompt_with_unknown_characters(self, generator):
        """Test processing prompt with unknown characters."""
        # Prompt contains characters not in tokenizer vocabulary
        prompt = "hello xyz"  # 'x', 'y', 'z' not in "hello world"
        
        with pytest.raises(GenerationError, match="unknown characters"):
            generator._process_prompt(prompt)
    
    def test_empty_prompt_processing(self, generator):
        """Test processing of empty prompt after tokenization."""
        # Mock tokenizer to return empty list
        generator.tokenizer.encode = Mock(return_value=[])
        
        with pytest.raises(GenerationError, match="empty token sequence"):
            generator._process_prompt("test")


class TestSamplingStrategies:
    """Test different sampling strategies."""
    
    @pytest.fixture
    def generator(self):
        """Create a test generator with trained-like weights."""
        tokenizer = CharacterTokenizer("abcdefghijklmnopqrstuvwxyz ")
        model = MicroLM(
            vocab_size=tokenizer.vocab_size(),
            embedding_dim=32,
            hidden_dim=64,
            num_layers=1
        )
        
        # Initialize with reasonable weights to avoid warnings
        with torch.no_grad():
            for param in model.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.zeros_(param)
        
        return TextGenerator(model, tokenizer)
    
    def test_greedy_generation(self, generator):
        """Test greedy decoding (deterministic generation)."""
        prompt = "hello"
        length = 10
        
        # Generate twice with same parameters
        text1 = generator.generate_greedy(prompt, length)
        text2 = generator.generate_greedy(prompt, length)
        
        # Results should be identical (deterministic)
        assert text1 == text2
        assert text1.startswith(prompt)
        assert len(text1) == len(prompt) + length
    
    def test_temperature_generation(self, generator):
        """Test temperature-based sampling."""
        prompt = "test"
        length = 20
        
        # Test different temperatures
        low_temp_text = generator.generate_with_temperature(prompt, length, temperature=0.1)
        high_temp_text = generator.generate_with_temperature(prompt, length, temperature=2.0)
        
        # Both should start with prompt and have correct length
        assert low_temp_text.startswith(prompt)
        assert high_temp_text.startswith(prompt)
        assert len(low_temp_text) == len(prompt) + length
        assert len(high_temp_text) == len(prompt) + length
    
    def test_reproducible_generation(self, generator):
        """Test that generation is reproducible with same seed."""
        prompt = "test"
        length = 15
        seed = 42
        
        # Generate with same seed
        text1 = generator.generate(prompt, length, temperature=0.8, seed=seed)
        text2 = generator.generate(prompt, length, temperature=0.8, seed=seed)
        
        # Results should be identical
        assert text1 == text2
    
    def test_top_k_sampling(self, generator):
        """Test top-k sampling."""
        prompt = "a"
        length = 10
        
        # Generate with top-k sampling
        text = generator.generate(prompt, length, temperature=0.8, top_k=5)
        
        assert text.startswith(prompt)
        assert len(text) == len(prompt) + length
    
    def test_top_p_sampling(self, generator):
        """Test top-p (nucleus) sampling."""
        prompt = "a"
        length = 10
        
        # Generate with top-p sampling
        text = generator.generate(prompt, length, temperature=0.8, top_p=0.9)
        
        assert text.startswith(prompt)
        assert len(text) == len(prompt) + length
    
    def test_repetition_penalty(self, generator):
        """Test repetition penalty functionality."""
        prompt = "a"
        length = 20
        
        # Generate with and without repetition penalty
        text_no_penalty = generator.generate(prompt, length, temperature=0.8, repetition_penalty=1.0)
        text_with_penalty = generator.generate(prompt, length, temperature=0.8, repetition_penalty=1.5)
        
        assert text_no_penalty.startswith(prompt)
        assert text_with_penalty.startswith(prompt)
        assert len(text_no_penalty) == len(prompt) + length
        assert len(text_with_penalty) == len(prompt) + length


class TestSamplingHelpers:
    """Test sampling helper methods."""
    
    @pytest.fixture
    def generator(self):
        """Create a test generator."""
        tokenizer = CharacterTokenizer("abc")
        model = MicroLM(vocab_size=3, embedding_dim=16, hidden_dim=32, num_layers=1)
        return TextGenerator(model, tokenizer)
    
    def test_repetition_penalty_application(self, generator):
        """Test repetition penalty logic."""
        # Create test logits
        logits = torch.tensor([1.0, 2.0, 3.0])
        generated_tokens = [0, 1]  # Tokens 0 and 1 have been generated
        penalty = 1.5
        
        penalized = generator._apply_repetition_penalty(logits, generated_tokens, penalty)
        
        # Token 0 and 1 should be penalized, token 2 should remain unchanged
        assert penalized[2] == logits[2]  # Unchanged
        assert penalized[0] < logits[0]   # Penalized (positive logit)
        assert penalized[1] < logits[1]   # Penalized (positive logit)
    
    def test_top_k_filtering(self, generator):
        """Test top-k filtering logic."""
        # Create test logits
        logits = torch.tensor([1.0, 3.0, 2.0, 0.5])
        k = 2
        
        filtered = generator._apply_top_k_filtering(logits, k)
        
        # Only top-2 values should remain, others should be -inf
        assert filtered[1] == logits[1]  # Highest value
        assert filtered[2] == logits[2]  # Second highest
        assert filtered[0] == float('-inf')  # Filtered out
        assert filtered[3] == float('-inf')  # Filtered out
    
    def test_top_k_filtering_no_effect(self, generator):
        """Test top-k filtering when k >= vocab_size."""
        logits = torch.tensor([1.0, 2.0, 3.0])
        k = 5  # Larger than vocab size
        
        filtered = generator._apply_top_k_filtering(logits, k)
        
        # Should be unchanged
        assert torch.equal(filtered, logits)
    
    def test_top_p_filtering(self, generator):
        """Test top-p (nucleus) filtering logic."""
        # Create test logits that will result in known probabilities
        logits = torch.tensor([2.0, 1.0, 0.0, -1.0])  # Descending order after softmax
        p = 0.8
        
        filtered = generator._apply_top_p_filtering(logits, p)
        
        # At least the most probable token should remain
        assert filtered[0] != float('-inf')
        
        # Some tokens should be filtered out
        assert any(filtered == float('-inf'))
    
    def test_top_p_filtering_no_effect(self, generator):
        """Test top-p filtering when p >= 1.0."""
        logits = torch.tensor([1.0, 2.0, 3.0])
        p = 1.0
        
        filtered = generator._apply_top_p_filtering(logits, p)
        
        # Should be unchanged
        assert torch.equal(filtered, logits)


class TestStoppingCriteria:
    """Test stopping criteria and length control."""
    
    @pytest.fixture
    def generator(self):
        """Create a test generator."""
        tokenizer = CharacterTokenizer("hello world stop")
        model = MicroLM(
            vocab_size=tokenizer.vocab_size(),
            embedding_dim=32,
            hidden_dim=64,
            num_layers=1
        )
        return TextGenerator(model, tokenizer)
    
    def test_length_control(self, generator):
        """Test that generation respects length parameter."""
        prompt = "hello"
        length = 15
        
        text = generator.generate(prompt, length, temperature=0.0)
        
        # Should generate exactly the requested length
        assert len(text) == len(prompt) + length
    
    def test_stop_tokens(self, generator):
        """Test stop token functionality."""
        # Mock the generation to produce a known sequence
        with patch.object(generator, '_generate_next_token') as mock_gen:
            # Make it generate "stop" character by character
            stop_chars = list("stop")
            char_indices = [generator.tokenizer.encode(char)[0] for char in stop_chars]
            
            # Return characters that spell "stop"
            mock_gen.side_effect = [(idx, None) for idx in char_indices]
            
            text = generator.generate("test", length=10, stop_tokens=["stop"])
            
            # Generation should stop when "stop" is encountered
            assert "stop" in text
    
    def test_should_stop_logic(self, generator):
        """Test stop token detection logic."""
        # Test with tokens that form a stop word
        stop_tokens = ["world"]
        
        # Create token sequence that ends with "world"
        text_tokens = generator.tokenizer.encode("hello world")
        
        should_stop = generator._should_stop(text_tokens, stop_tokens)
        assert should_stop
        
        # Test with tokens that don't form a stop word
        text_tokens = generator.tokenizer.encode("hello test")
        should_stop = generator._should_stop(text_tokens, stop_tokens)
        assert not should_stop
    
    def test_no_stop_tokens(self, generator):
        """Test behavior when no stop tokens are provided."""
        text_tokens = [1, 2, 3, 4]
        should_stop = generator._should_stop(text_tokens, None)
        assert not should_stop
        
        should_stop = generator._should_stop(text_tokens, [])
        assert not should_stop


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.fixture
    def generator(self):
        """Create a test generator."""
        tokenizer = CharacterTokenizer("hello world")
        model = MicroLM(
            vocab_size=tokenizer.vocab_size(),
            embedding_dim=32,
            hidden_dim=64,
            num_layers=1
        )
        return TextGenerator(model, tokenizer)
    
    def test_generation_with_model_error(self, generator):
        """Test handling of model errors during generation."""
        # Mock model to raise an exception
        generator.model.forward = Mock(side_effect=RuntimeError("Model error"))
        
        with pytest.raises(GenerationError, match="Text generation failed"):
            generator.generate("test", length=5)
    
    def test_generation_with_invalid_prompt(self, generator):
        """Test generation with prompt containing invalid characters."""
        # Prompt with characters not in vocabulary
        invalid_prompt = "xyz123"
        
        with pytest.raises(GenerationError):
            generator.generate(invalid_prompt, length=5)
    
    def test_zero_length_generation(self, generator):
        """Test generation with zero length."""
        with pytest.raises(GenerationError, match="Length must be positive"):
            generator.generate("test", length=0)


class TestUtilityMethods:
    """Test utility and information methods."""
    
    @pytest.fixture
    def generator(self):
        """Create a test generator."""
        tokenizer = CharacterTokenizer("hello world")
        model = MicroLM(
            vocab_size=tokenizer.vocab_size(),
            embedding_dim=32,
            hidden_dim=64,
            num_layers=1
        )
        return TextGenerator(model, tokenizer)
    
    def test_get_generation_info(self, generator):
        """Test generation information method."""
        info = generator.get_generation_info()
        
        # Verify required keys are present
        assert 'model_info' in info
        assert 'vocab_size' in info
        assert 'device' in info
        assert 'supported_strategies' in info
        assert 'features' in info
        
        # Verify values
        assert info['vocab_size'] == generator.tokenizer.vocab_size()
        assert isinstance(info['supported_strategies'], list)
        assert isinstance(info['features'], list)
    
    def test_string_representation(self, generator):
        """Test string representation of generator."""
        repr_str = repr(generator)
        
        assert "TextGenerator" in repr_str
        assert str(generator.tokenizer.vocab_size()) in repr_str
        assert str(generator.device) in repr_str


class TestConvenienceMethods:
    """Test convenience methods for common generation patterns."""
    
    @pytest.fixture
    def generator(self):
        """Create a test generator."""
        tokenizer = CharacterTokenizer("abcdefghijklmnopqrstuvwxyz ")
        model = MicroLM(
            vocab_size=tokenizer.vocab_size(),
            embedding_dim=32,
            hidden_dim=64,
            num_layers=1
        )
        return TextGenerator(model, tokenizer)
    
    def test_generate_greedy_convenience(self, generator):
        """Test greedy generation convenience method."""
        prompt = "test"
        length = 10
        
        text = generator.generate_greedy(prompt, length)
        
        assert text.startswith(prompt)
        assert len(text) == len(prompt) + length
    
    def test_generate_with_temperature_convenience(self, generator):
        """Test temperature generation convenience method."""
        prompt = "test"
        length = 10
        temperature = 0.7
        
        text = generator.generate_with_temperature(prompt, length, temperature)
        
        assert text.startswith(prompt)
        assert len(text) == len(prompt) + length


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"])