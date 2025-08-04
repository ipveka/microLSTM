"""
Simplified unit tests for the TextGenerator implementation.

This module contains focused tests for the TextGenerator class, covering:
- Basic generator initialization and setup
- Text generation functionality
- Parameter validation
- Error handling for invalid inputs
"""

import pytest
import torch
from micro_lstm.generator import TextGenerator
from micro_lstm.model import MicroLM
from micro_lstm.tokenizer import CharacterTokenizer
from micro_lstm.exceptions import ModelConfigurationError, GenerationError, TokenizationError


class TestTextGeneratorBasic:
    """Test basic generator functionality."""
    
    def test_basic_initialization(self):
        """Test basic generator initialization."""
        tokenizer = CharacterTokenizer("hello world")
        model = MicroLM(
            vocab_size=tokenizer.vocab_size(),
            embedding_dim=32,
            hidden_dim=64,
            num_layers=1
        )
        
        generator = TextGenerator(model, tokenizer)
        
        assert generator.model is model
        assert generator.tokenizer is tokenizer
        assert isinstance(generator.device, torch.device)
    
    def test_invalid_model_type(self):
        """Test initialization with invalid model type."""
        tokenizer = CharacterTokenizer("hello")
        
        with pytest.raises(ModelConfigurationError):
            TextGenerator("not_a_model", tokenizer)
    
    def test_invalid_tokenizer_type(self):
        """Test initialization with invalid tokenizer type."""
        model = MicroLM(vocab_size=10, embedding_dim=16, hidden_dim=32, num_layers=1)
        
        with pytest.raises(ModelConfigurationError):
            TextGenerator(model, "not_a_tokenizer")
    
    def test_vocab_size_mismatch(self):
        """Test initialization with mismatched vocabulary sizes."""
        tokenizer = CharacterTokenizer("hello")
        model = MicroLM(vocab_size=100, embedding_dim=16, hidden_dim=32, num_layers=1)
        
        with pytest.raises(ModelConfigurationError):
            TextGenerator(model, tokenizer)


class TestTextGeneration:
    """Test text generation functionality."""
    
    def setup_method(self):
        """Set up test components for each test."""
        self.tokenizer = CharacterTokenizer("hello world example text")
        self.model = MicroLM(
            vocab_size=self.tokenizer.vocab_size(),
            embedding_dim=32,
            hidden_dim=64,
            num_layers=1
        )
        self.generator = TextGenerator(self.model, self.tokenizer)
    
    def test_basic_generation(self):
        """Test basic text generation."""
        prompt = "hello"
        generated = self.generator.generate(prompt, length=10)
        
        assert isinstance(generated, str)
        assert len(generated) >= len(prompt)
        assert generated.startswith(prompt)
    
    def test_generation_with_different_lengths(self):
        """Test generation with different lengths."""
        prompt = "hello"
        
        # Short generation
        short_gen = self.generator.generate(prompt, length=5)
        assert len(short_gen) >= len(prompt)
        
        # Longer generation
        long_gen = self.generator.generate(prompt, length=20)
        assert len(long_gen) >= len(prompt)
        assert len(long_gen) > len(short_gen)
    
    def test_generation_with_different_temperatures(self):
        """Test generation with different temperatures."""
        prompt = "hello"
        
        # Low temperature (more deterministic)
        low_temp = self.generator.generate(prompt, length=10, temperature=0.1)
        
        # High temperature (more random)
        high_temp = self.generator.generate(prompt, length=10, temperature=1.0)
        
        assert isinstance(low_temp, str)
        assert isinstance(high_temp, str)
    
    def test_generation_with_top_k(self):
        """Test generation with top-k sampling."""
        prompt = "hello"
        generated = self.generator.generate(prompt, length=10, top_k=3)
        
        assert isinstance(generated, str)
        assert len(generated) >= len(prompt)
    
    def test_generation_with_top_p(self):
        """Test generation with top-p sampling."""
        prompt = "hello"
        generated = self.generator.generate(prompt, length=10, top_p=0.9)
        
        assert isinstance(generated, str)
        assert len(generated) >= len(prompt)


class TestParameterValidation:
    """Test parameter validation."""
    
    def setup_method(self):
        """Set up test components for each test."""
        self.tokenizer = CharacterTokenizer("hello world")
        self.model = MicroLM(
            vocab_size=self.tokenizer.vocab_size(),
            embedding_dim=32,
            hidden_dim=64,
            num_layers=1
        )
        self.generator = TextGenerator(self.model, self.tokenizer)
    
    def test_invalid_prompt_type(self):
        """Test generation with invalid prompt type."""
        with pytest.raises(GenerationError):
            self.generator.generate(123, length=10)
    
    def test_empty_prompt(self):
        """Test generation with empty prompt."""
        with pytest.raises(GenerationError):
            self.generator.generate("", length=10)
    
    def test_invalid_length(self):
        """Test generation with invalid length."""
        with pytest.raises(GenerationError):
            self.generator.generate("hello", length=0)
    
    def test_invalid_temperature(self):
        """Test generation with invalid temperature."""
        with pytest.raises(GenerationError):
            self.generator.generate("hello", length=10, temperature=-0.5)
    
    def test_invalid_top_k(self):
        """Test generation with invalid top_k."""
        with pytest.raises(GenerationError):
            self.generator.generate("hello", length=10, top_k=0)
    
    def test_invalid_top_p(self):
        """Test generation with invalid top_p."""
        with pytest.raises(GenerationError):
            self.generator.generate("hello", length=10, top_p=1.5)
    
    def test_invalid_repetition_penalty(self):
        """Test generation with invalid repetition penalty."""
        with pytest.raises(GenerationError):
            self.generator.generate("hello", length=10, repetition_penalty=0.0)


class TestPromptProcessing:
    """Test prompt processing functionality."""
    
    def setup_method(self):
        """Set up test components for each test."""
        self.tokenizer = CharacterTokenizer("hello world")
        self.model = MicroLM(
            vocab_size=self.tokenizer.vocab_size(),
            embedding_dim=32,
            hidden_dim=64,
            num_layers=1
        )
        self.generator = TextGenerator(self.model, self.tokenizer)
    
    def test_prompt_with_known_characters(self):
        """Test processing prompt with known characters."""
        prompt = "hello"
        processed = self.generator._process_prompt(prompt)
        
        assert isinstance(processed, list)
        assert all(isinstance(x, int) for x in processed)
        assert len(processed) == len(prompt)
    
    def test_prompt_with_unknown_characters(self):
        """Test processing prompt with unknown characters."""
        prompt = "xyz"  # Characters not in vocabulary
        
        with pytest.raises(TokenizationError):
            self.generator._process_prompt(prompt)
    
    def test_prompt_tokenization(self):
        """Test that prompt is properly tokenized."""
        prompt = "hello"
        processed = self.generator._process_prompt(prompt)
        
        # Should be a list with length equal to prompt length
        assert len(processed) == len(prompt)
        assert all(0 <= x < self.generator.tokenizer.vocab_size() for x in processed) 