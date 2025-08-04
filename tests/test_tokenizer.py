"""
Simplified unit tests for the CharacterTokenizer implementation.

This module contains focused tests for the CharacterTokenizer class, covering:
- Basic tokenization functionality
- Vocabulary creation and management
- Encoding and decoding operations
- Error handling for invalid inputs
"""

import pytest
import tempfile
import json
import os
from micro_lstm.tokenizer import CharacterTokenizer
from micro_lstm.exceptions import TokenizationError, FileOperationError


class TestCharacterTokenizerBasic:
    """Test basic tokenizer functionality."""
    
    def test_basic_initialization(self):
        """Test basic tokenizer initialization."""
        text = "hello world"
        tokenizer = CharacterTokenizer(text)
        
        assert tokenizer.vocab_size() == 8  # h, e, l, o, w, r, d, space
        assert len(tokenizer.get_vocab()) == 8
    
    def test_encode_decode_roundtrip(self):
        """Test that encode followed by decode returns original text."""
        text = "hello world"
        tokenizer = CharacterTokenizer(text)
        
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        
        assert decoded == text
        assert isinstance(encoded, list)
        assert all(isinstance(x, int) for x in encoded)
    
    def test_vocabulary_creation(self):
        """Test vocabulary creation from text."""
        text = "abc"
        tokenizer = CharacterTokenizer(text)
        
        vocab = tokenizer.get_vocab()
        assert len(vocab) == 3
        assert 'a' in vocab
        assert 'b' in vocab
        assert 'c' in vocab
    
    def test_empty_text_initialization(self):
        """Test initialization with empty text raises error."""
        with pytest.raises(TokenizationError):
            CharacterTokenizer("")
        
        with pytest.raises(TokenizationError):
            CharacterTokenizer(None)


class TestCharacterTokenizerEncoding:
    """Test encoding functionality."""
    
    def test_encode_valid_text(self):
        """Test encoding of valid text."""
        tokenizer = CharacterTokenizer("hello")
        encoded = tokenizer.encode("hello")
        
        assert isinstance(encoded, list)
        assert len(encoded) == 5
        assert all(isinstance(x, int) for x in encoded)
        assert all(0 <= x < tokenizer.vocab_size() for x in encoded)
    
    def test_encode_invalid_input_type(self):
        """Test encoding with invalid input type."""
        tokenizer = CharacterTokenizer("hello")
        
        with pytest.raises(TokenizationError):
            tokenizer.encode(123)
        
        with pytest.raises(TokenizationError):
            tokenizer.encode(None)
    
    def test_encode_unknown_character(self):
        """Test encoding with character not in vocabulary."""
        tokenizer = CharacterTokenizer("hello")
        
        with pytest.raises(TokenizationError):
            tokenizer.encode("xyz")


class TestCharacterTokenizerDecoding:
    """Test decoding functionality."""
    
    def test_decode_valid_indices(self):
        """Test decoding of valid indices."""
        tokenizer = CharacterTokenizer("hello")
        encoded = tokenizer.encode("hello")
        decoded = tokenizer.decode(encoded)
        
        assert decoded == "hello"
    
    def test_decode_invalid_input_type(self):
        """Test decoding with invalid input type."""
        tokenizer = CharacterTokenizer("hello")
        
        with pytest.raises(TokenizationError):
            tokenizer.decode("abc")
    
    def test_decode_invalid_index(self):
        """Test decoding with index out of range."""
        tokenizer = CharacterTokenizer("hello")
        
        with pytest.raises(TokenizationError):
            tokenizer.decode([999])


class TestCharacterTokenizerPersistence:
    """Test vocabulary save and load functionality."""
    
    def test_save_load_vocabulary(self):
        """Test saving and loading vocabulary."""
        text = "hello world"
        tokenizer = CharacterTokenizer(text)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            vocab_path = f.name
        
        try:
            # Save vocabulary
            tokenizer.save_vocab(vocab_path)
            
            # Load vocabulary
            loaded_tokenizer = CharacterTokenizer.load_vocab(vocab_path)
            
            # Verify they're the same
            assert loaded_tokenizer.vocab_size() == tokenizer.vocab_size()
            assert loaded_tokenizer.get_vocab() == tokenizer.get_vocab()
            
            # Test encoding/decoding still works
            test_text = "hello"
            original_encoded = tokenizer.encode(test_text)
            loaded_encoded = loaded_tokenizer.encode(test_text)
            assert original_encoded == loaded_encoded
            
        finally:
            if os.path.exists(vocab_path):
                os.unlink(vocab_path)
    
    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file."""
        with pytest.raises(FileOperationError):
            CharacterTokenizer.load_vocab("/nonexistent/file.json")
    
    def test_load_invalid_json(self):
        """Test loading invalid JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            vocab_path = f.name
        
        try:
            with pytest.raises(FileOperationError):
                CharacterTokenizer.load_vocab(vocab_path)
        finally:
            if os.path.exists(vocab_path):
                os.unlink(vocab_path) 