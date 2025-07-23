"""
Comprehensive unit tests for the CharacterTokenizer class.

This test suite covers all functionality of the CharacterTokenizer including:
- Vocabulary building from text corpus
- Text encoding and decoding
- Error handling for invalid inputs
- Edge cases and boundary conditions
- File I/O operations for vocabulary persistence
"""

import unittest
import tempfile
import os
import json
from typing import List, Dict

# Import the tokenizer class
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from micro_lm.tokenizer import CharacterTokenizer


class TestCharacterTokenizer(unittest.TestCase):
    """Test suite for CharacterTokenizer class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Simple test corpus with known characters
        self.simple_corpus = "hello world"
        self.simple_tokenizer = CharacterTokenizer(self.simple_corpus)
        
        # More complex corpus for advanced testing
        self.complex_corpus = "Hello, World! 123 @#$%"
        self.complex_tokenizer = CharacterTokenizer(self.complex_corpus)
        
        # Single character corpus for edge case testing
        self.single_char_corpus = "a"
        self.single_char_tokenizer = CharacterTokenizer(self.single_char_corpus)
    
    def test_initialization_simple_corpus(self):
        """Test tokenizer initialization with a simple text corpus."""
        tokenizer = CharacterTokenizer("abc")
        
        # Check that vocabulary is built correctly
        vocab = tokenizer.get_vocab()
        expected_vocab = {'a': 0, 'b': 1, 'c': 2}
        self.assertEqual(vocab, expected_vocab)
        
        # Check vocabulary size
        self.assertEqual(tokenizer.vocab_size(), 3)
        self.assertEqual(len(tokenizer), 3)
    
    def test_initialization_with_duplicates(self):
        """Test that duplicate characters are handled correctly."""
        tokenizer = CharacterTokenizer("aabbcc")
        
        # Should only have 3 unique characters despite duplicates
        self.assertEqual(tokenizer.vocab_size(), 3)
        
        vocab = tokenizer.get_vocab()
        expected_vocab = {'a': 0, 'b': 1, 'c': 2}
        self.assertEqual(vocab, expected_vocab)
    
    def test_initialization_empty_corpus(self):
        """Test that empty corpus raises appropriate error."""
        with self.assertRaises(ValueError) as context:
            CharacterTokenizer("")
        
        self.assertIn("Text corpus cannot be empty", str(context.exception))
        
        with self.assertRaises(ValueError):
            CharacterTokenizer(None)
    
    def test_initialization_special_characters(self):
        """Test initialization with special characters and whitespace."""
        corpus = "Hello, World!\n\t @#$%^&*()"
        tokenizer = CharacterTokenizer(corpus)
        
        # Verify all special characters are included
        vocab = tokenizer.get_vocab()
        
        # Check that spaces, newlines, tabs, and symbols are included
        self.assertIn(' ', vocab)
        self.assertIn('\n', vocab)
        self.assertIn('\t', vocab)
        self.assertIn('@', vocab)
        self.assertIn('!', vocab)
        self.assertIn('(', vocab)
        self.assertIn(')', vocab)
    
    def test_encode_simple_text(self):
        """Test encoding of simple text."""
        # Test with known simple corpus
        tokenizer = CharacterTokenizer("abc")
        
        # Test encoding each character
        self.assertEqual(tokenizer.encode("a"), [0])
        self.assertEqual(tokenizer.encode("b"), [1])
        self.assertEqual(tokenizer.encode("c"), [2])
        
        # Test encoding multiple characters
        self.assertEqual(tokenizer.encode("abc"), [0, 1, 2])
        self.assertEqual(tokenizer.encode("cab"), [2, 0, 1])
        self.assertEqual(tokenizer.encode("bac"), [1, 0, 2])
    
    def test_encode_with_repetition(self):
        """Test encoding text with repeated characters."""
        tokenizer = CharacterTokenizer("abc")
        
        # Test repeated characters
        self.assertEqual(tokenizer.encode("aaa"), [0, 0, 0])
        self.assertEqual(tokenizer.encode("abcabc"), [0, 1, 2, 0, 1, 2])
        self.assertEqual(tokenizer.encode("ababab"), [0, 1, 0, 1, 0, 1])
    
    def test_encode_empty_string(self):
        """Test encoding empty string."""
        result = self.simple_tokenizer.encode("")
        self.assertEqual(result, [])
    
    def test_encode_unknown_character(self):
        """Test that encoding unknown characters raises appropriate error."""
        tokenizer = CharacterTokenizer("abc")
        
        with self.assertRaises(ValueError) as context:
            tokenizer.encode("xyz")
        
        error_msg = str(context.exception)
        self.assertIn("Character 'x' not found in vocabulary", error_msg)
        self.assertIn("Available characters:", error_msg)
    
    def test_encode_invalid_input_type(self):
        """Test that non-string input to encode raises appropriate error."""
        with self.assertRaises(ValueError) as context:
            self.simple_tokenizer.encode(123)
        
        self.assertIn("Input must be a string", str(context.exception))
        
        with self.assertRaises(ValueError):
            self.simple_tokenizer.encode(None)
        
        with self.assertRaises(ValueError):
            self.simple_tokenizer.encode(['a', 'b', 'c'])
    
    def test_decode_simple_indices(self):
        """Test decoding of simple index lists."""
        tokenizer = CharacterTokenizer("abc")
        
        # Test decoding single indices
        self.assertEqual(tokenizer.decode([0]), "a")
        self.assertEqual(tokenizer.decode([1]), "b")
        self.assertEqual(tokenizer.decode([2]), "c")
        
        # Test decoding multiple indices
        self.assertEqual(tokenizer.decode([0, 1, 2]), "abc")
        self.assertEqual(tokenizer.decode([2, 0, 1]), "cab")
        self.assertEqual(tokenizer.decode([1, 0, 2]), "bac")
    
    def test_decode_with_repetition(self):
        """Test decoding indices with repetition."""
        tokenizer = CharacterTokenizer("abc")
        
        # Test repeated indices
        self.assertEqual(tokenizer.decode([0, 0, 0]), "aaa")
        self.assertEqual(tokenizer.decode([0, 1, 2, 0, 1, 2]), "abcabc")
        self.assertEqual(tokenizer.decode([0, 1, 0, 1, 0, 1]), "ababab")
    
    def test_decode_empty_list(self):
        """Test decoding empty index list."""
        result = self.simple_tokenizer.decode([])
        self.assertEqual(result, "")
    
    def test_decode_invalid_index(self):
        """Test that decoding invalid indices raises appropriate error."""
        tokenizer = CharacterTokenizer("abc")  # vocab_size = 3, valid indices: 0, 1, 2
        
        with self.assertRaises(ValueError) as context:
            tokenizer.decode([3])  # Index out of range
        
        error_msg = str(context.exception)
        self.assertIn("Index 3 not found in vocabulary", error_msg)
        self.assertIn("Valid range: 0-2", error_msg)
        
        with self.assertRaises(ValueError):
            tokenizer.decode([-1])  # Negative index
        
        with self.assertRaises(ValueError):
            tokenizer.decode([100])  # Way out of range
    
    def test_decode_invalid_input_type(self):
        """Test that invalid input types to decode raise appropriate errors."""
        with self.assertRaises(TypeError) as context:
            self.simple_tokenizer.decode("abc")
        
        self.assertIn("Indices must be a list", str(context.exception))
        
        with self.assertRaises(TypeError):
            self.simple_tokenizer.decode(123)
        
        with self.assertRaises(TypeError) as context:
            self.simple_tokenizer.decode([1, 2, "3"])
        
        self.assertIn("All indices must be integers", str(context.exception))
        
        with self.assertRaises(TypeError):
            self.simple_tokenizer.decode([1, 2, 3.5])
    
    def test_encode_decode_round_trip(self):
        """Test that encode->decode returns original text."""
        test_cases = [
            "hello",
            "world",
            "hello world",
            "a",
            "",
            "aaa",
            "abcdefg",
        ]
        
        for text in test_cases:
            with self.subTest(text=text):
                # Only test with characters that exist in the tokenizer's vocabulary
                if all(char in self.simple_tokenizer.get_vocab() for char in text):
                    encoded = self.simple_tokenizer.encode(text)
                    decoded = self.simple_tokenizer.decode(encoded)
                    self.assertEqual(decoded, text)
    
    def test_decode_encode_round_trip(self):
        """Test that decode->encode returns original indices."""
        tokenizer = CharacterTokenizer("abcde")
        
        test_cases = [
            [0],
            [1, 2, 3],
            [0, 1, 2, 3, 4],
            [4, 3, 2, 1, 0],
            [],
            [0, 0, 0],
            [1, 2, 1, 2],
        ]
        
        for indices in test_cases:
            with self.subTest(indices=indices):
                decoded = tokenizer.decode(indices)
                encoded = tokenizer.encode(decoded)
                self.assertEqual(encoded, indices)
    
    def test_vocab_size_method(self):
        """Test vocab_size method returns correct size."""
        # Test simple cases
        self.assertEqual(CharacterTokenizer("a").vocab_size(), 1)
        self.assertEqual(CharacterTokenizer("ab").vocab_size(), 2)
        self.assertEqual(CharacterTokenizer("abc").vocab_size(), 3)
        
        # Test with duplicates
        self.assertEqual(CharacterTokenizer("aaa").vocab_size(), 1)
        self.assertEqual(CharacterTokenizer("aabb").vocab_size(), 2)
        
        # Test with complex text
        complex_text = "Hello, World! 123"
        tokenizer = CharacterTokenizer(complex_text)
        unique_chars = set(complex_text)
        self.assertEqual(tokenizer.vocab_size(), len(unique_chars))
    
    def test_get_vocab_method(self):
        """Test get_vocab method returns correct vocabulary mapping."""
        tokenizer = CharacterTokenizer("cba")  # Will be sorted to "abc"
        vocab = tokenizer.get_vocab()
        
        # Check that vocabulary is correctly sorted
        expected_vocab = {'a': 0, 'b': 1, 'c': 2}
        self.assertEqual(vocab, expected_vocab)
        
        # Check that returned vocab is a copy (modifications don't affect original)
        vocab['x'] = 999
        original_vocab = tokenizer.get_vocab()
        self.assertNotIn('x', original_vocab)
    
    def test_vocabulary_consistency(self):
        """Test that vocabulary mapping is consistent and deterministic."""
        # Create multiple tokenizers with same corpus
        corpus = "hello world"
        tokenizer1 = CharacterTokenizer(corpus)
        tokenizer2 = CharacterTokenizer(corpus)
        
        # Vocabularies should be identical
        self.assertEqual(tokenizer1.get_vocab(), tokenizer2.get_vocab())
        
        # Encoding should be identical
        test_text = "hello"
        self.assertEqual(tokenizer1.encode(test_text), tokenizer2.encode(test_text))
    
    def test_save_and_load_vocab(self):
        """Test saving and loading vocabulary to/from file."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Save vocabulary
            original_tokenizer = CharacterTokenizer("hello world")
            original_tokenizer.save_vocab(tmp_path)
            
            # Load vocabulary
            loaded_tokenizer = CharacterTokenizer.load_vocab(tmp_path)
            
            # Compare vocabularies
            self.assertEqual(original_tokenizer.get_vocab(), loaded_tokenizer.get_vocab())
            self.assertEqual(original_tokenizer.vocab_size(), loaded_tokenizer.vocab_size())
            
            # Test that encoding/decoding works the same
            test_text = "hello"
            original_encoded = original_tokenizer.encode(test_text)
            loaded_encoded = loaded_tokenizer.encode(test_text)
            self.assertEqual(original_encoded, loaded_encoded)
            
            original_decoded = original_tokenizer.decode(original_encoded)
            loaded_decoded = loaded_tokenizer.decode(loaded_encoded)
            self.assertEqual(original_decoded, loaded_decoded)
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_save_vocab_invalid_path(self):
        """Test that saving to invalid path raises appropriate error."""
        tokenizer = CharacterTokenizer("abc")
        
        # Try to save to a directory that doesn't exist
        invalid_path = "/nonexistent/directory/vocab.json"
        
        with self.assertRaises(IOError) as context:
            tokenizer.save_vocab(invalid_path)
        
        self.assertIn("Failed to save vocabulary", str(context.exception))
    
    def test_load_vocab_invalid_path(self):
        """Test that loading from invalid path raises appropriate error."""
        nonexistent_path = "/nonexistent/file.json"
        
        with self.assertRaises(IOError) as context:
            CharacterTokenizer.load_vocab(nonexistent_path)
        
        self.assertIn("Failed to load vocabulary", str(context.exception))
    
    def test_load_vocab_invalid_json(self):
        """Test that loading invalid JSON raises appropriate error."""
        # Create a temporary file with invalid JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            tmp_file.write("invalid json content {")
            tmp_path = tmp_file.name
        
        try:
            with self.assertRaises(ValueError) as context:
                CharacterTokenizer.load_vocab(tmp_path)
            
            self.assertIn("Invalid JSON format", str(context.exception))
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_repr_method(self):
        """Test string representation of tokenizer."""
        tokenizer = CharacterTokenizer("abc")
        repr_str = repr(tokenizer)
        
        self.assertIn("CharacterTokenizer", repr_str)
        self.assertIn("vocab_size=3", repr_str)
    
    def test_len_method(self):
        """Test len() function on tokenizer."""
        tokenizer = CharacterTokenizer("abcde")
        self.assertEqual(len(tokenizer), 5)
        
        # Test with duplicates
        tokenizer_with_dups = CharacterTokenizer("aabbcc")
        self.assertEqual(len(tokenizer_with_dups), 3)
    
    def test_unicode_characters(self):
        """Test tokenizer with Unicode characters."""
        unicode_corpus = "Hello 世界! 🌍🚀"
        tokenizer = CharacterTokenizer(unicode_corpus)
        
        # Test that Unicode characters are handled correctly
        self.assertIn('世', tokenizer.get_vocab())
        self.assertIn('界', tokenizer.get_vocab())
        self.assertIn('🌍', tokenizer.get_vocab())
        self.assertIn('🚀', tokenizer.get_vocab())
        
        # Test encoding and decoding Unicode text
        test_text = "世界🌍"
        if all(char in tokenizer.get_vocab() for char in test_text):
            encoded = tokenizer.encode(test_text)
            decoded = tokenizer.decode(encoded)
            self.assertEqual(decoded, test_text)
    
    def test_large_vocabulary(self):
        """Test tokenizer with a large vocabulary."""
        # Create a corpus with many different characters
        import string
        large_corpus = string.ascii_letters + string.digits + string.punctuation + " \n\t"
        tokenizer = CharacterTokenizer(large_corpus)
        
        # Test that all characters are included
        expected_size = len(set(large_corpus))
        self.assertEqual(tokenizer.vocab_size(), expected_size)
        
        # Test encoding and decoding
        test_text = "Hello, World! 123"
        encoded = tokenizer.encode(test_text)
        decoded = tokenizer.decode(encoded)
        self.assertEqual(decoded, test_text)
    
    def test_edge_case_single_character(self):
        """Test tokenizer with single character corpus."""
        tokenizer = CharacterTokenizer("a")
        
        self.assertEqual(tokenizer.vocab_size(), 1)
        self.assertEqual(tokenizer.get_vocab(), {'a': 0})
        
        # Test encoding and decoding
        self.assertEqual(tokenizer.encode("a"), [0])
        self.assertEqual(tokenizer.decode([0]), "a")
        
        # Test multiple occurrences
        self.assertEqual(tokenizer.encode("aaa"), [0, 0, 0])
        self.assertEqual(tokenizer.decode([0, 0, 0]), "aaa")


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)