"""
Character-level tokenizer for MicroLSTM.

This module provides a simple character-level tokenization system that converts
text into numerical indices and back. The tokenizer builds a vocabulary from
a given text corpus and provides methods for encoding and decoding text.
"""

from typing import List, Dict, Set
import json
import os

from .exceptions import TokenizationError, FileOperationError


class CharacterTokenizer:
    """
    A character-level tokenizer that converts text to numerical indices and vice versa.
    
    This tokenizer operates at the character level, meaning each individual character
    (including spaces, punctuation, etc.) gets its own unique integer ID. This approach
    is simpler than word-level tokenization and works well for educational purposes
    and small-scale language modeling.
    
    The tokenizer builds its vocabulary from a provided text corpus, ensuring that
    all characters in the training data can be properly encoded.
    
    Attributes:
        char_to_idx (Dict[str, int]): Maps characters to their integer indices
        idx_to_char (Dict[int, str]): Maps integer indices back to characters
        _vocab_size (int): The total number of unique characters in the vocabulary
    
    Example:
        >>> tokenizer = CharacterTokenizer("hello world")
        >>> encoded = tokenizer.encode("hello")
        >>> print(encoded)  # [0, 1, 2, 2, 3] (example indices)
        >>> decoded = tokenizer.decode(encoded)
        >>> print(decoded)  # "hello"
    """
    
    def __init__(self, text_corpus: str):
        """
        Initialize the tokenizer by building vocabulary from the provided text corpus.
        
        The tokenizer analyzes the input text to identify all unique characters
        and assigns each character a unique integer index. The mapping is deterministic
        and based on sorted character order for consistency.
        
        Args:
            text_corpus (str): The text data used to build the character vocabulary.
                              All characters in this corpus will be included in the
                              vocabulary and can be encoded/decoded.
        
        Raises:
            ValueError: If the text_corpus is empty or None.
        
        Example:
            >>> tokenizer = CharacterTokenizer("Hello, World!")
            >>> print(tokenizer.vocab_size())  # 10 (unique characters)
        """
        if not text_corpus:
            raise TokenizationError(
                "Text corpus cannot be empty or None",
                text=text_corpus,
                operation="vocabulary_building"
            )
        
        # Extract unique characters from the corpus
        # Using a set to automatically handle duplicates
        unique_chars: Set[str] = set(text_corpus)
        
        # Sort characters for consistent ordering across runs
        # This ensures that the same text corpus always produces the same vocabulary
        sorted_chars: List[str] = sorted(list(unique_chars))
        
        # Build character-to-index mapping
        # Each character gets a unique integer ID starting from 0
        self.char_to_idx: Dict[str, int] = {
            char: idx for idx, char in enumerate(sorted_chars)
        }
        
        # Build index-to-character mapping for decoding
        # This is the reverse mapping used to convert indices back to text
        self.idx_to_char: Dict[int, str] = {
            idx: char for char, idx in self.char_to_idx.items()
        }
        
        # Cache vocabulary size for efficient access
        self._vocab_size: int = len(self.char_to_idx)
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text into a list of integer indices.
        
        Each character in the input text is mapped to its corresponding integer
        index based on the vocabulary built during initialization. This process
        converts human-readable text into numerical data that can be processed
        by neural networks.
        
        Args:
            text (str): The text to encode into numerical indices.
        
        Returns:
            List[int]: A list of integer indices representing the input text.
                      Each index corresponds to a character in the original text.
        
        Raises:
            ValueError: If the text contains characters not in the vocabulary.
                       This happens when trying to encode text with characters
                       that weren't present in the original training corpus.
        
        Example:
            >>> tokenizer = CharacterTokenizer("abc")
            >>> indices = tokenizer.encode("cab")
            >>> print(indices)  # [2, 0, 1] (example mapping)
        """
        if not isinstance(text, str):
            raise TokenizationError(
                "Input must be a string",
                text=text,
                operation="encoding"
            )
        
        encoded_indices: List[int] = []
        
        # Process each character in the input text
        for char in text:
            if char not in self.char_to_idx:
                raise TokenizationError(
                    f"Character '{char}' not found in vocabulary. "
                    f"Available characters: {sorted(self.char_to_idx.keys())}",
                    text=text,
                    character=char,
                    operation="encoding"
                )
            
            # Map character to its corresponding index
            encoded_indices.append(self.char_to_idx[char])
        
        return encoded_indices
    
    def decode(self, indices: List[int]) -> str:
        """
        Convert a list of integer indices back into text.
        
        This is the reverse operation of encode(). Each integer index is mapped
        back to its corresponding character, and the characters are joined to
        form the original text string.
        
        Args:
            indices (List[int]): A list of integer indices to decode into text.
                               Each index should correspond to a valid character
                               in the vocabulary.
        
        Returns:
            str: The decoded text string reconstructed from the indices.
        
        Raises:
            ValueError: If any index is not valid (not in the vocabulary).
                       This happens when trying to decode indices that don't
                       correspond to any character in the vocabulary.
            TypeError: If indices is not a list or contains non-integer values.
        
        Example:
            >>> tokenizer = CharacterTokenizer("abc")
            >>> text = tokenizer.decode([0, 1, 2])
            >>> print(text)  # "abc"
        """
        if not isinstance(indices, list):
            raise TokenizationError(
                "Indices must be a list",
                tokens=indices,
                operation="decoding"
            )
        
        decoded_chars: List[str] = []
        
        # Process each index in the input list
        for idx in indices:
            if not isinstance(idx, int):
                raise TokenizationError(
                    f"All indices must be integers, got {type(idx)}",
                    tokens=indices,
                    invalid_token=idx,
                    operation="decoding"
                )
            
            if idx not in self.idx_to_char:
                raise TokenizationError(
                    f"Index {idx} not found in vocabulary. "
                    f"Valid range: 0-{self._vocab_size - 1}",
                    tokens=indices,
                    invalid_token=idx,
                    operation="decoding"
                )
            
            # Map index back to its corresponding character
            decoded_chars.append(self.idx_to_char[idx])
        
        # Join all characters to form the final text string
        return ''.join(decoded_chars)
    
    def vocab_size(self) -> int:
        """
        Get the size of the character vocabulary.
        
        Returns the total number of unique characters in the tokenizer's
        vocabulary. This is useful for setting up neural network architectures
        that need to know the input/output dimensions.
        
        Returns:
            int: The number of unique characters in the vocabulary.
        
        Example:
            >>> tokenizer = CharacterTokenizer("hello")
            >>> print(tokenizer.vocab_size())  # 4 (h, e, l, o)
        """
        return self._vocab_size
    
    def get_vocab(self) -> Dict[str, int]:
        """
        Get a copy of the character-to-index vocabulary mapping.
        
        Returns a dictionary that maps each character in the vocabulary to its
        corresponding integer index. This is useful for inspecting the tokenizer's
        internal mapping or for debugging purposes.
        
        Returns:
            Dict[str, int]: A dictionary mapping characters to their indices.
                           The returned dictionary is a copy, so modifications
                           won't affect the tokenizer's internal state.
        
        Example:
            >>> tokenizer = CharacterTokenizer("abc")
            >>> vocab = tokenizer.get_vocab()
            >>> print(vocab)  # {'a': 0, 'b': 1, 'c': 2}
        """
        # Return a copy to prevent external modification of internal state
        return self.char_to_idx.copy()
    
    def save_vocab(self, filepath: str) -> None:
        """
        Save the vocabulary to a JSON file.
        
        This method allows persisting the tokenizer's vocabulary for later use.
        The vocabulary is saved as a JSON file containing the character-to-index
        mapping, which can be loaded later to recreate the same tokenizer.
        
        Args:
            filepath (str): Path where the vocabulary file should be saved.
        
        Raises:
            IOError: If the file cannot be written to the specified path.
        
        Example:
            >>> tokenizer = CharacterTokenizer("hello world")
            >>> tokenizer.save_vocab("vocab.json")
        """
        if not isinstance(filepath, str) or not filepath.strip():
            raise FileOperationError(
                "Filepath must be a non-empty string",
                filepath=filepath,
                operation="save_vocabulary"
            )
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.char_to_idx, f, ensure_ascii=False, indent=2)
        except (IOError, OSError, PermissionError) as e:
            raise FileOperationError(
                f"Failed to save vocabulary to {filepath}: {e}",
                filepath=filepath,
                operation="save_vocabulary",
                original_error=str(e)
            )
    
    @classmethod
    def load_vocab(cls, filepath: str) -> 'CharacterTokenizer':
        """
        Load a tokenizer from a saved vocabulary file.
        
        This class method creates a new CharacterTokenizer instance from a
        previously saved vocabulary file. This is useful for loading a tokenizer
        that was trained on a specific corpus without needing the original text.
        
        Args:
            filepath (str): Path to the vocabulary JSON file.
        
        Returns:
            CharacterTokenizer: A new tokenizer instance with the loaded vocabulary.
        
        Raises:
            IOError: If the file cannot be read from the specified path.
            ValueError: If the file format is invalid or corrupted.
        
        Example:
            >>> tokenizer = CharacterTokenizer.load_vocab("vocab.json")
            >>> print(tokenizer.vocab_size())
        """
        if not isinstance(filepath, str) or not filepath.strip():
            raise FileOperationError(
                "Filepath must be a non-empty string",
                filepath=filepath,
                operation="load_vocabulary"
            )
        
        if not os.path.exists(filepath):
            raise FileOperationError(
                f"Vocabulary file not found: {filepath}",
                filepath=filepath,
                operation="load_vocabulary"
            )
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                char_to_idx = json.load(f)
        except (IOError, OSError, PermissionError) as e:
            raise FileOperationError(
                f"Failed to load vocabulary from {filepath}: {e}",
                filepath=filepath,
                operation="load_vocabulary",
                original_error=str(e)
            )
        except json.JSONDecodeError as e:
            raise FileOperationError(
                f"Invalid JSON format in {filepath}: {e}",
                filepath=filepath,
                operation="load_vocabulary",
                original_error=str(e)
            )
        
        # Validate loaded vocabulary format
        if not isinstance(char_to_idx, dict):
            raise FileOperationError(
                f"Invalid vocabulary format: expected dictionary, got {type(char_to_idx)}",
                filepath=filepath,
                operation="load_vocabulary"
            )
        
        # Validate vocabulary contents
        for char, idx in char_to_idx.items():
            if not isinstance(char, str):
                raise FileOperationError(
                    f"Invalid character in vocabulary: {char} (type: {type(char)})",
                    filepath=filepath,
                    operation="load_vocabulary"
                )
            if not isinstance(idx, int) or idx < 0:
                raise FileOperationError(
                    f"Invalid index for character '{char}': {idx}",
                    filepath=filepath,
                    operation="load_vocabulary"
                )
        
        # Create a new instance and set the vocabulary directly
        # We use a dummy corpus since we're loading from file
        instance = cls.__new__(cls)
        instance.char_to_idx = char_to_idx
        instance.idx_to_char = {idx: char for char, idx in char_to_idx.items()}
        instance._vocab_size = len(char_to_idx)
        
        return instance
    
    def __repr__(self) -> str:
        """
        Return a string representation of the tokenizer.
        
        Returns:
            str: A string describing the tokenizer and its vocabulary size.
        """
        return f"CharacterTokenizer(vocab_size={self._vocab_size})"
    
    def __len__(self) -> int:
        """
        Return the vocabulary size when len() is called on the tokenizer.
        
        Returns:
            int: The vocabulary size.
        """
        return self._vocab_size