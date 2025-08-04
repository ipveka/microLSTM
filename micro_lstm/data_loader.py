#!/usr/bin/env python3
"""
Data Loading Utilities for MicroLSTM

This module provides utilities to download and prepare datasets from Hugging Face
for training character-level language models. It supports various text datasets
and provides preprocessing functions to clean and format the data.
"""

import os
import re
import requests
import zipfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import logging
from tqdm import tqdm
import json

try:
    from datasets import load_dataset, Dataset
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    print("Warning: Hugging Face datasets not available. Install with: pip install datasets")

from .tokenizer import CharacterTokenizer
from .exceptions import DataError


class DataLoader:
    """
    Utility class for downloading and preparing datasets for MicroLSTM training.
    
    This class provides methods to:
    - Download datasets from Hugging Face
    - Preprocess and clean text data
    - Split data into train/validation sets
    - Create character-level tokenizers
    - Save and load prepared datasets
    
    Args:
        cache_dir (str): Directory to cache downloaded datasets
        max_text_length (int): Maximum length of text sequences to process
        
    Example:
        >>> loader = DataLoader()
        >>> dataset = loader.load_huggingface_dataset("tiny_shakespeare")
        >>> train_data, val_data = loader.prepare_data(dataset, split_ratio=0.9)
    """
    
    def __init__(self, cache_dir: str = "./data_cache", max_text_length: int = 1000000):
        """
        Initialize the data loader.
        
        Args:
            cache_dir (str): Directory to cache downloaded datasets
            max_text_length (int): Maximum length of text sequences to process
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_text_length = max_text_length
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Available datasets
        self.available_datasets = {
            "roneneldan/TinyStories": {
                "name": "roneneldan/TinyStories",
                "description": "Simple stories for children (1MB)",
                "size": "~1MB",
                "type": "stories"
            },
            "wikitext-2-raw-v1": {
                "name": "wikitext-2-raw-v1",
                "description": "Wikipedia articles (2MB)",
                "size": "~2MB",
                "type": "encyclopedia"
            },
            "bookcorpus": {
                "name": "bookcorpus",
                "description": "Free books from Project Gutenberg (500MB)",
                "size": "~500MB",
                "type": "books"
            },
            "openwebtext": {
                "name": "openwebtext",
                "description": "Web content from Reddit (8GB)",
                "size": "~8GB",
                "type": "web_content"
            },
            "custom": {
                "name": "custom",
                "description": "Custom text file",
                "size": "variable",
                "type": "custom"
            }
        }
    
    def list_available_datasets(self) -> Dict[str, Dict]:
        """
        List all available datasets with their descriptions.
        
        Returns:
            Dict[str, Dict]: Dictionary of available datasets with metadata
        """
        return self.available_datasets
    
    def download_dataset(self, dataset_name: str, subset: Optional[str] = None) -> str:
        """
        Download a dataset from Hugging Face.
        
        Args:
            dataset_name (str): Name of the dataset to download
            subset (str, optional): Subset of the dataset to use
            
        Returns:
            str: Path to the downloaded dataset file
            
        Raises:
            DataError: If dataset download fails or Hugging Face is not available
        """
        if not HUGGINGFACE_AVAILABLE:
            raise DataError(
                "Hugging Face datasets not available. Install with: pip install datasets",
                suggestion="Run: pip install datasets"
            )
        
        try:
            self.logger.info(f"Downloading dataset: {dataset_name}")
            
            # Load dataset from Hugging Face
            if subset:
                dataset = load_dataset(dataset_name, subset, cache_dir=str(self.cache_dir))
            else:
                dataset = load_dataset(dataset_name, cache_dir=str(self.cache_dir))
            
            # Save to local file
            # Handle dataset names with slashes by replacing with underscores
            safe_name = dataset_name.replace('/', '_')
            output_path = self.cache_dir / f"{safe_name}.txt"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for split in dataset.keys():
                    self.logger.info(f"Processing split: {split}")
                    for item in tqdm(dataset[split], desc=f"Processing {split}"):
                        text = self._extract_text(item)
                        if text:
                            f.write(text + "\n\n")
            
            self.logger.info(f"Dataset saved to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            raise DataError(
                f"Failed to download dataset {dataset_name}: {str(e)}",
                dataset_name=dataset_name,
                original_error=str(e)
            )
    
    def _extract_text(self, item: Dict) -> str:
        """
        Extract text from a dataset item.
        
        Args:
            item (Dict): Dataset item
            
        Returns:
            str: Extracted text
        """
        # Common text field names
        text_fields = ['text', 'content', 'sentence', 'paragraph', 'article']
        
        for field in text_fields:
            if field in item and item[field]:
                return str(item[field]).strip()
        
        # If no text field found, try to concatenate all string fields
        text_parts = []
        for key, value in item.items():
            if isinstance(value, str) and value.strip():
                text_parts.append(value.strip())
        
        return " ".join(text_parts) if text_parts else ""
    
    def load_text_file(self, file_path: str) -> str:
        """
        Load text from a local file.
        
        Args:
            file_path (str): Path to the text file
            
        Returns:
            str: Content of the text file
            
        Raises:
            DataError: If file cannot be read
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise DataError(f"File not found: {file_path}")
            
            self.logger.info(f"Loading text file: {file_path}")
            
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252']
            content = None
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                raise DataError(f"Could not read file with any encoding: {file_path}")
            
            # Limit content length
            if len(content) > self.max_text_length:
                content = content[:self.max_text_length]
                self.logger.warning(f"Text truncated to {self.max_text_length} characters")
            
            return content
            
        except Exception as e:
            raise DataError(
                f"Failed to load text file {file_path}: {str(e)}",
                file_path=str(file_path),
                original_error=str(e)
            )
    
    def preprocess_text(self, text: str, 
                        remove_urls: bool = True,
                        remove_emails: bool = True,
                        normalize_whitespace: bool = True,
                        remove_special_chars: bool = False,
                        min_word_length: int = 1) -> str:
        """
        Preprocess and clean text data.
        
        Args:
            text (str): Raw text to preprocess
            remove_urls (bool): Remove URLs from text
            remove_emails (bool): Remove email addresses from text
            normalize_whitespace (bool): Normalize whitespace characters
            remove_special_chars (bool): Remove special characters
            min_word_length (int): Minimum word length to keep
            
        Returns:
            str: Preprocessed text
        """
        self.logger.info("Preprocessing text...")
        
        # Remove URLs
        if remove_urls:
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        if remove_emails:
            text = re.sub(r'\S+@\S+', '', text)
        
        # Normalize whitespace
        if normalize_whitespace:
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
        
        # Remove special characters (keep alphanumeric, spaces, and basic punctuation)
        if remove_special_chars:
            text = re.sub(r'[^a-zA-Z0-9\s.,!?;:()\'"\-]', '', text)
        
        # Remove very short words
        if min_word_length > 1:
            words = text.split()
            words = [word for word in words if len(word) >= min_word_length]
            text = ' '.join(words)
        
        return text
    
    def prepare_data(self, text: str, 
                     split_ratio: float = 0.9,
                     sequence_length: int = 100,
                     batch_size: int = 32,
                     shuffle: bool = True) -> Tuple[str, str]:
        """
        Prepare text data for training by splitting into train/validation sets.
        
        Args:
            text (str): Input text data
            split_ratio (float): Ratio of training data (0.0 to 1.0)
            sequence_length (int): Length of sequences for training
            batch_size (int): Batch size for training
            shuffle (bool): Whether to shuffle the data
            
        Returns:
            Tuple[str, str]: Paths to train and validation data files
        """
        self.logger.info("Preparing data for training...")
        
        # Calculate split point
        split_point = int(len(text) * split_ratio)
        
        # Split data
        train_text = text[:split_point]
        val_text = text[split_point:]
        
        # Save to files
        train_path = self.cache_dir / "train_data.txt"
        val_path = self.cache_dir / "val_data.txt"
        
        with open(train_path, 'w', encoding='utf-8') as f:
            f.write(train_text)
        
        with open(val_path, 'w', encoding='utf-8') as f:
            f.write(val_text)
        
        self.logger.info(f"Data prepared:")
        self.logger.info(f"  Training: {len(train_text):,} characters")
        self.logger.info(f"  Validation: {len(val_text):,} characters")
        self.logger.info(f"  Total: {len(text):,} characters")
        
        return str(train_path), str(val_path)
    
    def create_tokenizer(self, text: str) -> CharacterTokenizer:
        """
        Create a character tokenizer from text data.
        
        Args:
            text (str): Text data to create tokenizer from
            
        Returns:
            CharacterTokenizer: Created tokenizer
        """
        self.logger.info("Creating character tokenizer...")
        
        tokenizer = CharacterTokenizer(text)
        
        self.logger.info(f"Tokenizer created with vocabulary size: {tokenizer.vocab_size()}")
        self.logger.info(f"Vocabulary: {tokenizer.get_vocab()}")
        
        return tokenizer
    
    def get_dataset_info(self, text: str) -> Dict[str, any]:
        """
        Get information about a text dataset.
        
        Args:
            text (str): Text data to analyze
            
        Returns:
            Dict[str, any]: Dataset information
        """
        info = {
            'total_characters': len(text),
            'total_words': len(text.split()),
            'total_lines': len(text.split('\n')),
            'unique_characters': len(set(text)),
            'character_frequency': {},
            'word_frequency': {},
            'average_word_length': 0,
            'average_line_length': 0
        }
        
        # Character frequency
        char_freq = {}
        for char in text:
            char_freq[char] = char_freq.get(char, 0) + 1
        info['character_frequency'] = dict(sorted(char_freq.items(), key=lambda x: x[1], reverse=True)[:20])
        
        # Word frequency
        words = text.split()
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        info['word_frequency'] = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20])
        
        # Averages
        if words:
            info['average_word_length'] = sum(len(word) for word in words) / len(words)
        
        lines = text.split('\n')
        if lines:
            info['average_line_length'] = sum(len(line) for line in lines) / len(lines)
        
        return info
    
    def quick_setup(self, dataset_name: str = "roneneldan/TinyStories", 
                   preprocess: bool = True) -> Tuple[str, CharacterTokenizer, Dict]:
        """
        Quick setup for a dataset - download, preprocess, and prepare everything.
        
        Args:
            dataset_name (str): Name of the dataset to use
            preprocess (bool): Whether to preprocess the text
            
        Returns:
            Tuple[str, CharacterTokenizer, Dict]: Text data, tokenizer, and dataset info
        """
        self.logger.info(f"Quick setup for dataset: {dataset_name}")
        
        # Download dataset from Hugging Face
        if dataset_name in self.available_datasets and dataset_name != "custom":
            dataset_path = self.download_dataset(dataset_name)
            text = self.load_text_file(dataset_path)
        else:
            raise DataError(f"Unknown dataset: {dataset_name}")
        
        # Preprocess if requested
        if preprocess:
            text = self.preprocess_text(text)
        
        # Create tokenizer
        tokenizer = self.create_tokenizer(text)
        
        # Get dataset info
        info = self.get_dataset_info(text)
        
        self.logger.info("Quick setup complete!")
        
        return text, tokenizer, info


def download_sample_datasets():
    """
    Download a few sample datasets for testing and demonstration.
    """
    loader = DataLoader()
    
    print("📚 Downloading sample datasets...")
    
    # List available datasets
    datasets = loader.list_available_datasets()
    print("\nAvailable datasets:")
    for name, info in datasets.items():
        print(f"  {name}: {info['description']} ({info['size']})")
    
    # Download a small dataset for testing
    print(f"\nDownloading TinyStories dataset...")
    try:
        text, tokenizer, info = loader.quick_setup("roneneldan/TinyStories")
        print(f"✅ Successfully downloaded and prepared dataset:")
        print(f"   Characters: {info['total_characters']:,}")
        print(f"   Words: {info['total_words']:,}")
        print(f"   Vocabulary size: {tokenizer.vocab_size()}")
        print(f"   Sample text: {text[:200]}...")
    except Exception as e:
        print(f"❌ Failed to download dataset: {e}")
        print("Make sure you have internet connection and 'datasets' package installed:")
        print("pip install datasets")


if __name__ == "__main__":
    download_sample_datasets() 