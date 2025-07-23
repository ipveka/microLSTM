#!/usr/bin/env python3
"""
Data Preparation Demo for Micro Language Model

This script demonstrates comprehensive data preparation techniques for training
character-level language models. It covers data loading, cleaning, preprocessing,
analysis, and preparation for training.

Key Data Preparation Concepts Covered:
1. Text data loading from various sources
2. Data cleaning and normalization techniques
3. Character-level tokenization strategies
4. Sequence preparation and batching
5. Data quality analysis and validation
6. Train/validation/test splitting
7. Data augmentation techniques
8. Memory-efficient data loading

This demo provides practical examples of preparing real-world text data
for language model training, with extensive explanations of best practices.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
import re
from typing import List, Dict, Tuple, Any, Optional
import matplotlib.pyplot as plt
from collections import Counter
import unicodedata

# Import Micro LM components
from micro_lm import CharacterTokenizer, ModelConfigurationError


class TextDataset(Dataset):
    """
    Custom dataset class for character-level language modeling.
    
    This dataset efficiently handles large text corpora by creating
    overlapping sequences on-the-fly, reducing memory usage while
    providing comprehensive training data.
    """
    
    def __init__(self, text: str, tokenizer: CharacterTokenizer, 
                 sequence_length: int, stride: int = 1):
        """
        Initialize the text dataset.
        
        Args:
            text (str): Input text corpus
            tokenizer (CharacterTokenizer): Tokenizer for text conversion
            sequence_length (int): Length of each training sequence
            stride (int): Step size between sequences (1 = overlapping)
        """
        self.text = text
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.stride = stride
        
        # Encode the entire text
        self.encoded_text = tokenizer.encode(text)
        
        # Calculate number of sequences
        if len(self.encoded_text) < sequence_length + 1:
            raise ValueError(f"Text too short. Need at least {sequence_length + 1} characters, got {len(self.encoded_text)}")
        
        self.num_sequences = (len(self.encoded_text) - sequence_length) // stride
        
        print(f"Dataset created:")
        print(f"  Text length: {len(text):,} characters")
        print(f"  Encoded length: {len(self.encoded_text):,} tokens")
        print(f"  Sequence length: {sequence_length}")
        print(f"  Stride: {stride}")
        print(f"  Number of sequences: {self.num_sequences:,}")
    
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        """
        Get a training sequence and its target.
        
        Args:
            idx (int): Sequence index
        
        Returns:
            tuple: (input_sequence, target_sequence)
        """
        start_idx = idx * self.stride
        end_idx = start_idx + self.sequence_length
        
        # Input sequence
        input_seq = self.encoded_text[start_idx:end_idx]
        
        # Target sequence (shifted by one position)
        target_seq = self.encoded_text[start_idx + 1:end_idx + 1]
        
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)


def load_sample_texts() -> Dict[str, str]:
    """
    Load various sample texts to demonstrate different data types.
    
    Returns:
        Dict[str, str]: Dictionary of sample texts by category
    """
    print("📚 LOADING SAMPLE TEXT DATA")
    print("="*60)
    
    sample_texts = {
        'literature': """
        It was the best of times, it was the worst of times, it was the age of wisdom,
        it was the age of foolishness, it was the epoch of belief, it was the epoch of
        incredulity, it was the season of Light, it was the season of Darkness, it was the
        spring of hope, it was the winter of despair, we had everything before us, we had
        nothing before us, we were all going direct to Heaven, we were all going direct
        the other way – in short, the period was so far like the present period, that some
        of its noisiest authorities insisted on its being received, for good or for evil,
        in the superlative degree of comparison only.
        
        There were a king with a large jaw and a queen with a plain face, on the throne
        of England; there were a king with a large jaw and a queen with a fair face, on
        the throne of France. In both countries it was clearer than crystal to the lords
        of the State preserves of loaves and fishes, that things in general were settled
        for ever.
        """,
        
        'technical': """
        Machine learning is a subset of artificial intelligence that focuses on the
        development of algorithms and statistical models that enable computer systems
        to improve their performance on a specific task through experience.
        
        Neural networks are computing systems inspired by biological neural networks.
        They consist of interconnected nodes (neurons) that process information using
        a connectionist approach to computation. The connections between neurons have
        weights that are adjusted during training to minimize prediction errors.
        
        Deep learning is a subset of machine learning based on artificial neural
        networks with representation learning. Learning can be supervised, semi-supervised,
        or unsupervised. Deep learning architectures such as deep neural networks,
        deep belief networks, recurrent neural networks, and convolutional neural
        networks have been applied to fields including computer vision, speech
        recognition, natural language processing, and bioinformatics.
        """,
        
        'conversational': """
        "Hello there! How are you doing today?"
        "I'm doing great, thanks for asking! How about you?"
        "Pretty good, just working on some interesting projects."
        "That sounds exciting! What kind of projects?"
        "I'm learning about natural language processing and machine learning."
        "Oh wow, that's fascinating! Are you finding it challenging?"
        "It definitely has its moments, but it's really rewarding when things click."
        "I can imagine! The field is advancing so rapidly these days."
        "Absolutely. It's amazing what we can accomplish with modern AI techniques."
        "Do you have any favorite applications or use cases?"
        "I'm particularly interested in text generation and language understanding."
        "Those are such powerful capabilities. The possibilities seem endless!"
        """,
        
        'mixed_content': """
        # Data Science Project Report
        
        ## Introduction
        This report analyzes customer behavior data from Q3 2023.
        
        ### Key Findings:
        1. Customer retention increased by 15.3%
        2. Average order value: $47.82 (+12% YoY)
        3. Mobile traffic: 68.4% of total visits
        
        ## Methodology
        We used Python for data analysis:
        
        ```python
        import pandas as pd
        import numpy as np
        
        # Load data
        df = pd.read_csv('customer_data.csv')
        
        # Calculate metrics
        retention_rate = df['returning_customers'] / df['total_customers']
        print(f"Retention rate: {retention_rate:.2%}")
        ```
        
        ## Results & Recommendations
        - Implement mobile-first design (68% mobile users)
        - Focus on customer retention programs
        - Optimize checkout process for higher AOV
        
        Contact: data-team@company.com
        Date: 2023-10-15
        """
    }
    
    for category, text in sample_texts.items():
        char_count = len(text)
        word_count = len(text.split())
        print(f"  {category.title()}: {char_count:,} chars, {word_count:,} words")
    
    return sample_texts


def demonstrate_text_cleaning(raw_text: str) -> str:
    """
    Demonstrate various text cleaning and normalization techniques.
    
    Args:
        raw_text (str): Raw input text
    
    Returns:
        str: Cleaned and normalized text
    """
    print("\n🧹 TEXT CLEANING DEMONSTRATION")
    print("="*60)
    
    print(f"Original text length: {len(raw_text):,} characters")
    
    # Step 1: Unicode normalization
    print("\n1. Unicode Normalization:")
    normalized_text = unicodedata.normalize('NFKC', raw_text)
    print(f"   After normalization: {len(normalized_text):,} characters")
    
    # Step 2: Remove or replace problematic characters
    print("\n2. Character Filtering:")
    
    # Count character types before cleaning
    control_chars = sum(1 for c in normalized_text if unicodedata.category(c).startswith('C'))
    print(f"   Control characters found: {control_chars}")
    
    # Remove control characters except newlines and tabs
    cleaned_text = ''.join(c for c in normalized_text 
                          if not unicodedata.category(c).startswith('C') or c in '\n\t')
    print(f"   After removing control chars: {len(cleaned_text):,} characters")
    
    # Step 3: Normalize whitespace
    print("\n3. Whitespace Normalization:")
    
    # Replace multiple spaces with single space
    whitespace_normalized = re.sub(r' +', ' ', cleaned_text)
    
    # Replace multiple newlines with double newline (paragraph breaks)
    whitespace_normalized = re.sub(r'\n\s*\n\s*\n+', '\n\n', whitespace_normalized)
    
    # Remove trailing whitespace from lines
    lines = whitespace_normalized.split('\n')
    lines = [line.rstrip() for line in lines]
    whitespace_normalized = '\n'.join(lines)
    
    print(f"   After whitespace normalization: {len(whitespace_normalized):,} characters")
    
    # Step 4: Optional: Handle special cases
    print("\n4. Special Character Handling:")
    
    # Convert smart quotes to regular quotes
    final_text = whitespace_normalized.replace('"', '"').replace('"', '"')
    final_text = final_text.replace(''', "'").replace(''', "'")
    
    # Normalize dashes
    final_text = final_text.replace('—', ' - ').replace('–', ' - ')
    
    print(f"   After special character handling: {len(final_text):,} characters")
    
    # Show cleaning summary
    chars_removed = len(raw_text) - len(final_text)
    print(f"\nCleaning Summary:")
    print(f"   Characters removed: {chars_removed:,} ({chars_removed/len(raw_text)*100:.1f}%)")
    print(f"   Final text length: {len(final_text):,} characters")
    
    return final_text


def analyze_text_characteristics(text: str, title: str = "Text") -> Dict[str, Any]:
    """
    Perform comprehensive analysis of text characteristics.
    
    Args:
        text (str): Text to analyze
        title (str): Title for the analysis
    
    Returns:
        Dict[str, Any]: Analysis results
    """
    print(f"\n📊 TEXT ANALYSIS: {title.upper()}")
    print("="*60)
    
    # Basic statistics
    char_count = len(text)
    word_count = len(text.split())
    line_count = len(text.split('\n'))
    paragraph_count = len([p for p in text.split('\n\n') if p.strip()])
    
    print(f"Basic Statistics:")
    print(f"  Characters: {char_count:,}")
    print(f"  Words: {word_count:,}")
    print(f"  Lines: {line_count:,}")
    print(f"  Paragraphs: {paragraph_count:,}")
    print(f"  Avg chars per word: {char_count/word_count:.1f}")
    print(f"  Avg words per line: {word_count/line_count:.1f}")
    
    # Character frequency analysis
    char_freq = Counter(text)
    total_chars = sum(char_freq.values())
    
    print(f"\nCharacter Distribution:")
    print(f"  Unique characters: {len(char_freq)}")
    
    # Most common characters
    print(f"  Most common characters:")
    for char, count in char_freq.most_common(10):
        char_display = repr(char) if char in [' ', '\n', '\t'] else char
        percentage = (count / total_chars) * 100
        print(f"    {char_display}: {count:,} ({percentage:.1f}%)")
    
    # Character categories
    categories = {}
    for char in text:
        cat = unicodedata.category(char)
        categories[cat] = categories.get(cat, 0) + 1
    
    print(f"\nCharacter Categories:")
    for cat, count in sorted(categories.items()):
        percentage = (count / total_chars) * 100
        cat_name = {
            'Ll': 'Lowercase letters',
            'Lu': 'Uppercase letters', 
            'Nd': 'Decimal numbers',
            'Po': 'Other punctuation',
            'Zs': 'Space separators',
            'Cc': 'Control characters',
            'Pc': 'Connector punctuation',
            'Pd': 'Dash punctuation',
            'Pe': 'Close punctuation',
            'Pf': 'Final punctuation',
            'Pi': 'Initial punctuation',
            'Ps': 'Open punctuation',
            'Sc': 'Currency symbols',
            'Sk': 'Modifier symbols',
            'Sm': 'Math symbols',
            'So': 'Other symbols',
            'Zl': 'Line separators',
            'Zp': 'Paragraph separators'
        }.get(cat, cat)
        print(f"    {cat_name}: {count:,} ({percentage:.1f}%)")
    
    # Vocabulary richness
    words = text.lower().split()
    unique_words = len(set(words))
    vocab_richness = unique_words / len(words) if words else 0
    
    print(f"\nVocabulary Analysis:")
    print(f"  Total words: {len(words):,}")
    print(f"  Unique words: {unique_words:,}")
    print(f"  Vocabulary richness: {vocab_richness:.3f}")
    
    # Word length distribution
    word_lengths = [len(word) for word in words]
    if word_lengths:
        avg_word_length = sum(word_lengths) / len(word_lengths)
        max_word_length = max(word_lengths)
        print(f"  Average word length: {avg_word_length:.1f}")
        print(f"  Longest word length: {max_word_length}")
    
    return {
        'char_count': char_count,
        'word_count': word_count,
        'line_count': line_count,
        'paragraph_count': paragraph_count,
        'unique_chars': len(char_freq),
        'char_frequencies': dict(char_freq.most_common(50)),
        'vocab_richness': vocab_richness,
        'unique_words': unique_words,
        'avg_word_length': avg_word_length if word_lengths else 0
    }


def demonstrate_tokenization_strategies(text: str) -> Dict[str, CharacterTokenizer]:
    """
    Demonstrate different tokenization approaches and their trade-offs.
    
    Args:
        text (str): Text to tokenize
    
    Returns:
        Dict[str, CharacterTokenizer]: Different tokenizers
    """
    print("\n🔤 TOKENIZATION STRATEGIES DEMONSTRATION")
    print("="*60)
    
    tokenizers = {}
    
    # Strategy 1: Basic character tokenization
    print("\n1. Basic Character Tokenization:")
    basic_tokenizer = CharacterTokenizer(text)
    tokenizers['basic'] = basic_tokenizer
    
    vocab = basic_tokenizer.get_vocab()
    print(f"   Vocabulary size: {basic_tokenizer.vocab_size()}")
    print(f"   Characters: {sorted(vocab.keys())}")
    
    # Test encoding/decoding
    test_text = "Hello, world!"
    if all(c in vocab for c in test_text):
        encoded = basic_tokenizer.encode(test_text)
        decoded = basic_tokenizer.decode(encoded)
        print(f"   Test - Original: '{test_text}'")
        print(f"   Test - Encoded: {encoded}")
        print(f"   Test - Decoded: '{decoded}'")
        print(f"   Test - Round-trip successful: {test_text == decoded}")
    
    # Strategy 2: Case-insensitive tokenization
    print("\n2. Case-Insensitive Tokenization:")
    lowercase_text = text.lower()
    case_insensitive_tokenizer = CharacterTokenizer(lowercase_text)
    tokenizers['case_insensitive'] = case_insensitive_tokenizer
    
    print(f"   Vocabulary size: {case_insensitive_tokenizer.vocab_size()}")
    print(f"   Reduction from basic: {basic_tokenizer.vocab_size() - case_insensitive_tokenizer.vocab_size()}")
    
    # Strategy 3: Filtered tokenization (common characters only)
    print("\n3. Filtered Tokenization (Common Characters):")
    
    # Keep only common characters
    char_freq = Counter(text)
    common_chars = {char for char, count in char_freq.items() if count >= 5}
    filtered_text = ''.join(c if c in common_chars else ' ' for c in text)
    
    filtered_tokenizer = CharacterTokenizer(filtered_text)
    tokenizers['filtered'] = filtered_tokenizer
    
    print(f"   Original unique chars: {len(char_freq)}")
    print(f"   Common chars (freq >= 5): {len(common_chars)}")
    print(f"   Filtered vocabulary size: {filtered_tokenizer.vocab_size()}")
    
    # Strategy comparison
    print("\n4. Strategy Comparison:")
    print("   Strategy          | Vocab Size | Memory | Pros & Cons")
    print("   ------------------|------------|--------|------------------")
    
    strategies_info = {
        'basic': {
            'vocab_size': basic_tokenizer.vocab_size(),
            'memory': 'High',
            'pros_cons': 'Complete representation, but large vocab'
        },
        'case_insensitive': {
            'vocab_size': case_insensitive_tokenizer.vocab_size(),
            'memory': 'Medium',
            'pros_cons': 'Smaller vocab, loses case information'
        },
        'filtered': {
            'vocab_size': filtered_tokenizer.vocab_size(),
            'memory': 'Low',
            'pros_cons': 'Smallest vocab, may lose rare characters'
        }
    }
    
    for strategy, info in strategies_info.items():
        print(f"   {strategy:<17} | {info['vocab_size']:<10} | {info['memory']:<6} | {info['pros_cons']}")
    
    return tokenizers


def demonstrate_sequence_preparation(text: str, tokenizer: CharacterTokenizer):
    """
    Demonstrate different sequence preparation strategies for training.
    
    Args:
        text (str): Input text
        tokenizer (CharacterTokenizer): Tokenizer to use
    """
    print("\n📝 SEQUENCE PREPARATION DEMONSTRATION")
    print("="*60)
    
    encoded_text = tokenizer.encode(text)
    print(f"Encoded text length: {len(encoded_text):,} tokens")
    
    # Strategy 1: Non-overlapping sequences
    print("\n1. Non-overlapping Sequences:")
    sequence_length = 20
    
    non_overlapping_sequences = []
    for i in range(0, len(encoded_text) - sequence_length, sequence_length):
        seq = encoded_text[i:i + sequence_length]
        target = encoded_text[i + 1:i + sequence_length + 1]
        non_overlapping_sequences.append((seq, target))
    
    print(f"   Sequence length: {sequence_length}")
    print(f"   Number of sequences: {len(non_overlapping_sequences):,}")
    print(f"   Data utilization: {len(non_overlapping_sequences) * sequence_length / len(encoded_text):.1%}")
    
    # Strategy 2: Overlapping sequences (stride = 1)
    print("\n2. Overlapping Sequences (stride=1):")
    
    overlapping_sequences = []
    for i in range(len(encoded_text) - sequence_length):
        seq = encoded_text[i:i + sequence_length]
        target = encoded_text[i + 1:i + sequence_length + 1]
        overlapping_sequences.append((seq, target))
    
    print(f"   Sequence length: {sequence_length}")
    print(f"   Number of sequences: {len(overlapping_sequences):,}")
    print(f"   Data utilization: {len(overlapping_sequences) * sequence_length / len(encoded_text):.1f}x")
    
    # Strategy 3: Custom stride
    print("\n3. Custom Stride Sequences:")
    stride = 5
    
    stride_sequences = []
    for i in range(0, len(encoded_text) - sequence_length, stride):
        seq = encoded_text[i:i + sequence_length]
        target = encoded_text[i + 1:i + sequence_length + 1]
        stride_sequences.append((seq, target))
    
    print(f"   Sequence length: {sequence_length}")
    print(f"   Stride: {stride}")
    print(f"   Number of sequences: {len(stride_sequences):,}")
    print(f"   Data utilization: {len(stride_sequences) * sequence_length / len(encoded_text):.1%}")
    
    # Show example sequences
    print("\n4. Example Sequences:")
    if overlapping_sequences:
        for i, (seq, target) in enumerate(overlapping_sequences[:3]):
            seq_text = tokenizer.decode(seq)
            target_text = tokenizer.decode(target)
            print(f"   Sequence {i+1}:")
            print(f"     Input:  '{seq_text}'")
            print(f"     Target: '{target_text}'")
    
    # Memory and training implications
    print("\n5. Training Implications:")
    print("   Strategy        | Sequences | Memory | Training Speed | Data Coverage")
    print("   ----------------|-----------|--------|----------------|---------------")
    print(f"   Non-overlapping | {len(non_overlapping_sequences):<9,} | Low    | Fast           | Partial")
    print(f"   Overlapping     | {len(overlapping_sequences):<9,} | High   | Slow           | Complete")
    print(f"   Custom stride   | {len(stride_sequences):<9,} | Medium | Medium         | Good")


def demonstrate_data_splitting(text: str, tokenizer: CharacterTokenizer) -> Dict[str, str]:
    """
    Demonstrate proper data splitting for training, validation, and testing.
    
    Args:
        text (str): Input text
        tokenizer (CharacterTokenizer): Tokenizer to use
    
    Returns:
        Dict[str, str]: Split datasets
    """
    print("\n✂️  DATA SPLITTING DEMONSTRATION")
    print("="*60)
    
    # Split ratios
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    
    print(f"Split ratios - Train: {train_ratio:.0%}, Val: {val_ratio:.0%}, Test: {test_ratio:.0%}")
    
    # Calculate split points
    text_length = len(text)
    train_end = int(text_length * train_ratio)
    val_end = int(text_length * (train_ratio + val_ratio))
    
    # Split the text
    train_text = text[:train_end]
    val_text = text[train_end:val_end]
    test_text = text[val_end:]
    
    splits = {
        'train': train_text,
        'validation': val_text,
        'test': test_text
    }
    
    print(f"\nSplit Results:")
    for split_name, split_text in splits.items():
        char_count = len(split_text)
        word_count = len(split_text.split())
        percentage = (char_count / text_length) * 100
        
        print(f"  {split_name.title()}:")
        print(f"    Characters: {char_count:,} ({percentage:.1f}%)")
        print(f"    Words: {word_count:,}")
        
        # Check for vocabulary coverage
        split_chars = set(split_text)
        total_chars = set(text)
        coverage = len(split_chars) / len(total_chars) * 100
        print(f"    Character coverage: {coverage:.1f}%")
    
    # Validate splits don't have data leakage
    print(f"\nData Leakage Check:")
    
    # Check for overlapping sequences at boundaries
    boundary_length = 50  # Check 50 characters around boundaries
    
    train_end_snippet = train_text[-boundary_length:] if len(train_text) >= boundary_length else train_text
    val_start_snippet = val_text[:boundary_length] if len(val_text) >= boundary_length else val_text
    
    val_end_snippet = val_text[-boundary_length:] if len(val_text) >= boundary_length else val_text
    test_start_snippet = test_text[:boundary_length] if len(test_text) >= boundary_length else test_text
    
    print(f"  Train-Val boundary:")
    print(f"    Train end: '...{train_end_snippet[-20:]}'")
    print(f"    Val start: '{val_start_snippet[:20]}...'")
    
    print(f"  Val-Test boundary:")
    print(f"    Val end: '...{val_end_snippet[-20:]}'")
    print(f"    Test start: '{test_start_snippet[:20]}...'")
    
    return splits


def create_efficient_dataloader(text: str, tokenizer: CharacterTokenizer, 
                              sequence_length: int, batch_size: int, 
                              shuffle: bool = True) -> DataLoader:
    """
    Create an efficient DataLoader for training.
    
    Args:
        text (str): Training text
        tokenizer (CharacterTokenizer): Tokenizer
        sequence_length (int): Length of training sequences
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle data
    
    Returns:
        DataLoader: Configured data loader
    """
    print(f"\n🔄 CREATING EFFICIENT DATALOADER")
    print("="*60)
    
    # Create dataset
    dataset = TextDataset(text, tokenizer, sequence_length, stride=1)
    
    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Use 0 for compatibility
        pin_memory=torch.cuda.is_available(),
        drop_last=True  # Drop incomplete batches
    )
    
    print(f"DataLoader Configuration:")
    print(f"  Dataset size: {len(dataset):,} sequences")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of batches: {len(dataloader):,}")
    print(f"  Shuffle: {shuffle}")
    print(f"  Pin memory: {torch.cuda.is_available()}")
    print(f"  Drop last batch: True")
    
    # Test the dataloader
    print(f"\nDataLoader Test:")
    try:
        batch_iter = iter(dataloader)
        input_batch, target_batch = next(batch_iter)
        
        print(f"  Batch shapes: input {input_batch.shape}, target {target_batch.shape}")
        print(f"  Data type: {input_batch.dtype}")
        print(f"  Memory usage: ~{input_batch.numel() * 4 / 1024:.1f} KB per batch")
        
        # Show sample from batch
        sample_input = input_batch[0]
        sample_target = target_batch[0]
        
        sample_input_text = tokenizer.decode(sample_input.tolist())
        sample_target_text = tokenizer.decode(sample_target.tolist())
        
        print(f"  Sample input:  '{sample_input_text[:50]}...'")
        print(f"  Sample target: '{sample_target_text[:50]}...'")
        
    except Exception as e:
        print(f"  ❌ DataLoader test failed: {e}")
    
    return dataloader


def demonstrate_memory_optimization(text: str, tokenizer: CharacterTokenizer):
    """
    Demonstrate memory optimization techniques for large datasets.
    
    Args:
        text (str): Input text
        tokenizer (CharacterTokenizer): Tokenizer
    """
    print(f"\n💾 MEMORY OPTIMIZATION DEMONSTRATION")
    print("="*60)
    
    # Calculate memory requirements for different approaches
    text_size_mb = len(text.encode('utf-8')) / (1024 * 1024)
    encoded_size_mb = len(tokenizer.encode(text)) * 4 / (1024 * 1024)  # int32
    
    print(f"Memory Analysis:")
    print(f"  Original text: {text_size_mb:.2f} MB")
    print(f"  Encoded text: {encoded_size_mb:.2f} MB")
    
    # Strategy 1: Pre-encode and store
    print(f"\n1. Pre-encoding Strategy:")
    print(f"   Pros: Fast training, no repeated encoding")
    print(f"   Cons: High memory usage ({encoded_size_mb:.2f} MB)")
    print(f"   Best for: Small to medium datasets")
    
    # Strategy 2: On-the-fly encoding
    print(f"\n2. On-the-fly Encoding Strategy:")
    print(f"   Pros: Low memory usage ({text_size_mb:.2f} MB)")
    print(f"   Cons: Slower training due to repeated encoding")
    print(f"   Best for: Large datasets with memory constraints")
    
    # Strategy 3: Chunked processing
    print(f"\n3. Chunked Processing Strategy:")
    chunk_size = len(text) // 4  # Process in 4 chunks
    chunk_size_mb = chunk_size * 4 / (1024 * 1024)
    
    print(f"   Chunk size: {chunk_size:,} characters")
    print(f"   Memory per chunk: {chunk_size_mb:.2f} MB")
    print(f"   Pros: Balanced memory usage and speed")
    print(f"   Cons: More complex implementation")
    print(f"   Best for: Very large datasets")
    
    # Memory recommendations
    print(f"\n4. Memory Recommendations:")
    
    if encoded_size_mb < 100:
        print(f"   ✅ Small dataset: Use pre-encoding strategy")
    elif encoded_size_mb < 1000:
        print(f"   ⚠️  Medium dataset: Consider chunked processing")
    else:
        print(f"   🔴 Large dataset: Use on-the-fly encoding or chunking")
    
    # Show memory-efficient dataset implementation
    print(f"\n5. Memory-Efficient Dataset Example:")
    print(f"   class MemoryEfficientDataset(Dataset):")
    print(f"       def __init__(self, text_file, tokenizer, seq_len):")
    print(f"           self.text_file = text_file")
    print(f"           self.tokenizer = tokenizer")
    print(f"           # Store only file path, not content")
    print(f"       ")
    print(f"       def __getitem__(self, idx):")
    print(f"           # Read and encode only needed portion")
    print(f"           with open(self.text_file) as f:")
    print(f"               f.seek(start_position)")
    print(f"               chunk = f.read(chunk_size)")
    print(f"           return self.tokenizer.encode(chunk)")


def main():
    """
    Main function demonstrating comprehensive data preparation techniques.
    """
    print("MICRO LANGUAGE MODEL - DATA PREPARATION DEMO")
    print("="*80)
    print("This demo covers comprehensive data preparation techniques for")
    print("character-level language model training, from raw text to training-ready data.")
    print("="*80)
    
    try:
        # Step 1: Load sample texts
        sample_texts = load_sample_texts()
        
        # Step 2: Choose a text for detailed demonstration
        demo_text = sample_texts['mixed_content']  # Use mixed content for variety
        
        # Step 3: Demonstrate text cleaning
        cleaned_text = demonstrate_text_cleaning(demo_text)
        
        # Step 4: Analyze text characteristics
        analysis = analyze_text_characteristics(cleaned_text, "Cleaned Mixed Content")
        
        # Step 5: Demonstrate tokenization strategies
        tokenizers = demonstrate_tokenization_strategies(cleaned_text)
        chosen_tokenizer = tokenizers['basic']  # Use basic tokenizer for remaining demos
        
        # Step 6: Demonstrate sequence preparation
        demonstrate_sequence_preparation(cleaned_text, chosen_tokenizer)
        
        # Step 7: Demonstrate data splitting
        splits = demonstrate_data_splitting(cleaned_text, chosen_tokenizer)
        
        # Step 8: Create efficient data loader
        train_text = splits['train']
        dataloader = create_efficient_dataloader(
            text=train_text,
            tokenizer=chosen_tokenizer,
            sequence_length=30,
            batch_size=8,
            shuffle=True
        )
        
        # Step 9: Demonstrate memory optimization
        demonstrate_memory_optimization(cleaned_text, chosen_tokenizer)
        
        print("\n" + "="*80)
        print("DATA PREPARATION DEMO COMPLETED!")
        print("="*80)
        print("Key takeaways from this demonstration:")
        print("• Text cleaning is crucial for model performance")
        print("• Character analysis helps understand data characteristics")
        print("• Different tokenization strategies have different trade-offs")
        print("• Sequence preparation strategy affects training efficiency")
        print("• Proper data splitting prevents overfitting")
        print("• Memory optimization is important for large datasets")
        print("• Efficient data loading improves training speed")
        print("\nBest practices for data preparation:")
        print("• Always analyze your data before training")
        print("• Clean and normalize text consistently")
        print("• Choose tokenization strategy based on your needs")
        print("• Use appropriate sequence length for your model")
        print("• Implement proper train/validation/test splits")
        print("• Monitor memory usage during data loading")
        print("• Test your data pipeline before training")
        
    except Exception as e:
        print(f"\n\nDemo failed with error: {e}")
        print("This might be due to:")
        print("• Missing dependencies (torch, matplotlib)")
        print("• Insufficient memory for data processing")
        print("• Text encoding issues")
        raise


if __name__ == "__main__":
    main()