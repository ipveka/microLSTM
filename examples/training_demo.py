#!/usr/bin/env python3
"""
Comprehensive Training Demo for MicroLSTM

This script demonstrates the complete training pipeline for MicroLSTM,
from data preparation through model training to evaluation. It serves as both a
working example and an educational tutorial showing how to train character-level
language models from scratch.

Key Learning Objectives:
1. Understanding the complete training pipeline
2. Data preparation and preprocessing techniques
3. Model configuration and hyperparameter selection
4. Training loop implementation with progress monitoring
5. Model evaluation and text generation validation
6. Best practices for training neural language models

This demo uses sample text data and provides extensive comments explaining
each step of the process, making it ideal for learning purposes.
"""

import torch
import torch.nn as nn
from pathlib import Path
import time
import json
from typing import Dict, List, Any

# Import Micro LM components
from micro_lm import (
    CharacterTokenizer, MicroLM, ModelTrainer, TextGenerator,
    TrainingError, ModelConfigurationError
)


def get_sample_training_data() -> str:
    """
    Provide sample text data for training demonstration.
    
    In a real application, you would load text from files, but for this demo
    we provide a curated sample that demonstrates various text patterns the
    model can learn from.
    
    Returns:
        str: Sample training text with diverse patterns
    """
    sample_text = """
    The art of language modeling lies in understanding patterns within text.
    Neural networks excel at discovering these hidden relationships between words and characters.
    
    Consider how we naturally predict the next word in a sentence. When we read "The cat sat on the...",
    our brain automatically suggests "mat" or "chair" as likely continuations. Language models
    work similarly, but they learn these patterns from vast amounts of text data.
    
    Character-level models, like the one we're training, operate at an even more fundamental level.
    They learn relationships between individual characters, discovering patterns like:
    - Common letter combinations (th, er, ing, tion)
    - Word boundaries and punctuation usage
    - Capitalization patterns at sentence beginnings
    - Spelling conventions and morphological rules
    
    Training a language model involves several key steps:
    
    1. Data Preparation: Converting raw text into numerical sequences
    2. Model Architecture: Designing networks that can capture sequential patterns
    3. Training Process: Using backpropagation to adjust model parameters
    4. Evaluation: Testing the model's ability to generate coherent text
    
    The beauty of neural language models is their ability to generalize from training data.
    A well-trained model doesn't just memorize text—it learns the underlying structure
    of language itself. This enables it to generate novel, coherent text that follows
    the same patterns it observed during training.
    
    Modern applications of language models include:
    - Text completion and autocorrect systems
    - Creative writing assistance tools
    - Code generation and programming aids
    - Translation and summarization systems
    - Conversational AI and chatbots
    
    The field continues to evolve rapidly, with new architectures and training techniques
    emerging regularly. However, the fundamental principles remain the same: learn patterns
    from data, and use those patterns to generate new, meaningful content.
    
    This training demo will walk you through each step of the process, from tokenization
    to text generation, helping you understand how these remarkable systems work.
    Remember that language modeling is both an art and a science—it requires careful
    attention to data quality, model architecture, and training procedures.
    
    As you experiment with different configurations and datasets, you'll develop
    an intuition for what works well in different scenarios. The key is to start
    simple, understand the basics thoroughly, and then gradually explore more
    advanced techniques and larger models.
    
    Happy learning, and welcome to the fascinating world of neural language modeling!
    """
    
    # Clean up the text (remove extra whitespace while preserving structure)
    lines = [line.strip() for line in sample_text.strip().split('\n') if line.strip()]
    return '\n'.join(lines)


def demonstrate_data_analysis(text: str, tokenizer: CharacterTokenizer) -> Dict[str, Any]:
    """
    Analyze the training data to understand its characteristics.
    
    Data analysis is a crucial first step in any machine learning project.
    Understanding your data helps you make informed decisions about model
    architecture, training parameters, and expected performance.
    
    Args:
        text (str): Training text data
        tokenizer (CharacterTokenizer): Initialized tokenizer
    
    Returns:
        Dict[str, Any]: Analysis results and statistics
    """
    print("\n" + "="*60)
    print("DATA ANALYSIS")
    print("="*60)
    
    # Basic text statistics
    char_count = len(text)
    word_count = len(text.split())
    line_count = len(text.split('\n'))
    
    print(f"Text Statistics:")
    print(f"  Total characters: {char_count:,}")
    print(f"  Total words: {word_count:,}")
    print(f"  Total lines: {line_count:,}")
    print(f"  Average words per line: {word_count/line_count:.1f}")
    print(f"  Average characters per word: {char_count/word_count:.1f}")
    
    # Vocabulary analysis
    vocab = tokenizer.get_vocab()
    vocab_size = tokenizer.vocab_size()
    
    print(f"\nVocabulary Analysis:")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Characters: {sorted(vocab.keys())}")
    
    # Character frequency analysis
    char_freq = {}
    for char in text:
        char_freq[char] = char_freq.get(char, 0) + 1
    
    # Sort by frequency
    sorted_chars = sorted(char_freq.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nMost common characters:")
    for char, freq in sorted_chars[:10]:
        char_display = repr(char) if char in [' ', '\n', '\t'] else char
        percentage = (freq / char_count) * 100
        print(f"  {char_display}: {freq:,} ({percentage:.1f}%)")
    
    # Sequence length recommendations
    print(f"\nSequence Length Recommendations:")
    avg_word_length = char_count / word_count
    print(f"  Average word length: {avg_word_length:.1f} characters")
    print(f"  Recommended sequence lengths:")
    print(f"    Short sequences (word-level): {int(avg_word_length * 3)}-{int(avg_word_length * 5)}")
    print(f"    Medium sequences (phrase-level): {int(avg_word_length * 8)}-{int(avg_word_length * 12)}")
    print(f"    Long sequences (sentence-level): {int(avg_word_length * 15)}-{int(avg_word_length * 20)}")
    
    return {
        'char_count': char_count,
        'word_count': word_count,
        'line_count': line_count,
        'vocab_size': vocab_size,
        'char_frequencies': dict(sorted_chars),
        'avg_word_length': avg_word_length
    }


def create_model_configurations() -> Dict[str, Dict[str, Any]]:
    """
    Define different model configurations for various use cases.
    
    Model configuration is crucial for balancing performance, training time,
    and memory usage. Different configurations are suitable for different
    scenarios and computational constraints.
    
    Returns:
        Dict[str, Dict[str, Any]]: Named model configurations
    """
    configurations = {
        'tiny': {
            'description': 'Minimal model for quick experimentation and learning',
            'vocab_size': None,  # Will be set based on data
            'embedding_dim': 32,
            'hidden_dim': 64,
            'num_layers': 1,
            'dropout': 0.1,
            'use_case': 'Quick prototyping, educational demos, resource-constrained environments',
            'expected_params': '~10K parameters',
            'training_time': 'Very fast (minutes)',
            'memory_usage': 'Very low (<50MB)'
        },
        
        'small': {
            'description': 'Compact model with reasonable performance',
            'vocab_size': None,
            'embedding_dim': 64,
            'hidden_dim': 128,
            'num_layers': 2,
            'dropout': 0.2,
            'use_case': 'Small datasets, mobile deployment, educational projects',
            'expected_params': '~50K parameters',
            'training_time': 'Fast (tens of minutes)',
            'memory_usage': 'Low (<100MB)'
        },
        
        'medium': {
            'description': 'Balanced model for general-purpose use',
            'vocab_size': None,
            'embedding_dim': 128,
            'hidden_dim': 256,
            'num_layers': 2,
            'dropout': 0.2,
            'use_case': 'Medium datasets, good quality text generation, research',
            'expected_params': '~200K parameters',
            'training_time': 'Moderate (hours)',
            'memory_usage': 'Moderate (200-500MB)'
        },
        
        'large': {
            'description': 'High-capacity model for best performance',
            'vocab_size': None,
            'embedding_dim': 256,
            'hidden_dim': 512,
            'num_layers': 3,
            'dropout': 0.3,
            'use_case': 'Large datasets, high-quality generation, production systems',
            'expected_params': '~1M+ parameters',
            'training_time': 'Slow (many hours)',
            'memory_usage': 'High (1GB+)'
        }
    }
    
    return configurations


def demonstrate_model_configuration(configs: Dict[str, Dict[str, Any]], vocab_size: int):
    """
    Demonstrate different model configurations and their trade-offs.
    
    Args:
        configs (Dict): Model configurations
        vocab_size (int): Vocabulary size from tokenizer
    """
    print("\n" + "="*60)
    print("MODEL CONFIGURATION OPTIONS")
    print("="*60)
    
    for name, config in configs.items():
        print(f"\n{name.upper()} Configuration:")
        print(f"  Description: {config['description']}")
        print(f"  Architecture:")
        print(f"    Vocabulary size: {vocab_size}")
        print(f"    Embedding dimension: {config['embedding_dim']}")
        print(f"    Hidden dimension: {config['hidden_dim']}")
        print(f"    Number of layers: {config['num_layers']}")
        print(f"    Dropout rate: {config['dropout']}")
        print(f"  Characteristics:")
        print(f"    Expected parameters: {config['expected_params']}")
        print(f"    Training time: {config['training_time']}")
        print(f"    Memory usage: {config['memory_usage']}")
        print(f"  Best for: {config['use_case']}")


def create_and_analyze_model(config: Dict[str, Any], vocab_size: int) -> MicroLM:
    """
    Create a model with the specified configuration and analyze its properties.
    
    Args:
        config (Dict): Model configuration
        vocab_size (int): Vocabulary size
    
    Returns:
        MicroLM: Created and analyzed model
    """
    print(f"\nCreating model with configuration...")
    
    # Create model
    model = MicroLM(
        vocab_size=vocab_size,
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )
    
    # Analyze model properties
    model_info = model.get_model_info()
    
    print(f"Model Analysis:")
    print(f"  Total parameters: {model_info['parameters']['total']:,}")
    print(f"  Model size: {model_info['model_size_mb']} MB")
    print(f"  Architecture breakdown:")
    for component, params in model_info['parameters'].items():
        if component != 'total':
            percentage = (params / model_info['parameters']['total']) * 100
            print(f"    {component}: {params:,} ({percentage:.1f}%)")
    
    return model


def demonstrate_training_process(
    model: MicroLM, 
    tokenizer: CharacterTokenizer, 
    text: str,
    config_name: str
) -> Dict[str, List[float]]:
    """
    Demonstrate the complete training process with detailed explanations.
    
    This function walks through each step of training a language model,
    providing educational commentary on what's happening at each stage.
    
    Args:
        model (MicroLM): Model to train
        tokenizer (CharacterTokenizer): Text tokenizer
        text (str): Training text
        config_name (str): Configuration name for logging
    
    Returns:
        Dict[str, List[float]]: Training history
    """
    print("\n" + "="*60)
    print("TRAINING PROCESS DEMONSTRATION")
    print("="*60)
    
    # Step 1: Initialize trainer
    print("\nStep 1: Initializing trainer...")
    trainer = ModelTrainer(model, tokenizer)
    
    # Step 2: Prepare training data
    print("\nStep 2: Preparing training data...")
    
    # Choose sequence length based on model size
    sequence_lengths = {'tiny': 20, 'small': 30, 'medium': 40, 'large': 50}
    sequence_length = sequence_lengths.get(config_name, 30)
    
    # Choose batch size based on model size
    batch_sizes = {'tiny': 16, 'small': 12, 'medium': 8, 'large': 4}
    batch_size = batch_sizes.get(config_name, 8)
    
    print(f"  Sequence length: {sequence_length} characters")
    print(f"  Batch size: {batch_size} sequences")
    print(f"  Stride: 1 (overlapping sequences for more training data)")
    
    data_loader = trainer.prepare_data(
        text=text,
        sequence_length=sequence_length,
        batch_size=batch_size,
        stride=1,
        shuffle=True
    )
    
    print(f"  Created {len(data_loader)} training batches")
    print(f"  Total training sequences: {len(data_loader.dataset)}")
    
    # Step 3: Configure training parameters
    print("\nStep 3: Configuring training parameters...")
    
    # Adjust training parameters based on model size
    learning_rates = {'tiny': 0.003, 'small': 0.002, 'medium': 0.001, 'large': 0.0005}
    learning_rate = learning_rates.get(config_name, 0.001)
    
    epochs = {'tiny': 20, 'small': 30, 'medium': 40, 'large': 50}
    num_epochs = epochs.get(config_name, 30)
    
    print(f"  Learning rate: {learning_rate}")
    print(f"  Number of epochs: {num_epochs}")
    print(f"  Optimizer: Adam with weight decay")
    print(f"  Gradient clipping: 1.0 (prevents exploding gradients)")
    
    # Step 4: Start training
    print("\nStep 4: Starting training process...")
    print("  The model will learn to predict the next character in sequences")
    print("  Loss should decrease over time as the model learns patterns")
    print("  Training progress will be displayed for each epoch")
    
    start_time = time.time()
    
    try:
        # Train the model
        training_history = trainer.train(
            data_loader=data_loader,
            epochs=num_epochs,
            learning_rate=learning_rate,
            optimizer_type='adam',
            weight_decay=1e-5,
            gradient_clip_norm=1.0,
            save_every=max(num_epochs // 3, 5),  # Save 3 times during training
            save_path=f"./models/{config_name}_demo"
        )
        
        training_time = time.time() - start_time
        
        # Step 5: Analyze training results
        print(f"\nStep 5: Training completed successfully!")
        print(f"  Total training time: {training_time/60:.1f} minutes")
        print(f"  Final loss: {training_history['train_loss'][-1]:.6f}")
        print(f"  Best loss: {min(training_history['train_loss']):.6f}")
        
        # Calculate loss improvement
        initial_loss = training_history['train_loss'][0]
        final_loss = training_history['train_loss'][-1]
        improvement = ((initial_loss - final_loss) / initial_loss) * 100
        print(f"  Loss improvement: {improvement:.1f}%")
        
        return training_history
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return trainer.training_history
    except Exception as e:
        print(f"\nTraining failed: {e}")
        raise


def demonstrate_text_generation(model: MicroLM, tokenizer: CharacterTokenizer):
    """
    Demonstrate text generation capabilities with various sampling strategies.
    
    Args:
        model (MicroLM): Trained model
        tokenizer (CharacterTokenizer): Text tokenizer
    """
    print("\n" + "="*60)
    print("TEXT GENERATION DEMONSTRATION")
    print("="*60)
    
    # Initialize generator
    generator = TextGenerator(model, tokenizer)
    
    # Test prompts that should work well with our training data
    test_prompts = [
        "The art of",
        "Neural networks",
        "Language models",
        "Training a model"
    ]
    
    # Demonstrate different generation strategies
    strategies = [
        {'name': 'Greedy (Deterministic)', 'temperature': 0.0, 'description': 'Always picks most likely character'},
        {'name': 'Conservative', 'temperature': 0.3, 'description': 'Low randomness, coherent text'},
        {'name': 'Balanced', 'temperature': 0.7, 'description': 'Good balance of creativity and coherence'},
        {'name': 'Creative', 'temperature': 1.0, 'description': 'Higher randomness, more creative'}
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")
        print("-" * 40)
        
        for strategy in strategies:
            try:
                generated_text = generator.generate(
                    prompt=prompt,
                    length=80,
                    temperature=strategy['temperature']
                )
                
                # Extract just the generated part (after the prompt)
                generated_part = generated_text[len(prompt):]
                
                print(f"{strategy['name']} (T={strategy['temperature']}):")
                print(f"  {prompt}{generated_part}")
                print(f"  ({strategy['description']})")
                print()
                
            except Exception as e:
                print(f"  Generation failed: {e}")
                print()


def save_training_results(
    config_name: str,
    training_history: Dict[str, List[float]],
    model_info: Dict[str, Any],
    data_stats: Dict[str, Any]
):
    """
    Save training results and metadata for future reference.
    
    Args:
        config_name (str): Configuration name
        training_history (Dict): Training metrics
        model_info (Dict): Model information
        data_stats (Dict): Data statistics
    """
    results_dir = Path("./training_results")
    results_dir.mkdir(exist_ok=True)
    
    results = {
        'config_name': config_name,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'training_history': training_history,
        'model_info': model_info,
        'data_statistics': data_stats,
        'final_metrics': {
            'final_loss': training_history['train_loss'][-1],
            'best_loss': min(training_history['train_loss']),
            'total_epochs': len(training_history['train_loss']),
            'total_training_time': sum(training_history['epoch_times'])
        }
    }
    
    results_file = results_dir / f"{config_name}_training_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTraining results saved to: {results_file}")


def main():
    """
    Main demonstration function that orchestrates the complete training pipeline.
    
    This function demonstrates the entire process of training a character-level
    language model, from data preparation through model training to text generation.
    It serves as both a working example and an educational tutorial.
    """
    print("MICROLSTM - COMPREHENSIVE TRAINING DEMO")
    print("="*80)
    print("This demo walks through the complete process of training a character-level")
    print("language model, providing detailed explanations at each step.")
    print("="*80)
    
    try:
        # Step 1: Prepare sample data
        print("\n🔍 STEP 1: DATA PREPARATION")
        sample_text = get_sample_training_data()
        print(f"Loaded sample text: {len(sample_text)} characters")
        
        # Step 2: Create tokenizer and analyze data
        print("\n🔍 STEP 2: TOKENIZATION AND DATA ANALYSIS")
        tokenizer = CharacterTokenizer(sample_text)
        data_stats = demonstrate_data_analysis(sample_text, tokenizer)
        
        # Step 3: Show model configuration options
        print("\n🔍 STEP 3: MODEL CONFIGURATION")
        configs = create_model_configurations()
        demonstrate_model_configuration(configs, tokenizer.vocab_size())
        
        # Step 4: Choose configuration for demo
        # Use 'small' configuration for reasonable training time
        chosen_config = 'small'
        print(f"\n🔍 STEP 4: CREATING MODEL")
        print(f"Using '{chosen_config}' configuration for this demo...")
        
        config = configs[chosen_config].copy()
        config['vocab_size'] = tokenizer.vocab_size()
        
        model = create_and_analyze_model(config, tokenizer.vocab_size())
        
        # Step 5: Train the model
        print("\n🔍 STEP 5: MODEL TRAINING")
        training_history = demonstrate_training_process(
            model, tokenizer, sample_text, chosen_config
        )
        
        # Step 6: Demonstrate text generation
        print("\n🔍 STEP 6: TEXT GENERATION")
        demonstrate_text_generation(model, tokenizer)
        
        # Step 7: Save results
        print("\n🔍 STEP 7: SAVING RESULTS")
        model_info = model.get_model_info()
        save_training_results(chosen_config, training_history, model_info, data_stats)
        
        print("\n" + "="*80)
        print("TRAINING DEMO COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("Key takeaways from this demonstration:")
        print("• Data preparation is crucial for successful training")
        print("• Model configuration affects training time and performance")
        print("• Training involves iterative improvement through backpropagation")
        print("• Different sampling strategies produce different text styles")
        print("• Monitoring training progress helps identify issues early")
        print("\nNext steps you might consider:")
        print("• Try different model configurations (tiny, medium, large)")
        print("• Experiment with your own text data")
        print("• Adjust hyperparameters (learning rate, sequence length)")
        print("• Explore advanced generation techniques (top-k, top-p sampling)")
        print("• Train for more epochs to see continued improvement")
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user. Partial results may be available.")
    except Exception as e:
        print(f"\n\nDemo failed with error: {e}")
        print("This might be due to:")
        print("• Insufficient memory (try smaller model configuration)")
        print("• Missing dependencies (ensure PyTorch is installed)")
        print("• CUDA issues (try CPU-only training)")
        raise


if __name__ == "__main__":
    main()