#!/usr/bin/env python3
"""
Training and Model Inspection for MicroLSTM

This script demonstrates model training, inspection, and analysis capabilities.
"""

import torch
import torch.nn as nn
from pathlib import Path
import time
from typing import Dict, List

from micro_lstm import (
    CharacterTokenizer, MicroLSTM, ModelTrainer, TextGenerator, 
    DataLoader, ModelInspector
)


def get_sample_text() -> str:
    """Get sample text for training."""
    # Load TinyStories dataset
    loader = DataLoader()
    text, tokenizer, info = loader.quick_setup("roneneldan/TinyStories", preprocess=True)
    return text


def create_model(vocab_size: int) -> MicroLSTM:
    """Create a simple model."""
    return MicroLSTM(
        vocab_size=vocab_size,
        embedding_dim=64,
        hidden_dim=128,
        num_layers=2,
        dropout=0.1
    )


def inspect_model_before_training(model: MicroLSTM, tokenizer: CharacterTokenizer):
    """Inspect the model before training."""
    print("\n🔍 MODEL INSPECTION (Before Training)")
    print("=" * 50)
    
    # Create inspector
    inspector = ModelInspector(model)
    
    # Print model summary
    print("\n1. Model Summary:")
    print("-" * 20)
    inspector.print_model_summary()
    
    # Get architecture summary
    print("\n2. Architecture Analysis:")
    print("-" * 20)
    summary = inspector.get_architecture_summary()
    
    basic_info = summary['basic_info']
    print(f"Total parameters: {basic_info['parameters']['total']:,}")
    print(f"Model size: {basic_info['model_size_mb']:.2f} MB")
    print(f"Layers: {basic_info['architecture']['num_layers']}")
    print(f"Hidden dimensions: {basic_info['architecture']['hidden_dim']}")
    
    # Parameter analysis
    print("\n3. Parameter Statistics:")
    print("-" * 20)
    param_stats = inspector.get_parameter_statistics()
    
    for layer_name, stats in param_stats['layer_stats'].items():
        print(f"\n{layer_name}:")
        print(f"  Mean: {stats['mean']:.6f}")
        print(f"  Std: {stats['std']:.6f}")
        print(f"  Min: {stats['min']:.6f}")
        print(f"  Max: {stats['max']:.6f}")
    
    # Activation inspection
    print("\n4. Activation Inspection:")
    print("-" * 20)
    sample_text = "The quick brown fox"
    encoded = tokenizer.encode(sample_text)
    input_tensor = torch.tensor([encoded], dtype=torch.long)
    
    print(f"Sample input: '{sample_text}'")
    print(f"Encoded: {encoded}")
    
    # Get activations
    activations = inspector.inspect_activations(input_tensor)
    
    print("\nActivation shapes:")
    for layer_name, activation in activations.items():
        if hasattr(activation, 'shape'):
            print(f"  {layer_name}: {activation.shape}")
        else:
            print(f"  {layer_name}: {type(activation)}")


def train_model(model: MicroLSTM, tokenizer: CharacterTokenizer, text: str, 
                epochs: int = 5, sequence_length: int = 30, batch_size: int = 12):
    """Train the model."""
    print(f"\n🚀 TRAINING MODEL")
    print("=" * 50)
    print(f"Training for {epochs} epochs...")
    
    # Create trainer
    trainer = ModelTrainer(model, tokenizer)
    
    # Show device information
    device = next(model.parameters()).device
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Prepare data using trainer's method
    data_loader = trainer.prepare_data(
        text=text,
        sequence_length=sequence_length,
        batch_size=batch_size,
        shuffle=True
    )
    
    # Train the model
    training_history = trainer.train(
        data_loader=data_loader,
        epochs=epochs,
        learning_rate=0.01
    )
    
    return training_history['train_loss']


def inspect_model_after_training(model: MicroLSTM, tokenizer: CharacterTokenizer):
    """Inspect the model after training."""
    print("\n🔍 MODEL INSPECTION (After Training)")
    print("=" * 50)
    
    # Create inspector
    inspector = ModelInspector(model)
    
    # Get updated parameter statistics
    print("\n1. Updated Parameter Statistics:")
    print("-" * 20)
    param_stats = inspector.get_parameter_statistics()
    
    for layer_name, stats in param_stats['layer_stats'].items():
        print(f"\n{layer_name}:")
        print(f"  Mean: {stats['mean']:.6f}")
        print(f"  Std: {stats['std']:.6f}")
        print(f"  Min: {stats['min']:.6f}")
        print(f"  Max: {stats['max']:.6f}")
    
    # Activation inspection with trained model
    print("\n2. Activation Inspection (Trained Model):")
    print("-" * 20)
    sample_text = "The little girl"
    encoded = tokenizer.encode(sample_text)
    input_tensor = torch.tensor([encoded], dtype=torch.long)
    
    print(f"Sample input: '{sample_text}'")
    print(f"Encoded: {encoded}")
    
    # Get activations
    activations = inspector.inspect_activations(input_tensor)
    
    print("\nActivation shapes:")
    for layer_name, activation in activations.items():
        if hasattr(activation, 'shape'):
            print(f"  {layer_name}: {activation.shape}")
        else:
            print(f"  {layer_name}: {type(activation)}")


def generate_text(model: MicroLSTM, tokenizer: CharacterTokenizer, prompt: str = "The", length: int = 100):
    """Generate text using the trained model."""
    print(f"\n📝 TEXT GENERATION")
    print("=" * 50)
    
    # Show device information
    device = next(model.parameters()).device
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    generator = TextGenerator(model, tokenizer)
    generated = generator.generate(prompt, length=length, temperature=0.8)
    return generated


def main():
    """Main demonstration function."""
    print("🚀 TRAINING AND MODEL INSPECTION")
    print("=" * 50)
    
    # Get sample text
    text = get_sample_text()
    print(f"Training text length: {len(text)} characters")
    
    # Create tokenizer
    tokenizer = CharacterTokenizer(text)
    print(f"Vocabulary size: {tokenizer.vocab_size()}")
    
    # Create model
    model = create_model(tokenizer.vocab_size())
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Inspect model before training
    inspect_model_before_training(model, tokenizer)
    
    # Train model
    losses = train_model(model, tokenizer, text, epochs=5)
    
    # Inspect model after training
    inspect_model_after_training(model, tokenizer)
    
    # Generate text
    generated = generate_text(model, tokenizer, "The", 100)
    print(f"Generated: '{generated}'")
    
    print("\n✅ Training and inspection complete!")


if __name__ == "__main__":
    main() 