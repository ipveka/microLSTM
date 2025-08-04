#!/usr/bin/env python3
"""
Simplified Model Inspection Demo for MicroLSTM

This script demonstrates basic model inspection and visualization capabilities.
"""

import torch
import numpy as np
from pathlib import Path

from micro_lstm import (
    MicroLM, CharacterTokenizer, ModelInspector, DataLoader
)


def create_sample_model(vocab_size=50):
    """Create a sample model for demonstration."""
    print("Creating sample model...")
    
    model = MicroLM(
        vocab_size=vocab_size,
        embedding_dim=64,
        hidden_dim=128,
        num_layers=2,
        dropout=0.1
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model


def create_sample_tokenizer():
    """Create a sample tokenizer."""
    print("Creating sample tokenizer...")
    
    # Load TinyStories dataset
    loader = DataLoader()
    text, tokenizer, info = loader.quick_setup("roneneldan/TinyStories", preprocess=True)
    print(f"Tokenizer created with vocabulary size: {tokenizer.vocab_size()}")
    return tokenizer, text


def demonstrate_basic_inspection(model):
    """Demonstrate basic model inspection."""
    print("\n🔍 BASIC MODEL INSPECTION")
    print("=" * 40)
    
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


def demonstrate_parameter_analysis(inspector):
    """Demonstrate parameter analysis."""
    print("\n📊 PARAMETER ANALYSIS")
    print("=" * 40)
    
    # Analyze parameters
    param_stats = inspector.get_parameter_statistics()
    
    print("\nParameter Statistics:")
    print("-" * 20)
    for layer_name, stats in param_stats['layer_stats'].items():
        print(f"\n{layer_name}:")
        print(f"  Mean: {stats['mean']:.6f}")
        print(f"  Std: {stats['std']:.6f}")
        print(f"  Min: {stats['min']:.6f}")
        print(f"  Max: {stats['max']:.6f}")


def demonstrate_activation_inspection(inspector, tokenizer):
    """Demonstrate activation inspection."""
    print("\n⚡ ACTIVATION INSPECTION")
    print("=" * 40)
    
    # Create sample input
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


def main():
    """Main demonstration function."""
    print("🔬 MODEL INSPECTION DEMO")
    print("=" * 40)
    
    # Create sample tokenizer first to get vocab size
    tokenizer, sample_text = create_sample_tokenizer()
    vocab_size = tokenizer.vocab_size()
    
    # Create sample model with matching vocabulary size
    model = create_sample_model(vocab_size=vocab_size)
    
    # Basic inspection
    demonstrate_basic_inspection(model)
    
    # Parameter analysis
    inspector = ModelInspector(model)
    demonstrate_parameter_analysis(inspector)
    
    # Activation inspection
    demonstrate_activation_inspection(inspector, tokenizer)
    
    print("\n✅ Inspection demo complete!")


if __name__ == "__main__":
    main()