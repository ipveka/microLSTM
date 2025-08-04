#!/usr/bin/env python3
"""
Simplified Training Demo for MicroLSTM

This script demonstrates basic model training for character-level language models.
"""

import torch
import torch.nn as nn
from pathlib import Path
import time
from typing import Dict, List

from micro_lstm import (
    CharacterTokenizer, MicroLM, ModelTrainer, TextGenerator, DataLoader
)


def get_sample_text() -> str:
    """Get sample text for training."""
    # Load TinyStories dataset
    loader = DataLoader()
    text, tokenizer, info = loader.quick_setup("roneneldan/TinyStories", preprocess=True)
    return text


def create_model(vocab_size: int) -> MicroLM:
    """Create a simple model."""
    return MicroLM(
        vocab_size=vocab_size,
        embedding_dim=64,
        hidden_dim=128,
        num_layers=2,
        dropout=0.1
    )


def train_model(model: MicroLM, tokenizer: CharacterTokenizer, text: str, 
                epochs: int = 10, sequence_length: int = 30, batch_size: int = 8):
    """Train the model."""
    print(f"Training for {epochs} epochs...")
    
    # Create trainer
    trainer = ModelTrainer(model, tokenizer)
    
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


def generate_text(model: MicroLM, tokenizer: CharacterTokenizer, prompt: str = "The", length: int = 50):
    """Generate text using the trained model."""
    generator = TextGenerator(model, tokenizer)
    generated = generator.generate(prompt, length=length, temperature=0.8)
    return generated


def main():
    """Main demonstration function."""
    print("🚀 TRAINING DEMO")
    print("=" * 40)
    
    # Get sample text
    text = get_sample_text()
    print(f"Training text length: {len(text)} characters")
    
    # Create tokenizer
    tokenizer = CharacterTokenizer(text)
    print(f"Vocabulary size: {tokenizer.vocab_size()}")
    
    # Create model
    model = create_model(tokenizer.vocab_size())
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("\nStarting training...")
    losses = train_model(model, tokenizer, text, epochs=5)
    
    # Generate text
    print("\nGenerating text...")
    generated = generate_text(model, tokenizer, "The", 30)
    print(f"Generated: '{generated}'")
    
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()