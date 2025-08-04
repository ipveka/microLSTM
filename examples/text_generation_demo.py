#!/usr/bin/env python3
"""
Simplified Text Generation Demo for MicroLSTM

This script demonstrates basic text generation capabilities using MicroLSTM.
"""

import torch
from pathlib import Path
from typing import Dict, List

from micro_lstm import (
    CharacterTokenizer, MicroLM, TextGenerator, DataLoader
)


def get_sample_text() -> str:
    """Get sample text for training."""
    # Load TinyStories dataset
    loader = DataLoader()
    text, tokenizer, info = loader.quick_setup("roneneldan/TinyStories", preprocess=True)
    return text


def create_simple_model(vocab_size: int) -> MicroLM:
    """Create a simple model for demonstration."""
    return MicroLM(
        vocab_size=vocab_size,
        embedding_dim=64,
        hidden_dim=128,
        num_layers=2,
        dropout=0.1
    )


def train_quick_model(model: MicroLM, tokenizer: CharacterTokenizer, text: str):
    """Quickly train the model for demonstration."""
    from micro_lstm import ModelTrainer
    
    print("Training model quickly for demo...")
    trainer = ModelTrainer(model, tokenizer)
    
    # Prepare data and train
    data_loader = trainer.prepare_data(
        text=text,
        sequence_length=20,
        batch_size=8,
        shuffle=True
    )
    
    training_history = trainer.train(
        data_loader=data_loader,
        epochs=10,
        learning_rate=0.01
    )


def demonstrate_generation(generator: TextGenerator):
    """Demonstrate different generation strategies."""
    print("\n🎯 GENERATION DEMONSTRATIONS")
    print("=" * 40)
    
    prompts = ["The", "Neural", "Language"]
    
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        print("-" * 20)
        
        # Different temperatures
        temperatures = [0.5, 0.8, 1.2]
        for temp in temperatures:
            generated = generator.generate(prompt, length=30, temperature=temp)
            print(f"T={temp}: {generated}")


def interactive_generation(generator: TextGenerator):
    """Interactive text generation."""
    print("\n💬 INTERACTIVE GENERATION")
    print("=" * 40)
    print("Enter prompts to generate text (or 'quit' to exit):")
    
    while True:
        try:
            prompt = input("\nEnter prompt: ").strip()
            if prompt.lower() in ['quit', 'exit', 'q']:
                break
            
            if prompt:
                generated = generator.generate(prompt, length=50, temperature=0.8)
                print(f"Generated: {generated}")
            else:
                print("Please enter a prompt.")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Generation error: {e}")


def main():
    """Main demonstration function."""
    print("🎭 TEXT GENERATION DEMO")
    print("=" * 40)
    
    # Get sample text
    text = get_sample_text()
    print(f"Sample text length: {len(text)} characters")
    
    # Create tokenizer
    tokenizer = CharacterTokenizer(text)
    print(f"Vocabulary size: {tokenizer.vocab_size()}")
    
    # Create and train model
    model = create_simple_model(tokenizer.vocab_size())
    train_quick_model(model, tokenizer, text)
    
    # Create generator
    generator = TextGenerator(model, tokenizer)
    
    # Demonstrate generation
    demonstrate_generation(generator)
    
    # Interactive generation
    interactive_generation(generator)
    
    print("\n✅ Generation demo complete!")


if __name__ == "__main__":
    main()