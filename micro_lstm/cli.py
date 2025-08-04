#!/usr/bin/env python3
"""
Command-line interface for MicroLSTM.

This module provides a CLI for training, generating text, and managing
MicroLSTM models from the command line.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any

try:
    from . import (
        CharacterTokenizer, MicroLM, ModelTrainer, TextGenerator,
        ModelConfigurationError, TrainingError, GenerationError
    )
except ImportError:
    # For direct script execution
    from micro_lstm import (
        CharacterTokenizer, MicroLM, ModelTrainer, TextGenerator,
        ModelConfigurationError, TrainingError, GenerationError
    )


def create_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="MicroLSTM - Educational LSTM Language Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a model
  microlstm train --text-file data.txt --epochs 100 --save-model model.pt

  # Generate text
  microlstm generate --model model.pt --prompt "Hello" --length 100

  # Interactive generation
  microlstm generate --model model.pt --interactive

  # Analyze model
  microlstm analyze --model model.pt
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('--text-file', required=True, help='Path to training text file')
    train_parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    train_parser.add_argument('--embedding-dim', type=int, default=128, help='Embedding dimension')
    train_parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension')
    train_parser.add_argument('--num-layers', type=int, default=2, help='Number of LSTM layers')
    train_parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    train_parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    train_parser.add_argument('--sequence-length', type=int, default=50, help='Sequence length')
    train_parser.add_argument('--save-model', required=True, help='Path to save trained model')
    train_parser.add_argument('--save-tokenizer', help='Path to save tokenizer vocabulary')
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate text')
    generate_parser.add_argument('--model', required=True, help='Path to trained model')
    generate_parser.add_argument('--prompt', default='', help='Text prompt to start generation')
    generate_parser.add_argument('--length', type=int, default=100, help='Length of generated text')
    generate_parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    generate_parser.add_argument('--interactive', action='store_true', help='Interactive generation mode')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a trained model')
    analyze_parser.add_argument('--model', required=True, help='Path to trained model')
    analyze_parser.add_argument('--output', help='Path to save analysis report')
    
    return parser


def train_model(args: argparse.Namespace) -> None:
    """Train a new model."""
    print("🚀 Starting MicroLSTM training...")
    
    # Load training text
    try:
        with open(args.text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"📖 Loaded {len(text):,} characters from {args.text_file}")
    except FileNotFoundError:
        print(f"❌ Error: Training file '{args.text_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error reading training file: {e}")
        sys.exit(1)
    
    # Create tokenizer
    print("🔤 Creating tokenizer...")
    tokenizer = CharacterTokenizer(text)
    print(f"   Vocabulary size: {tokenizer.vocab_size()}")
    
    # Create model
    print("🧠 Creating model...")
    model = MicroLM(
        vocab_size=tokenizer.vocab_size(),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    )
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = ModelTrainer(model, tokenizer)
    
    # Prepare data
    print("📊 Preparing training data...")
    data_loader = trainer.prepare_data(
        text,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size
    )
    print(f"   Training batches: {len(data_loader)}")
    
    # Train model
    print(f"🎯 Training for {args.epochs} epochs...")
    try:
        history = trainer.train(
            data_loader,
            epochs=args.epochs,
            learning_rate=args.learning_rate
        )
        print("✅ Training completed successfully!")
    except TrainingError as e:
        print(f"❌ Training failed: {e}")
        sys.exit(1)
    
    # Save model
    print(f"💾 Saving model to {args.save_model}...")
    try:
        trainer.save_model(args.save_model)
        print("✅ Model saved successfully!")
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        sys.exit(1)
    
    # Save tokenizer if requested
    if args.save_tokenizer:
        print(f"💾 Saving tokenizer to {args.save_tokenizer}...")
        try:
            tokenizer.save_vocab(args.save_tokenizer)
            print("✅ Tokenizer saved successfully!")
        except Exception as e:
            print(f"❌ Error saving tokenizer: {e}")
    
    # Print training summary
    final_loss = history['loss'][-1] if history['loss'] else 'N/A'
    print(f"\n📈 Training Summary:")
    print(f"   Final loss: {final_loss}")
    print(f"   Model saved: {args.save_model}")
    if args.save_tokenizer:
        print(f"   Tokenizer saved: {args.save_tokenizer}")


def generate_text(args: argparse.Namespace) -> None:
    """Generate text using a trained model."""
    print("🎭 Starting text generation...")
    
    # Load model and tokenizer
    try:
        trainer = ModelTrainer.load_model(args.model)
        model = trainer.model
        tokenizer = trainer.tokenizer
        print(f"✅ Loaded model from {args.model}")
        print(f"   Vocabulary size: {tokenizer.vocab_size()}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)
    
    # Create generator
    generator = TextGenerator(model, tokenizer)
    
    if args.interactive:
        print("\n🎮 Interactive generation mode (type 'quit' to exit)")
        print("=" * 50)
        
        while True:
            try:
                prompt = input("\n💬 Enter prompt: ").strip()
                if prompt.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not prompt:
                    prompt = "The"
                
                length = int(input("📏 Length (default 100): ") or "100")
                temperature = float(input("🌡️  Temperature (default 1.0): ") or "1.0")
                
                print("\n🤖 Generating...")
                generated = generator.generate(
                    prompt, 
                    length=length, 
                    temperature=temperature
                )
                print(f"📝 Generated text:\n{generated}\n")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Generation error: {e}")
    else:
        # Single generation
        print(f"🎯 Generating {args.length} characters...")
        try:
            generated = generator.generate(
                args.prompt,
                length=args.length,
                temperature=args.temperature
            )
            print(f"\n📝 Generated text:\n{generated}")
        except GenerationError as e:
            print(f"❌ Generation failed: {e}")
            sys.exit(1)


def analyze_model(args: argparse.Namespace) -> None:
    """Analyze a trained model."""
    print("🔍 Analyzing model...")
    
    # Load model
    try:
        trainer = ModelTrainer.load_model(args.model)
        model = trainer.model
        tokenizer = trainer.tokenizer
        print(f"✅ Loaded model from {args.model}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)
    
    # Get model info
    model_info = model.get_model_info()
    
    # Print analysis
    print("\n📊 Model Analysis:")
    print("=" * 50)
    print(f"Vocabulary size: {model_info['vocab_size']}")
    print(f"Embedding dimension: {model_info['embedding_dim']}")
    print(f"Hidden dimension: {model_info['hidden_dim']}")
    print(f"Number of layers: {model_info['num_layers']}")
    print(f"Total parameters: {model_info['total_parameters']:,}")
    print(f"Trainable parameters: {model_info['trainable_parameters']:,}")
    
    # Architecture insights
    print(f"\n🏗️  Architecture Insights:")
    print(f"Model size: {model_info['model_size_mb']:.2f} MB")
    print(f"Memory usage: {model_info['memory_usage_mb']:.2f} MB")
    
    # Save analysis if requested
    if args.output:
        analysis_data = {
            'model_info': model_info,
            'tokenizer_info': {
                'vocab_size': tokenizer.vocab_size(),
                'vocabulary': tokenizer.get_vocab()
            }
        }
        
        try:
            with open(args.output, 'w') as f:
                json.dump(analysis_data, f, indent=2)
            print(f"✅ Analysis saved to {args.output}")
        except Exception as e:
            print(f"❌ Error saving analysis: {e}")


def main() -> None:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == 'train':
            train_model(args)
        elif args.command == 'generate':
            generate_text(args)
        elif args.command == 'analyze':
            analyze_model(args)
        else:
            print(f"❌ Unknown command: {args.command}")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main() 