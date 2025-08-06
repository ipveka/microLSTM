#!/usr/bin/env python3
"""
MicroLSTM Big Model Training Script

This script trains a larger MicroLSTM model using configuration from config.yaml,
saves the trained model, and generates text samples.

Usage:
    python src/run.py
"""

import os
import sys
import yaml
import torch
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from micro_lstm import (
    MicroLSTM, CharacterTokenizer, ModelTrainer, 
    TextGenerator, DataLoader, ModelInspector
)
from micro_lstm.exceptions import ModelConfigurationError, TrainingError


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    logger = logging.getLogger(__name__)
    
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file {config_path} not found!")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file: {e}")
        sys.exit(1)


def setup_device(config: Dict[str, Any]) -> torch.device:
    """Setup the device (CPU/GPU) based on configuration."""
    logger = logging.getLogger(__name__)
    
    if config['hardware']['use_gpu'] and torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device('cpu')
        if config['hardware']['use_gpu']:
            logger.warning("GPU requested but not available, using CPU")
        else:
            logger.info("Using CPU")
    
    return device


def load_dataset(config: Dict[str, Any]) -> tuple:
    """Load and prepare the dataset."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading dataset: {config['dataset']['name']}")
    
    # Initialize data loader
    loader = DataLoader()
    
    # Load dataset with configuration
    text, tokenizer, info = loader.quick_setup(
        dataset_name=config['dataset']['name'],
        preprocess=config['dataset']['preprocess']
    )
    
    # Truncate text if needed
    if len(text) > config['dataset']['max_chars']:
        text = text[:config['dataset']['max_chars']]
        logger.info(f"Text truncated to {config['dataset']['max_chars']:,} characters")
    
    logger.info(f"Dataset loaded: {len(text):,} characters")
    logger.info(f"Vocabulary size: {tokenizer.vocab_size()}")
    logger.info(f"Unique characters: {len(tokenizer.char_to_idx)}")
    
    return text, tokenizer, info


def create_model(config: Dict[str, Any], vocab_size: int) -> MicroLSTM:
    """Create the MicroLSTM model with specified configuration."""
    logger = logging.getLogger(__name__)
    
    model_config = config['model']
    
    logger.info("Creating MicroLSTM model with configuration:")
    logger.info(f"  - Embedding dimension: {model_config['embedding_dim']}")
    logger.info(f"  - Hidden dimension: {model_config['hidden_dim']}")
    logger.info(f"  - Number of layers: {model_config['num_layers']}")
    logger.info(f"  - Dropout: {model_config['dropout']}")
    
    model = MicroLSTM(
        vocab_size=vocab_size,
        embedding_dim=model_config['embedding_dim'],
        hidden_dim=model_config['hidden_dim'],
        num_layers=model_config['num_layers'],
        dropout=model_config['dropout']
    )
    
    # Calculate and log model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model created successfully!")
    logger.info(f"  - Total parameters: {total_params:,}")
    logger.info(f"  - Trainable parameters: {trainable_params:,}")
    logger.info(f"  - Model size: {total_params * 4 / 1024 / 1024:.1f} MB")
    
    return model


def train_model(model: MicroLSTM, tokenizer: CharacterTokenizer, 
                text: str, config: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Train the model with specified configuration."""
    logger = logging.getLogger(__name__)
    
    training_config = config['training']
    output_config = config['output']
    
    logger.info("Starting model training...")
    logger.info(f"  - Epochs: {training_config['epochs']}")
    logger.info(f"  - Batch size: {training_config['batch_size']}")
    logger.info(f"  - Sequence length: {training_config['sequence_length']}")
    logger.info(f"  - Learning rate: {training_config['learning_rate']}")
    
    # Initialize trainer
    trainer = ModelTrainer(model, tokenizer, device=device)
    
    # Prepare data
    data_loader = trainer.prepare_data(
        text=text,
        sequence_length=training_config['sequence_length'],
        batch_size=training_config['batch_size']
    )
    
    # Train the model
    history = trainer.train(
        data_loader=data_loader,
        epochs=training_config['epochs'],
        learning_rate=training_config['learning_rate'],
        gradient_clip_norm=training_config['gradient_clipping'],
        save_every=output_config['save_interval']
    )
    
    logger.info("Training completed successfully!")
    return history


def save_model(model: MicroLSTM, tokenizer: CharacterTokenizer, 
               config: Dict[str, Any], history: Dict[str, Any]):
    """Save the trained model and related files."""
    logger = logging.getLogger(__name__)
    
    output_config = config['output']
    save_dir = Path(output_config['save_dir'])
    save_dir.mkdir(exist_ok=True)
    
    # Create timestamp for unique naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{output_config['model_name']}_{timestamp}"
    
    # Save model
    model_path = save_dir / f"{model_name}.{output_config['save_format']}"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': config['model'],
        'vocab_size': model.vocab_size,
        'training_history': history,
        'timestamp': timestamp
    }, model_path)
    
    # Save tokenizer
    tokenizer_path = save_dir / f"{model_name}_tokenizer.pkl"
    tokenizer.save(tokenizer_path)
    
    # Save configuration
    config_path = save_dir / f"{model_name}_config.yaml"
    with open(config_path, 'w', encoding='utf-8') as file:
        yaml.dump(config, file, default_flow_style=False, indent=2)
    
    logger.info(f"Model saved successfully!")
    logger.info(f"  - Model: {model_path}")
    logger.info(f"  - Tokenizer: {tokenizer_path}")
    logger.info(f"  - Config: {config_path}")
    
    return model_path, tokenizer_path, config_path


def generate_text_samples(model: MicroLSTM, tokenizer: CharacterTokenizer, 
                         config: Dict[str, Any], device: torch.device):
    """Generate text samples using the trained model."""
    logger = logging.getLogger(__name__)
    
    generation_config = config['generation']
    
    logger.info("Generating text samples...")
    logger.info(f"  - Number of samples: {generation_config['num_samples']}")
    logger.info(f"  - Length per sample: {generation_config['length']}")
    logger.info(f"  - Temperature: {generation_config['temperature']}")
    
    # Initialize generator
    generator = TextGenerator(model, tokenizer, device=device)
    
    # Generate samples
    samples = []
    prompts = [
        "Once upon a time",
        "The little girl",
        "In a magical forest",
        "The brave knight",
        "Deep in the ocean"
    ]
    
    for i in range(generation_config['num_samples']):
        prompt = prompts[i % len(prompts)]
        logger.info(f"Generating sample {i+1}/{generation_config['num_samples']} with prompt: '{prompt}'")
        
        generated_text = generator.generate(
            prompt=prompt,
            length=generation_config['length'],
            temperature=generation_config['temperature'],
            top_k=generation_config['top_k'],
            top_p=generation_config['top_p']
        )
        
        samples.append({
            'prompt': prompt,
            'generated': generated_text,
            'full_text': prompt + generated_text
        })
        
        logger.info(f"Sample {i+1} generated successfully!")
    
    # Save generated samples
    output_config = config['output']
    save_dir = Path(output_config['save_dir'])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    samples_path = save_dir / f"generated_samples_{timestamp}.txt"
    with open(samples_path, 'w', encoding='utf-8') as file:
        file.write("MicroLSTM Generated Text Samples\n")
        file.write("=" * 50 + "\n\n")
        
        for i, sample in enumerate(samples, 1):
            file.write(f"Sample {i}:\n")
            file.write(f"Prompt: {sample['prompt']}\n")
            file.write(f"Generated: {sample['generated']}\n")
            file.write(f"Full Text: {sample['full_text']}\n")
            file.write("-" * 50 + "\n\n")
    
    logger.info(f"Generated samples saved to: {samples_path}")
    
    # Print a sample to console
    logger.info("\n" + "=" * 50)
    logger.info("SAMPLE GENERATED TEXT:")
    logger.info("=" * 50)
    logger.info(samples[0]['full_text'])
    logger.info("=" * 50)
    
    return samples


def main():
    """Main training pipeline."""
    logger = setup_logging()
    
    logger.info("Starting MicroLSTM Big Model Training Pipeline")
    logger.info("=" * 60)
    
    try:
        # Load configuration
        config = load_config()
        
        # Setup device
        device = setup_device(config)
        
        # Load dataset
        text, tokenizer, dataset_info = load_dataset(config)
        
        # Create model
        model = create_model(config, tokenizer.vocab_size())
        model = model.to(device)
        
        # Train model
        history = train_model(model, tokenizer, text, config, device)
        
        # Save model
        model_path, tokenizer_path, config_path = save_model(model, tokenizer, config, history)
        
        # Generate text samples
        samples = generate_text_samples(model, tokenizer, config, device)
        
        logger.info("Training pipeline completed successfully!")
        logger.info("=" * 60)
        
        # Print summary
        logger.info("SUMMARY:")
        logger.info(f"  - Model saved: {model_path}")
        logger.info(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"  - Training epochs: {config['training']['epochs']}")
        logger.info(f"  - Generated samples: {len(samples)}")
        logger.info(f"  - Device used: {device}")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main() 