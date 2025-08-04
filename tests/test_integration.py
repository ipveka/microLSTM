"""
Simplified integration tests for MicroLSTM.

This module contains focused integration tests that verify the complete
functionality of the language model system, including:
- Complete training pipeline from text to trained model
- End-to-end text generation workflow
- Model persistence (save/load) functionality
"""

import pytest
import torch
import tempfile
import os
from pathlib import Path

from micro_lstm.tokenizer import CharacterTokenizer
from micro_lstm.model import MicroLM
from micro_lstm.trainer import ModelTrainer
from micro_lstm.generator import TextGenerator


class TestCompleteTrainingPipeline:
    """Test the complete training pipeline from text input to trained model."""
    
    def test_end_to_end_training_workflow(self):
        """
        Test complete training workflow: text → tokenizer → model → training → generation.
        """
        # Sample training text
        training_text = """
        Language models are neural networks that learn to predict the next word or character
        in a sequence. They are trained on large amounts of text data and can generate
        coherent text by learning patterns in language.
        """
        
        # Step 1: Create tokenizer from training text
        tokenizer = CharacterTokenizer(training_text)
        vocab_size = tokenizer.vocab_size()
        
        # Verify tokenizer functionality
        assert vocab_size > 0
        assert len(tokenizer.get_vocab()) == vocab_size
        
        # Test round-trip encoding/decoding
        sample_text = "Language models"
        encoded = tokenizer.encode(sample_text)
        decoded = tokenizer.decode(encoded)
        assert decoded == sample_text
        
        # Step 2: Create model with appropriate architecture
        model = MicroLM(
            vocab_size=vocab_size,
            embedding_dim=64,
            hidden_dim=128,
            num_layers=2,
            dropout=0.1
        )
        
        # Verify model architecture
        model_info = model.get_model_info()
        assert model_info['architecture']['vocab_size'] == vocab_size
        assert model_info['parameters']['total'] > 0
        
        # Step 3: Initialize trainer
        trainer = ModelTrainer(model, tokenizer)
        
        # Verify trainer setup
        assert trainer.model is model
        assert trainer.tokenizer is tokenizer
        assert trainer.device is not None
        
        # Step 4: Prepare training data
        data_loader = trainer.prepare_data(
            text=training_text,
            sequence_length=20,
            batch_size=4,
            shuffle=True
        )
        
        # Verify data preparation
        assert data_loader is not None
        
        # Step 5: Train the model
        trainer.train(data_loader, epochs=2, learning_rate=0.01)
        
        # Verify training completed
        assert trainer.current_epoch > 0
        assert trainer.best_loss < float('inf')
        assert len(trainer.training_history['train_loss']) > 0
        
        # Step 6: Initialize generator
        generator = TextGenerator(model, tokenizer)
        
        # Step 7: Generate text
        prompt = "Language"
        generated_text = generator.generate(prompt, length=20)
        
        # Verify generation
        assert isinstance(generated_text, str)
        assert len(generated_text) >= len(prompt)
        assert generated_text.startswith(prompt)


class TestModelPersistence:
    """Test model persistence functionality."""
    
    def test_complete_model_persistence_workflow(self):
        """Test complete model save and load workflow."""
        # Setup components
        training_text = "Hello world example text for training the model."
        tokenizer = CharacterTokenizer(training_text)
        model = MicroLM(
            vocab_size=tokenizer.vocab_size(),
            embedding_dim=32,
            hidden_dim=64,
            num_layers=1
        )
        trainer = ModelTrainer(model, tokenizer)
        
        # Train the model
        data_loader = trainer.prepare_data(
            text=training_text,
            sequence_length=10,
            batch_size=2
        )
        trainer.train(data_loader, epochs=1, learning_rate=0.01)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            model_path = f.name
        
        try:
            trainer.save_model(model_path)
            assert os.path.exists(model_path)
            
            # Load model
            loaded_trainer = ModelTrainer(model, tokenizer)
            loaded_trainer.load_model(model_path)
            
            # Verify model parameters are preserved
            for param1, param2 in zip(trainer.model.parameters(), loaded_trainer.model.parameters()):
                assert torch.allclose(param1, param2)
            
            # Test generation with loaded model
            generator = TextGenerator(loaded_trainer.model, tokenizer)
            generated = generator.generate("Hello", length=10)
            
            assert isinstance(generated, str)
            assert generated.startswith("Hello")
            
        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)
    
    def test_tokenizer_vocabulary_persistence(self):
        """Test tokenizer vocabulary save and load."""
        # Create tokenizer
        text = "Hello world example text"
        tokenizer = CharacterTokenizer(text)
        
        # Save vocabulary
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            vocab_path = f.name
        
        try:
            tokenizer.save_vocab(vocab_path)
            assert os.path.exists(vocab_path)
            
            # Load vocabulary
            loaded_tokenizer = CharacterTokenizer.load_vocab(vocab_path)
            
            # Verify vocabulary is preserved
            assert loaded_tokenizer.vocab_size() == tokenizer.vocab_size()
            assert loaded_tokenizer.get_vocab() == tokenizer.get_vocab()
            
            # Test encoding/decoding still works
            test_text = "Hello"
            original_encoded = tokenizer.encode(test_text)
            loaded_encoded = loaded_tokenizer.encode(test_text)
            assert original_encoded == loaded_encoded
            
        finally:
            if os.path.exists(vocab_path):
                os.unlink(vocab_path)


class TestDifferentModelConfigurations:
    """Test different model configurations."""
    
    @pytest.mark.parametrize("config", [
        # Small model configuration
        {
            "vocab_size": 20,
            "embedding_dim": 16,
            "hidden_dim": 32,
            "num_layers": 1,
            "dropout": 0.0,
            "description": "minimal"
        },
        # Medium model configuration
        {
            "vocab_size": 50,
            "embedding_dim": 64,
            "hidden_dim": 128,
            "num_layers": 2,
            "dropout": 0.1,
            "description": "medium"
        }
    ])
    def test_different_model_configurations(self, config):
        """Test that different model configurations work correctly."""
        # Setup
        training_text = "Hello world example text for training."
        tokenizer = CharacterTokenizer(training_text)
        
        # Create model with given configuration
        model = MicroLM(
            vocab_size=tokenizer.vocab_size(),
            embedding_dim=config["embedding_dim"],
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            dropout=config["dropout"]
        )
        
        # Verify model info
        model_info = model.get_model_info()
        assert model_info['architecture']['embedding_dim'] == config["embedding_dim"]
        assert model_info['architecture']['hidden_dim'] == config["hidden_dim"]
        assert model_info['architecture']['num_layers'] == config["num_layers"]
        assert model_info['architecture']['dropout'] == config["dropout"]
        
        # Test forward pass
        batch_size = 2
        seq_length = 5
        input_tensor = torch.randint(0, tokenizer.vocab_size(), (batch_size, seq_length))
        output = model(input_tensor)
        
        assert output.shape == (batch_size, seq_length, tokenizer.vocab_size())
        assert not torch.isnan(output).any()
        
        # Test training
        trainer = ModelTrainer(model, tokenizer)
        data_loader = trainer.prepare_data(
            text=training_text,
            sequence_length=5,
            batch_size=2
        )
        trainer.train(data_loader, epochs=1, learning_rate=0.01)
        
        # Test generation
        generator = TextGenerator(model, tokenizer)
        generated = generator.generate("Hello", length=10)
        
        assert isinstance(generated, str)
        assert generated.startswith("Hello") 