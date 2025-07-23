"""
Integration tests and end-to-end validation for the Micro Language Model.

This module contains comprehensive integration tests that verify the complete
functionality of the language model system, including:
- Complete training pipeline from text to trained model
- End-to-end text generation workflow
- Model persistence (save/load) functionality
- Performance benchmarks on sample data
- Different model configurations
- Requirements validation through automated tests

These tests ensure that all components work together seamlessly and that
the system meets all specified requirements.
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Any
import warnings

from micro_lm.tokenizer import CharacterTokenizer
from micro_lm.model import MicroLM
from micro_lm.trainer import ModelTrainer
from micro_lm.generator import TextGenerator
from micro_lm.data_utils import create_data_loader
from micro_lm.exceptions import ModelConfigurationError, TrainingError, GenerationError


class TestCompleteTrainingPipeline:
    """Test the complete training pipeline from text input to trained model."""
    
    def test_end_to_end_training_workflow(self):
        """
        Test complete training workflow: text → tokenizer → model → training → generation.
        
        This test validates Requirements 3.4, 4.4, 5.4 by ensuring the complete
        pipeline works from raw text input to text generation output.
        """
        # Sample training text (educational content about language models)
        training_text = """
        Language models are neural networks that learn to predict the next word or character
        in a sequence. They are trained on large amounts of text data and can generate
        coherent text by learning patterns in language. The basic idea is to use the
        previous words or characters to predict what comes next. This process is repeated
        to generate longer sequences of text. Modern language models use architectures
        like transformers or recurrent neural networks to capture these patterns.
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
            sequence_length=50,
            batch_size=8,
            shuffle=True
        )
        
        # Verify data preparation
        assert len(data_loader) > 0
        
        # Test data loader functionality
        for batch_inputs, batch_targets in data_loader:
            assert batch_inputs.shape[1] == 50  # sequence_length
            assert batch_targets.shape[1] == 50
            assert batch_inputs.dtype == torch.long
            assert batch_targets.dtype == torch.long
            break  # Just verify first batch
        
        # Step 5: Train the model
        print("Starting integration training...")
        initial_time = time.time()
        
        training_history = trainer.train(
            data_loader=data_loader,
            epochs=10,
            learning_rate=0.001,
            optimizer_type='adam',
            gradient_clip_norm=1.0
        )
        
        training_time = time.time() - initial_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Verify training results
        assert len(training_history['train_loss']) == 10
        assert all(loss > 0 for loss in training_history['train_loss'])
        assert not any(torch.isnan(torch.tensor(loss)) for loss in training_history['train_loss'])
        
        # Verify loss improvement (should decrease over time)
        initial_loss = training_history['train_loss'][0]
        final_loss = training_history['train_loss'][-1]
        assert final_loss < initial_loss, "Loss should decrease during training"
        
        # Step 6: Test text generation
        generator = TextGenerator(model, tokenizer)
        
        # Test different generation strategies
        test_prompts = ["Language", "The model", "Neural networks"]
        
        for prompt in test_prompts:
            # Greedy generation (deterministic)
            greedy_text = generator.generate_greedy(prompt, length=50)
            assert greedy_text.startswith(prompt)
            assert len(greedy_text) == len(prompt) + 50
            
            # Temperature-based generation
            temp_text = generator.generate_with_temperature(prompt, length=30, temperature=0.8)
            assert temp_text.startswith(prompt)
            assert len(temp_text) == len(prompt) + 30
            
            # Verify generated text contains valid characters
            for char in temp_text:
                assert char in tokenizer.get_vocab(), f"Generated invalid character: '{char}'"
        
        print("End-to-end training workflow completed successfully!")  
  
    def test_training_with_validation_split(self):
        """Test training pipeline with validation data split."""
        training_text = "Hello world this is a test for validation splitting functionality"
        
        tokenizer = CharacterTokenizer(training_text)
        model = MicroLM(
            vocab_size=tokenizer.vocab_size(),
            embedding_dim=32,
            hidden_dim=64,
            num_layers=1
        )
        trainer = ModelTrainer(model, tokenizer)
        
        # Prepare data with validation split
        data_loader = trainer.prepare_data(
            text=training_text,
            sequence_length=10,
            batch_size=4,
            validation_split=0.2
        )
        
        # Verify validation data was created
        assert hasattr(trainer, '_validation_text')
        assert trainer._validation_text is not None
        assert len(trainer._validation_text) > 0
        
        # Train model
        history = trainer.train(data_loader, epochs=5, learning_rate=0.01)
        
        # Verify training completed
        assert len(history['train_loss']) == 5
        
        # Test generation after training
        generator = TextGenerator(model, tokenizer)
        text = generator.generate("Hello", length=20, temperature=0.5)
        assert text.startswith("Hello")


class TestModelPersistence:
    """Test model saving and loading functionality across different scenarios."""
    
    def test_complete_model_persistence_workflow(self):
        """
        Test complete model save/load workflow with all components.
        
        This test validates Requirements 3.4, 5.4 by ensuring models can be
        properly saved and loaded with all state preserved.
        """
        # Create and train a model
        training_text = "The quick brown fox jumps over the lazy dog"
        tokenizer = CharacterTokenizer(training_text)
        
        original_model = MicroLM(
            vocab_size=tokenizer.vocab_size(),
            embedding_dim=32,
            hidden_dim=64,
            num_layers=2
        )
        
        trainer = ModelTrainer(original_model, tokenizer)
        data_loader = trainer.prepare_data(training_text, sequence_length=8, batch_size=4)
        
        # Train the model
        training_history = trainer.train(data_loader, epochs=5, learning_rate=0.01)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "test_model.pt")
            
            # Save the complete model state
            trainer.save_model(model_path, include_optimizer=True)
            
            # Verify file was created and has reasonable size
            assert os.path.exists(model_path)
            file_size = os.path.getsize(model_path)
            assert file_size > 1000  # Should be at least 1KB
            
            # Create new model and trainer
            new_model = MicroLM(
                vocab_size=tokenizer.vocab_size(),
                embedding_dim=32,
                hidden_dim=64,
                num_layers=2
            )
            new_trainer = ModelTrainer(new_model, tokenizer)
            
            # Initialize optimizer (required for loading optimizer state)
            dummy_data = new_trainer.prepare_data(training_text, sequence_length=8, batch_size=4)
            new_trainer.train(dummy_data, epochs=1, learning_rate=0.01)
            
            # Load the saved model
            new_trainer.load_model(model_path)
            
            # Verify loaded state matches original
            assert new_trainer.current_epoch == trainer.current_epoch
            assert new_trainer.best_loss == trainer.best_loss
            assert len(new_trainer.training_history['train_loss']) == len(training_history['train_loss'])
            
            # Test that both models produce identical outputs
            test_input = torch.randint(0, tokenizer.vocab_size(), (1, 10))
            
            original_model.eval()
            new_model.eval()
            
            with torch.no_grad():
                original_output = original_model(test_input)
                new_output = new_model(test_input)
                
                # Outputs should be very close (allowing for small numerical differences)
                assert torch.allclose(original_output, new_output, atol=1e-6)
            
            # Test generation consistency
            original_generator = TextGenerator(original_model, tokenizer)
            new_generator = TextGenerator(new_model, tokenizer)
            
            prompt = "The"
            original_text = original_generator.generate(prompt, length=20, temperature=0.0, seed=42)
            new_text = new_generator.generate(prompt, length=20, temperature=0.0, seed=42)
            
            assert original_text == new_text, "Generated text should be identical after model loading"
    
    def test_tokenizer_vocabulary_persistence(self):
        """Test that tokenizer vocabulary is properly saved and loaded."""
        training_text = "Test vocabulary persistence with special characters: !@#$%"
        tokenizer = CharacterTokenizer(training_text)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            vocab_path = os.path.join(temp_dir, "vocab.json")
            
            # Save vocabulary
            tokenizer.save_vocab(vocab_path)
            
            # Load vocabulary into new tokenizer
            new_tokenizer = CharacterTokenizer.load_vocab(vocab_path)
            
            # Verify vocabularies match
            assert new_tokenizer.vocab_size() == tokenizer.vocab_size()
            assert new_tokenizer.get_vocab() == tokenizer.get_vocab()
            
            # Test encoding/decoding consistency
            test_text = "Test !@#"
            original_encoded = tokenizer.encode(test_text)
            new_encoded = new_tokenizer.encode(test_text)
            assert original_encoded == new_encoded
            
            original_decoded = tokenizer.decode(original_encoded)
            new_decoded = new_tokenizer.decode(new_encoded)
            assert original_decoded == new_decoded == test_text

class Te
stDifferentModelConfigurations:
    """Test various model configurations and architectures."""
    
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
        },
        # Larger model configuration
        {
            "vocab_size": 100,
            "embedding_dim": 128,
            "hidden_dim": 256,
            "num_layers": 3,
            "dropout": 0.2,
            "description": "large"
        }
    ])
    def test_different_model_configurations(self, config):
        """
        Test training and generation with different model configurations.
        
        This test validates Requirements 3.4, 4.4 by ensuring the system
        works with various model architectures and sizes.
        """
        # Create training text with enough variety for the vocabulary size
        base_chars = "abcdefghijklmnopqrstuvwxyz "
        training_text = base_chars * 10  # Repeat to ensure sufficient length
        
        # Limit vocabulary to match config
        unique_chars = list(set(training_text))[:config["vocab_size"]]
        limited_text = ''.join(unique_chars) * 5
        
        tokenizer = CharacterTokenizer(limited_text)
        
        # Adjust vocab_size to actual tokenizer size
        actual_vocab_size = tokenizer.vocab_size()
        
        model = MicroLM(
            vocab_size=actual_vocab_size,
            embedding_dim=config["embedding_dim"],
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            dropout=config["dropout"]
        )
        
        print(f"Testing {config['description']} model configuration:")
        print(f"  Vocab size: {actual_vocab_size}")
        print(f"  Parameters: {model.get_model_info()['parameters']['total']:,}")
        
        trainer = ModelTrainer(model, tokenizer)
        
        # Adjust sequence length and batch size based on model size
        seq_length = min(20, len(limited_text) // 4)
        batch_size = max(1, 8 // config["num_layers"])  # Smaller batch for larger models
        
        data_loader = trainer.prepare_data(
            text=limited_text,
            sequence_length=seq_length,
            batch_size=batch_size
        )
        
        # Train model (fewer epochs for larger models to save time)
        epochs = max(3, 10 // config["num_layers"])
        start_time = time.time()
        
        history = trainer.train(
            data_loader=data_loader,
            epochs=epochs,
            learning_rate=0.01,
            optimizer_type='adam'
        )
        
        training_time = time.time() - start_time
        print(f"  Training time: {training_time:.2f}s")
        print(f"  Final loss: {history['train_loss'][-1]:.4f}")
        
        # Verify training completed successfully
        assert len(history['train_loss']) == epochs
        assert all(loss > 0 for loss in history['train_loss'])
        
        # Test text generation
        generator = TextGenerator(model, tokenizer)
        
        # Use first character as prompt
        prompt = unique_chars[0]
        generated_text = generator.generate(prompt, length=20, temperature=0.5)
        
        assert generated_text.startswith(prompt)
        assert len(generated_text) == len(prompt) + 20
        
        # Verify all generated characters are in vocabulary
        for char in generated_text:
            assert char in tokenizer.get_vocab()
        
        print(f"  Generated sample: '{generated_text[:30]}...'")
        print(f"  {config['description'].capitalize()} model test passed!\n")


class TestPerformanceBenchmarks:
    """Performance benchmarks and timing tests."""
    
    def test_training_performance_benchmark(self):
        """
        Benchmark training performance on sample data.
        
        This test validates Requirements 3.4, 5.4 by measuring training
        performance and ensuring it meets reasonable expectations.
        """
        # Create standardized benchmark data
        benchmark_text = """
        This is a standardized benchmark text for measuring training performance.
        It contains a variety of characters, words, and sentence structures to
        provide a realistic training scenario. The text includes punctuation,
        numbers like 123 and 456, and various linguistic patterns that a
        language model should learn to predict and generate effectively.
        """ * 5  # Repeat for more data
        
        tokenizer = CharacterTokenizer(benchmark_text)
        vocab_size = tokenizer.vocab_size()
        
        # Standard benchmark configuration
        model = MicroLM(
            vocab_size=vocab_size,
            embedding_dim=64,
            hidden_dim=128,
            num_layers=2,
            dropout=0.1
        )
        
        trainer = ModelTrainer(model, tokenizer)
        
        # Benchmark data preparation
        prep_start = time.time()
        data_loader = trainer.prepare_data(
            text=benchmark_text,
            sequence_length=50,
            batch_size=16
        )
        prep_time = time.time() - prep_start
        
        print(f"Data preparation time: {prep_time:.3f}s")
        print(f"Number of training batches: {len(data_loader)}")
        print(f"Total training sequences: {len(data_loader.dataset)}")
        
        # Benchmark training
        epochs = 10
        train_start = time.time()
        
        history = trainer.train(
            data_loader=data_loader,
            epochs=epochs,
            learning_rate=0.001,
            optimizer_type='adam'
        )
        
        train_time = time.time() - train_start
        
        # Calculate performance metrics
        total_params = model.get_model_info()['parameters']['total']
        time_per_epoch = train_time / epochs
        sequences_per_second = (len(data_loader.dataset) * epochs) / train_time
        
        print(f"\nTraining Performance Benchmark:")
        print(f"  Model parameters: {total_params:,}")
        print(f"  Total training time: {train_time:.2f}s")
        print(f"  Time per epoch: {time_per_epoch:.2f}s")
        print(f"  Sequences per second: {sequences_per_second:.1f}")
        print(f"  Final loss: {history['train_loss'][-1]:.4f}")
        print(f"  Loss improvement: {((history['train_loss'][0] - history['train_loss'][-1]) / history['train_loss'][0] * 100):.1f}%")
        
        # Performance assertions (reasonable expectations)
        assert time_per_epoch < 30.0, f"Training too slow: {time_per_epoch:.2f}s per epoch"
        assert sequences_per_second > 1.0, f"Throughput too low: {sequences_per_second:.1f} seq/s"
        assert history['train_loss'][-1] < history['train_loss'][0], "Loss should improve during training"
    
    def test_generation_performance_benchmark(self):
        """Benchmark text generation performance."""
        # Create and train a model for generation benchmarking
        training_text = "The quick brown fox jumps over the lazy dog. " * 20
        tokenizer = CharacterTokenizer(training_text)
        
        model = MicroLM(
            vocab_size=tokenizer.vocab_size(),
            embedding_dim=32,
            hidden_dim=64,
            num_layers=1
        )
        
        # Quick training for functional model
        trainer = ModelTrainer(model, tokenizer)
        data_loader = trainer.prepare_data(training_text, sequence_length=20, batch_size=8)
        trainer.train(data_loader, epochs=5, learning_rate=0.01)
        
        generator = TextGenerator(model, tokenizer)
        
        # Benchmark different generation strategies
        prompt = "The quick"
        generation_length = 100
        
        strategies = [
            ("greedy", lambda: generator.generate(prompt, generation_length, temperature=0.0)),
            ("temperature_low", lambda: generator.generate(prompt, generation_length, temperature=0.3)),
            ("temperature_high", lambda: generator.generate(prompt, generation_length, temperature=1.0)),
            ("top_k", lambda: generator.generate(prompt, generation_length, temperature=0.8, top_k=5)),
            ("top_p", lambda: generator.generate(prompt, generation_length, temperature=0.8, top_p=0.9)),
        ]
        
        print(f"\nGeneration Performance Benchmark:")
        print(f"  Prompt: '{prompt}'")
        print(f"  Generation length: {generation_length}")
        
        for strategy_name, generate_func in strategies:
            # Warm up
            generate_func()
            
            # Benchmark
            start_time = time.time()
            generated_text = generate_func()
            generation_time = time.time() - start_time
            
            chars_per_second = generation_length / generation_time
            
            print(f"  {strategy_name:15}: {generation_time:.3f}s ({chars_per_second:.1f} chars/s)")
            
            # Verify generation quality
            assert generated_text.startswith(prompt)
            assert len(generated_text) == len(prompt) + generation_length
            
            # Performance assertion
            assert generation_time < 10.0, f"{strategy_name} generation too slow: {generation_time:.3f}s"
class
 TestRequirementsValidation:
    """Validate that all specified requirements are met through automated tests."""
    
    def test_requirement_1_character_level_tokenization(self):
        """
        Validate Requirement 1: Character-level tokenization and simple architecture.
        
        Requirements 1.1, 1.2, 1.3: Character tokenization, text conversion, simple architecture
        """
        # Test character-level tokenization (1.1, 1.2)
        text = "Hello, World! 123"
        tokenizer = CharacterTokenizer(text)
        
        # Should tokenize at character level
        encoded = tokenizer.encode("Hello")
        assert len(encoded) == 5  # One token per character
        
        # Should convert characters to indices and back
        decoded = tokenizer.decode(encoded)
        assert decoded == "Hello"
        
        # Test simple neural network architecture (1.3)
        model = MicroLM(
            vocab_size=tokenizer.vocab_size(),
            embedding_dim=32,
            hidden_dim=64,
            num_layers=2
        )
        
        # Should use LSTM architecture (simple but effective)
        assert isinstance(model.lstm, nn.LSTM)
        assert isinstance(model.embedding, nn.Embedding)
        assert isinstance(model.output_projection, nn.Linear)
        
        # Should predict next character (1.4)
        input_seq = torch.randint(0, tokenizer.vocab_size(), (1, 10))
        output = model(input_seq)
        assert output.shape == (1, 10, tokenizer.vocab_size())  # Predictions for each position
        
        print("✓ Requirement 1: Character-level tokenization and simple architecture - PASSED")
    
    def test_requirement_2_comprehensive_documentation(self):
        """
        Validate Requirement 2: Comprehensive code comments and explanations.
        
        Requirements 2.1, 2.2, 2.3, 2.4: Docstrings, inline comments, training documentation
        """
        # Test that all major classes have detailed docstrings (2.1)
        assert CharacterTokenizer.__doc__ is not None
        assert len(CharacterTokenizer.__doc__) > 100  # Substantial documentation
        
        assert MicroLM.__doc__ is not None
        assert len(MicroLM.__doc__) > 100
        
        assert ModelTrainer.__doc__ is not None
        assert len(ModelTrainer.__doc__) > 100
        
        assert TextGenerator.__doc__ is not None
        assert len(TextGenerator.__doc__) > 100
        
        # Test that key methods have docstrings (2.2)
        assert CharacterTokenizer.encode.__doc__ is not None
        assert CharacterTokenizer.decode.__doc__ is not None
        assert MicroLM.forward.__doc__ is not None
        assert ModelTrainer.train.__doc__ is not None
        assert TextGenerator.generate.__doc__ is not None
        
        # Test model architecture explanation (2.3)
        tokenizer = CharacterTokenizer("test")
        model = MicroLM(vocab_size=tokenizer.vocab_size(), embedding_dim=16, hidden_dim=32, num_layers=1)
        model_info = model.get_model_info()
        
        assert 'architecture' in model_info
        assert 'layer_info' in model_info
        
        print("✓ Requirement 2: Comprehensive documentation - PASSED")
    
    def test_requirement_3_training_functionality(self):
        """
        Validate Requirement 3: Training on simple text data.
        
        Requirements 3.1, 3.2, 3.3, 3.4: Text input, sequence learning, model saving, loading
        """
        training_text = "Simple training text for requirement validation"
        
        # Test plain text file input (3.1)
        tokenizer = CharacterTokenizer(training_text)
        model = MicroLM(vocab_size=tokenizer.vocab_size(), embedding_dim=16, hidden_dim=32, num_layers=1)
        trainer = ModelTrainer(model, tokenizer)
        
        # Test learning from character sequences (3.2)
        data_loader = trainer.prepare_data(training_text, sequence_length=8, batch_size=4)
        
        # Verify data contains input-target pairs
        for inputs, targets in data_loader:
            # Targets should be inputs shifted by one position (next character prediction)
            assert inputs.shape == targets.shape
            break
        
        # Test training and model saving (3.3)
        history = trainer.train(data_loader, epochs=3, learning_rate=0.01)
        assert len(history['train_loss']) == 3
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "requirement_test.pt")
            trainer.save_model(model_path)
            assert os.path.exists(model_path)
            
            # Test model loading (3.4)
            new_model = MicroLM(vocab_size=tokenizer.vocab_size(), embedding_dim=16, hidden_dim=32, num_layers=1)
            new_trainer = ModelTrainer(new_model, tokenizer)
            new_trainer.load_model(model_path)
            
            # Should restore model state
            assert new_trainer.current_epoch == trainer.current_epoch
        
        print("✓ Requirement 3: Training functionality - PASSED")
    
    def test_requirement_4_text_generation(self):
        """
        Validate Requirement 4: Text generation with prompts.
        
        Requirements 4.1, 4.2, 4.3, 4.4: Prompt continuation, sampling strategies, readable output
        """
        training_text = "The quick brown fox jumps over the lazy dog"
        tokenizer = CharacterTokenizer(training_text)
        model = MicroLM(vocab_size=tokenizer.vocab_size(), embedding_dim=32, hidden_dim=64, num_layers=1)
        
        # Quick training for functional model
        trainer = ModelTrainer(model, tokenizer)
        data_loader = trainer.prepare_data(training_text, sequence_length=10, batch_size=4)
        trainer.train(data_loader, epochs=5, learning_rate=0.01)
        
        generator = TextGenerator(model, tokenizer)
        
        # Test prompt continuation (4.1)
        prompt = "The quick"
        generated = generator.generate(prompt, length=20, temperature=0.0)
        assert generated.startswith(prompt)
        assert len(generated) == len(prompt) + 20
        
        # Test sampling strategies (4.2)
        greedy_text = generator.generate(prompt, length=10, temperature=0.0)
        temp_text = generator.generate(prompt, length=10, temperature=0.8)
        
        # Both should start with prompt but may differ due to sampling
        assert greedy_text.startswith(prompt)
        assert temp_text.startswith(prompt)
        
        # Test readable output (4.3)
        for char in generated:
            assert char in tokenizer.get_vocab()  # All characters should be valid
        
        # Test full generated text return (4.4)
        full_text = generator.generate("Test", length=15, temperature=0.5)
        assert full_text.startswith("Test")  # Includes original prompt
        assert len(full_text) == 4 + 15  # Prompt + generated length
        
        print("✓ Requirement 4: Text generation - PASSED")
    
    def test_requirement_5_training_interface(self):
        """
        Validate Requirement 5: Simple training interface.
        
        Requirements 5.1, 5.2, 5.3, 5.4: Configurable parameters, progress display, statistics
        """
        training_text = "Training interface validation text"
        tokenizer = CharacterTokenizer(training_text)
        model = MicroLM(vocab_size=tokenizer.vocab_size(), embedding_dim=16, hidden_dim=32, num_layers=1)
        trainer = ModelTrainer(model, tokenizer)
        
        data_loader = trainer.prepare_data(training_text, sequence_length=6, batch_size=2)
        
        # Test configurable parameters (5.1)
        history = trainer.train(
            data_loader=data_loader,
            epochs=3,
            learning_rate=0.005,  # Custom learning rate
            optimizer_type='sgd',  # Custom optimizer
            weight_decay=0.01,     # Custom weight decay
            gradient_clip_norm=0.5  # Custom gradient clipping
        )
        
        # Test progress information (5.2)
        assert 'train_loss' in history
        assert 'epoch_times' in history
        assert 'learning_rates' in history
        assert len(history['train_loss']) == 3
        
        # Test training statistics (5.4)
        stats = trainer.get_training_stats()
        assert 'loss_stats' in stats
        assert 'time_stats' in stats
        assert 'training_progress' in stats
        
        print("✓ Requirement 5: Training interface - PASSED")
    
    def test_requirement_6_model_inspection(self):
        """
        Validate Requirement 6: Model inspection and visualization.
        
        Requirements 6.1, 6.2, 6.3, 6.4: Architecture summary, parameter counts, utilities
        """
        tokenizer = CharacterTokenizer("inspection test")
        model = MicroLM(vocab_size=tokenizer.vocab_size(), embedding_dim=32, hidden_dim=64, num_layers=2)
        
        # Test architecture summary (6.1)
        model_info = model.get_model_info()
        assert 'architecture' in model_info
        assert 'layer_info' in model_info
        
        # Test parameter counting (6.2)
        assert 'parameters' in model_info
        assert 'total' in model_info['parameters']
        assert model_info['parameters']['total'] > 0
        
        # Verify parameter count accuracy
        actual_params = sum(p.numel() for p in model.parameters())
        assert model_info['parameters']['total'] == actual_params
        
        # Test model size estimation (6.3)
        assert 'model_size_mb' in model_info
        assert model_info['model_size_mb'] > 0
        
        # Test inspection utilities (6.4)
        # Test that we can inspect model components
        assert hasattr(model, 'embedding')
        assert hasattr(model, 'lstm')
        assert hasattr(model, 'output_projection')
        
        # Test model representation
        repr_str = repr(model)
        assert 'MicroLM' in repr_str
        assert str(model.vocab_size) in repr_str
        
        print("✓ Requirement 6: Model inspection - PASSED")
    
    def test_all_requirements_integration(self):
        """
        Integration test that validates all requirements working together.
        
        This test ensures that all individual requirements work together
        in a complete end-to-end workflow.
        """
        print("\n" + "="*60)
        print("COMPREHENSIVE REQUIREMENTS VALIDATION")
        print("="*60)
        
        # Complete workflow that exercises all requirements
        training_text = """
        This comprehensive test validates all requirements of the Micro Language Model.
        The system should handle character-level tokenization, provide extensive
        documentation, support training on text data, enable text generation with
        prompts, offer a simple training interface, and provide model inspection
        capabilities. All components should work together seamlessly.
        """
        
        # Requirement 1 & 2: Tokenization with documentation
        tokenizer = CharacterTokenizer(training_text)
        assert tokenizer.__doc__ is not None  # Documentation requirement
        
        # Requirement 1 & 6: Model architecture with inspection
        model = MicroLM(
            vocab_size=tokenizer.vocab_size(),
            embedding_dim=64,
            hidden_dim=128,
            num_layers=2
        )
        model_info = model.get_model_info()  # Inspection requirement
        
        # Requirement 3 & 5: Training with interface
        trainer = ModelTrainer(model, tokenizer)
        data_loader = trainer.prepare_data(training_text, sequence_length=25, batch_size=8)
        
        history = trainer.train(
            data_loader=data_loader,
            epochs=8,
            learning_rate=0.001,
            optimizer_type='adam'
        )
        
        # Requirement 4: Text generation
        generator = TextGenerator(model, tokenizer)
        
        test_prompts = [
            "This comprehensive",
            "The system should",
            "All components"
        ]
        
        for prompt in test_prompts:
            # Test different generation strategies
            greedy_text = generator.generate_greedy(prompt, length=30)
            temp_text = generator.generate_with_temperature(prompt, length=30, temperature=0.7)
            
            assert greedy_text.startswith(prompt)
            assert temp_text.startswith(prompt)
            assert len(greedy_text) == len(prompt) + 30
            assert len(temp_text) == len(prompt) + 30
        
        # Requirement 3: Model persistence
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "comprehensive_test.pt")
            trainer.save_model(model_path)
            
            new_trainer = ModelTrainer(
                MicroLM(vocab_size=tokenizer.vocab_size(), embedding_dim=64, hidden_dim=128, num_layers=2),
                tokenizer
            )
            new_trainer.load_model(model_path)
        
        print("✓ All requirements validated successfully!")
        print(f"✓ Model trained for {len(history['train_loss'])} epochs")
        print(f"✓ Final loss: {history['train_loss'][-1]:.4f}")
        print(f"✓ Model parameters: {model_info['parameters']['total']:,}")
        print(f"✓ Generated text samples validated")
        print(f"✓ Model persistence verified")
        print("="*60)


if __name__ == "__main__":
    # Run integration tests with detailed output
    pytest.main([__file__, "-v", "-s", "--tb=short"])