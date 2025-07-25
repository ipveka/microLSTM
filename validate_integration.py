#!/usr/bin/env python3
"""
Comprehensive integration validation script for Micro Language Model.

This script validates that all components work together seamlessly and
verifies that all requirements from the specification are met.
"""

import sys
import os
import tempfile
import traceback
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from micro_lm import (
    CharacterTokenizer, MicroLM, ModelTrainer, TextGenerator,
    ModelInspector, TrainingVisualizer, inspect_model, visualize_training,
    analyze_parameters
)
from micro_lm.exceptions import ModelError


def test_requirement_1_character_level_model():
    """Test Requirement 1: Character-level language model implementation."""
    print("Testing Requirement 1: Character-level language model...")
    
    try:
        # Test character-level tokenization
        text = "Hello, World! This is a test."
        tokenizer = CharacterTokenizer(text)
        
        # Verify character-level operation
        encoded = tokenizer.encode("Hello")
        decoded = tokenizer.decode(encoded)
        assert decoded == "Hello", f"Round-trip failed: {decoded} != Hello"
        
        # Test model with character vocabulary
        model = MicroLM(
            vocab_size=tokenizer.vocab_size(),
            embedding_dim=32,
            hidden_dim=64,
            num_layers=1
        )
        
        # Test next character prediction
        import torch
        input_seq = torch.randint(0, tokenizer.vocab_size(), (1, 10))
        output = model(input_seq)
        assert output.shape == (1, 10, tokenizer.vocab_size()), f"Wrong output shape: {output.shape}"
        
        print("✓ Requirement 1 passed: Character-level model working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Requirement 1 failed: {e}")
        traceback.print_exc()
        return False


def test_requirement_2_comprehensive_documentation():
    """Test Requirement 2: Comprehensive code comments and explanations."""
    print("Testing Requirement 2: Comprehensive documentation...")
    
    try:
        # Check that all main classes have docstrings
        classes_to_check = [CharacterTokenizer, MicroLM, ModelTrainer, TextGenerator]
        
        for cls in classes_to_check:
            assert cls.__doc__ is not None, f"{cls.__name__} missing docstring"
            assert len(cls.__doc__.strip()) > 50, f"{cls.__name__} docstring too short"
            
            # Check that __init__ method has docstring
            assert cls.__init__.__doc__ is not None, f"{cls.__name__}.__init__ missing docstring"
        
        # Check that key methods have docstrings
        tokenizer = CharacterTokenizer("test")
        assert tokenizer.encode.__doc__ is not None, "encode method missing docstring"
        assert tokenizer.decode.__doc__ is not None, "decode method missing docstring"
        
        print("✓ Requirement 2 passed: Comprehensive documentation present")
        return True
        
    except Exception as e:
        print(f"✗ Requirement 2 failed: {e}")
        traceback.print_exc()
        return False


def test_requirement_3_training_functionality():
    """Test Requirement 3: Training on simple text data."""
    print("Testing Requirement 3: Training functionality...")
    
    try:
        # Prepare training data
        text = "The quick brown fox jumps over the lazy dog. " * 10
        tokenizer = CharacterTokenizer(text)
        
        # Create model
        model = MicroLM(
            vocab_size=tokenizer.vocab_size(),
            embedding_dim=32,
            hidden_dim=64,
            num_layers=1
        )
        
        # Create trainer
        trainer = ModelTrainer(model, tokenizer)
        
        # Prepare data
        data_loader = trainer.prepare_data(text, sequence_length=20, batch_size=4)
        
        # Train for a few epochs
        history = trainer.train(data_loader, epochs=3, learning_rate=0.01)
        
        # Verify training worked
        assert len(history['train_loss']) == 3, f"Wrong number of epochs: {len(history['train_loss'])}"
        assert history['train_loss'][-1] < history['train_loss'][0], "Loss did not decrease"
        
        # Test model saving and loading
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "test_model.pt")
            trainer.save_model(model_path)
            assert os.path.exists(model_path), "Model file not saved"
            
            # Create new trainer and load model
            new_model = MicroLM(
                vocab_size=tokenizer.vocab_size(),
                embedding_dim=32,
                hidden_dim=64,
                num_layers=1
            )
            new_trainer = ModelTrainer(new_model, tokenizer)
            new_trainer.load_model(model_path)
        
        print("✓ Requirement 3 passed: Training functionality working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Requirement 3 failed: {e}")
        traceback.print_exc()
        return False


def test_requirement_4_text_generation():
    """Test Requirement 4: Text generation with prompts."""
    print("Testing Requirement 4: Text generation...")
    
    try:
        # Create and train a simple model
        text = "hello world " * 50
        tokenizer = CharacterTokenizer(text)
        
        model = MicroLM(
            vocab_size=tokenizer.vocab_size(),
            embedding_dim=16,
            hidden_dim=32,
            num_layers=1
        )
        
        trainer = ModelTrainer(model, tokenizer)
        data_loader = trainer.prepare_data(text, sequence_length=10, batch_size=2)
        trainer.train(data_loader, epochs=5, learning_rate=0.01)
        
        # Test text generation
        generator = TextGenerator(model, tokenizer)
        
        # Test greedy generation
        generated = generator.generate("hello", length=10, temperature=0.0)
        assert generated.startswith("hello"), f"Generated text doesn't start with prompt: {generated}"
        assert len(generated) >= 15, f"Generated text too short: {len(generated)}"
        
        # Test temperature sampling
        generated_temp = generator.generate("hello", length=10, temperature=0.8)
        assert generated_temp.startswith("hello"), f"Temperature generated text doesn't start with prompt: {generated_temp}"
        
        # Test different generation methods
        greedy_text = generator.generate_greedy("hello", length=10)
        temp_text = generator.generate_with_temperature("hello", length=10, temperature=0.5)
        
        assert greedy_text.startswith("hello"), "Greedy generation failed"
        assert temp_text.startswith("hello"), "Temperature generation failed"
        
        print("✓ Requirement 4 passed: Text generation working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Requirement 4 failed: {e}")
        traceback.print_exc()
        return False


def test_requirement_5_training_interface():
    """Test Requirement 5: Simple training interface."""
    print("Testing Requirement 5: Training interface...")
    
    try:
        text = "simple training test " * 20
        tokenizer = CharacterTokenizer(text)
        
        model = MicroLM(
            vocab_size=tokenizer.vocab_size(),
            embedding_dim=16,
            hidden_dim=32,
            num_layers=1
        )
        
        trainer = ModelTrainer(model, tokenizer)
        
        # Test configurable parameters
        data_loader = trainer.prepare_data(text, sequence_length=8, batch_size=2)
        
        # Test training with different parameters
        history = trainer.train(
            data_loader=data_loader,
            epochs=3,
            learning_rate=0.005,
            optimizer_type='adam'
        )
        
        # Verify progress reporting
        assert 'train_loss' in history, "Training history missing loss"
        assert 'epoch_times' in history, "Training history missing timing"
        assert len(history['train_loss']) == 3, "Wrong number of epochs recorded"
        
        # Test early stopping capability (simulated)
        try:
            # This should work without errors
            trainer.train(data_loader, epochs=1, learning_rate=0.01)
        except KeyboardInterrupt:
            pass  # This is expected behavior
        
        print("✓ Requirement 5 passed: Training interface working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Requirement 5 failed: {e}")
        traceback.print_exc()
        return False


def test_requirement_6_model_inspection():
    """Test Requirement 6: Model inspection and visualization."""
    print("Testing Requirement 6: Model inspection...")
    
    try:
        text = "inspection test data " * 15
        tokenizer = CharacterTokenizer(text)
        
        model = MicroLM(
            vocab_size=tokenizer.vocab_size(),
            embedding_dim=24,
            hidden_dim=48,
            num_layers=2
        )
        
        # Test model info
        info = model.get_model_info()
        assert 'architecture' in info, "Model info missing architecture"
        assert 'parameters' in info, "Model info missing parameters"
        assert 'total' in info['parameters'], "Model info missing total parameters"
        
        # Test model inspector
        inspector = ModelInspector(model)
        summary = inspector.get_architecture_summary()
        
        assert 'basic_info' in summary, "Inspector summary missing basic info"
        assert 'layer_details' in summary, "Inspector summary missing layer details"
        assert 'parameter_analysis' in summary, "Inspector summary missing parameter analysis"
        
        # Test parameter statistics
        param_stats = inspector.get_parameter_statistics()
        assert 'layer_stats' in param_stats, "Parameter stats missing layer stats"
        assert 'overall_stats' in param_stats, "Parameter stats missing overall stats"
        
        # Test inspect_model utility function
        model_info = inspect_model(model)
        assert isinstance(model_info, dict), "inspect_model should return dict"
        
        # Test analyze_parameters utility function
        param_analysis = analyze_parameters(model)
        assert isinstance(param_analysis, dict), "analyze_parameters should return dict"
        
        print("✓ Requirement 6 passed: Model inspection working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Requirement 6 failed: {e}")
        traceback.print_exc()
        return False


def test_error_handling():
    """Test comprehensive error handling."""
    print("Testing error handling...")
    
    try:
        # Test tokenizer errors
        try:
            CharacterTokenizer("")  # Empty corpus
            assert False, "Should have raised error for empty corpus"
        except ModelError:
            pass  # Expected
        
        # Test model configuration errors
        try:
            MicroLM(vocab_size=0, embedding_dim=32, hidden_dim=64, num_layers=1)
            assert False, "Should have raised error for invalid vocab_size"
        except ModelError:
            pass  # Expected
        
        # Test trainer errors
        tokenizer = CharacterTokenizer("test")
        model = MicroLM(vocab_size=tokenizer.vocab_size(), embedding_dim=16, hidden_dim=32, num_layers=1)
        trainer = ModelTrainer(model, tokenizer)
        
        try:
            trainer.prepare_data("", sequence_length=5)  # Empty text
            assert False, "Should have raised error for empty text"
        except ModelError:
            pass  # Expected
        
        # Test generator errors
        generator = TextGenerator(model, tokenizer)
        
        try:
            generator.generate("", length=10)  # Empty prompt
            assert False, "Should have raised error for empty prompt"
        except ModelError:
            pass  # Expected
        
        print("✓ Error handling working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Error handling failed: {e}")
        traceback.print_exc()
        return False


def test_integration_workflow():
    """Test complete end-to-end workflow."""
    print("Testing complete integration workflow...")
    
    try:
        # Step 1: Prepare data
        training_text = """
        Once upon a time, in a land far away, there lived a wise old wizard.
        The wizard had a magical book that could predict the future.
        Every day, people would come from miles around to seek his wisdom.
        The wizard would open his book and read the ancient words within.
        """ * 5
        
        # Step 2: Create tokenizer
        tokenizer = CharacterTokenizer(training_text)
        print(f"Vocabulary size: {tokenizer.vocab_size()}")
        
        # Step 3: Create model
        model = MicroLM(
            vocab_size=tokenizer.vocab_size(),
            embedding_dim=64,
            hidden_dim=128,
            num_layers=2,
            dropout=0.1
        )
        
        # Step 4: Create trainer
        trainer = ModelTrainer(model, tokenizer)
        
        # Step 5: Prepare training data
        data_loader = trainer.prepare_data(
            training_text, 
            sequence_length=50, 
            batch_size=8
        )
        
        # Step 6: Train model
        print("Training model...")
        history = trainer.train(
            data_loader=data_loader,
            epochs=10,
            learning_rate=0.001,
            optimizer_type='adam'
        )
        
        # Step 7: Generate text
        generator = TextGenerator(model, tokenizer)
        
        print("\nGenerating text samples:")
        prompts = ["Once upon", "The wizard", "Every day"]
        
        for prompt in prompts:
            generated = generator.generate(prompt, length=100, temperature=0.7)
            print(f"Prompt: '{prompt}' -> Generated: '{generated[:50]}...'")
        
        # Step 8: Inspect model
        inspector = ModelInspector(model)
        inspector.print_model_summary(detailed=False)
        
        # Step 9: Save and load model
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "final_model.pt")
            trainer.save_model(model_path)
            
            # Load in new trainer
            new_model = MicroLM(
                vocab_size=tokenizer.vocab_size(),
                embedding_dim=64,
                hidden_dim=128,
                num_layers=2,
                dropout=0.1
            )
            new_trainer = ModelTrainer(new_model, tokenizer)
            new_trainer.load_model(model_path)
            
            # Test loaded model
            new_generator = TextGenerator(new_model, tokenizer)
            loaded_text = new_generator.generate("Once upon", length=50, temperature=0.7)
            print(f"Loaded model generation: '{loaded_text[:30]}...'")
        
        print("✓ Complete integration workflow successful")
        return True
        
    except Exception as e:
        print(f"✗ Integration workflow failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all integration tests."""
    print("=" * 80)
    print("MICRO LANGUAGE MODEL - INTEGRATION VALIDATION")
    print("=" * 80)
    
    tests = [
        test_requirement_1_character_level_model,
        test_requirement_2_comprehensive_documentation,
        test_requirement_3_training_functionality,
        test_requirement_4_text_generation,
        test_requirement_5_training_interface,
        test_requirement_6_model_inspection,
        test_error_handling,
        test_integration_workflow
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        print("\n" + "-" * 60)
        if test():
            passed += 1
        print("-" * 60)
    
    print("\n" + "=" * 80)
    print(f"INTEGRATION VALIDATION RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL INTEGRATION TESTS PASSED! 🎉")
        print("The Micro Language Model is ready for release.")
    else:
        print(f"❌ {total - passed} tests failed. Please review and fix issues.")
    
    print("=" * 80)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)