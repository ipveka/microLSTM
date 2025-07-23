"""
Comprehensive tests for error handling and validation in the Micro Language Model.

This test module verifies that all custom exceptions are raised correctly
under various error conditions, and that error messages provide helpful
context for debugging and fixing issues.
"""

import pytest
import torch
import tempfile
import os
import json
from unittest.mock import patch, MagicMock

from micro_lm.exceptions import (
    ModelError, TrainingError, GenerationError, TokenizationError,
    DataError, ModelConfigurationError, FileOperationError, CudaError,
    create_model_error
)
from micro_lm.tokenizer import CharacterTokenizer
from micro_lm.model import MicroLM
from micro_lm.trainer import ModelTrainer
from micro_lm.generator import TextGenerator


class TestCustomExceptions:
    """Test custom exception classes and their functionality."""
    
    def test_model_error_base_class(self):
        """Test ModelError base class functionality."""
        error = ModelError("Test error", error_code="TEST_001", context={"key": "value"})
        
        assert str(error) == "[TEST_001] Test error"
        assert error.message == "Test error"
        assert error.error_code == "TEST_001"
        assert error.context == {"key": "value"}
        assert "ModelError" in repr(error)
    
    def test_training_error_with_context(self):
        """Test TrainingError with training-specific context."""
        error = TrainingError("Training failed", epoch=5, batch=10, learning_rate=0.001)
        
        assert error.context["epoch"] == 5
        assert error.context["batch"] == 10
        assert error.context["learning_rate"] == 0.001
        assert error.error_code == "TRAINING_ERROR"
    
    def test_generation_error_with_context(self):
        """Test GenerationError with generation-specific context."""
        error = GenerationError("Generation failed", prompt="hello", length=50)
        
        assert error.context["prompt"] == "hello"
        assert error.context["length"] == 50
        assert error.error_code == "GENERATION_ERROR"
    
    def test_tokenization_error_with_context(self):
        """Test TokenizationError with tokenization-specific context."""
        error = TokenizationError("Tokenization failed", text="hello", tokens=[1, 2, 3])
        
        assert error.context["text"] == "hello"
        assert error.context["tokens"] == [1, 2, 3]
        assert error.error_code == "TOKENIZATION_ERROR"
    
    def test_create_model_error_factory(self):
        """Test the error factory function."""
        training_error = create_model_error("training", "Training failed", epoch=5)
        assert isinstance(training_error, TrainingError)
        assert training_error.context["epoch"] == 5
        
        generation_error = create_model_error("generation", "Generation failed", prompt="test")
        assert isinstance(generation_error, GenerationError)
        assert generation_error.context["prompt"] == "test"
        
        generic_error = create_model_error("unknown", "Unknown error")
        assert isinstance(generic_error, ModelError)


class TestTokenizerErrorHandling:
    """Test error handling in the CharacterTokenizer class."""
    
    def test_empty_corpus_error(self):
        """Test error when initializing with empty corpus."""
        with pytest.raises(TokenizationError) as exc_info:
            CharacterTokenizer("")
        
        assert "Text corpus cannot be empty" in str(exc_info.value)
        assert exc_info.value.context["operation"] == "vocabulary_building"
    
    def test_none_corpus_error(self):
        """Test error when initializing with None corpus."""
        with pytest.raises(TokenizationError) as exc_info:
            CharacterTokenizer(None)
        
        assert "Text corpus cannot be empty" in str(exc_info.value)
    
    def test_encode_non_string_input(self):
        """Test error when encoding non-string input."""
        tokenizer = CharacterTokenizer("hello")
        
        with pytest.raises(TokenizationError) as exc_info:
            tokenizer.encode(123)
        
        assert "Input must be a string" in str(exc_info.value)
        assert exc_info.value.context["operation"] == "encoding"
    
    def test_encode_unknown_character(self):
        """Test error when encoding unknown character."""
        tokenizer = CharacterTokenizer("hello")
        
        with pytest.raises(TokenizationError) as exc_info:
            tokenizer.encode("hello world")  # 'w', 'r', 'd', ' ' not in vocab
        
        assert "not found in vocabulary" in str(exc_info.value)
        assert exc_info.value.context["operation"] == "encoding"
    
    def test_decode_non_list_input(self):
        """Test error when decoding non-list input."""
        tokenizer = CharacterTokenizer("hello")
        
        with pytest.raises(TokenizationError) as exc_info:
            tokenizer.decode("not a list")
        
        assert "Indices must be a list" in str(exc_info.value)
        assert exc_info.value.context["operation"] == "decoding"
    
    def test_decode_non_integer_indices(self):
        """Test error when decoding non-integer indices."""
        tokenizer = CharacterTokenizer("hello")
        
        with pytest.raises(TokenizationError) as exc_info:
            tokenizer.decode([0, 1.5, 2])
        
        assert "All indices must be integers" in str(exc_info.value)
        assert exc_info.value.context["operation"] == "decoding"
    
    def test_decode_invalid_indices(self):
        """Test error when decoding invalid indices."""
        tokenizer = CharacterTokenizer("hello")
        
        with pytest.raises(TokenizationError) as exc_info:
            tokenizer.decode([0, 1, 999])  # 999 is out of range
        
        assert "not found in vocabulary" in str(exc_info.value)
        assert exc_info.value.context["operation"] == "decoding"
    
    def test_save_vocab_invalid_filepath(self):
        """Test error when saving with invalid filepath."""
        tokenizer = CharacterTokenizer("hello")
        
        with pytest.raises(FileOperationError) as exc_info:
            tokenizer.save_vocab("")
        
        assert "Filepath must be a non-empty string" in str(exc_info.value)
        assert exc_info.value.context["operation"] == "save_vocabulary"
    
    def test_save_vocab_permission_error(self):
        """Test error when saving to protected location."""
        tokenizer = CharacterTokenizer("hello")
        
        with pytest.raises(FileOperationError) as exc_info:
            tokenizer.save_vocab("/root/protected.json")  # Assuming no write access
        
        assert "Failed to save vocabulary" in str(exc_info.value)
        assert exc_info.value.context["operation"] == "save_vocabulary"
    
    def test_load_vocab_invalid_filepath(self):
        """Test error when loading with invalid filepath."""
        with pytest.raises(FileOperationError) as exc_info:
            CharacterTokenizer.load_vocab("")
        
        assert "Filepath must be a non-empty string" in str(exc_info.value)
        assert exc_info.value.context["operation"] == "load_vocabulary"
    
    def test_load_vocab_nonexistent_file(self):
        """Test error when loading nonexistent file."""
        with pytest.raises(FileOperationError) as exc_info:
            CharacterTokenizer.load_vocab("/nonexistent/path.json")
        
        assert "Vocabulary file not found" in str(exc_info.value)
        assert exc_info.value.context["operation"] == "load_vocabulary"
    
    def test_load_vocab_invalid_json(self):
        """Test error when loading invalid JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            temp_path = f.name
        
        try:
            with pytest.raises(FileOperationError) as exc_info:
                CharacterTokenizer.load_vocab(temp_path)
            
            assert "Invalid JSON format" in str(exc_info.value)
            assert exc_info.value.context["operation"] == "load_vocabulary"
        finally:
            os.unlink(temp_path)
    
    def test_load_vocab_invalid_format(self):
        """Test error when loading vocabulary with invalid format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(["not", "a", "dict"], f)  # Should be dict, not list
            temp_path = f.name
        
        try:
            with pytest.raises(FileOperationError) as exc_info:
                CharacterTokenizer.load_vocab(temp_path)
            
            assert "Invalid vocabulary format" in str(exc_info.value)
            assert exc_info.value.context["operation"] == "load_vocabulary"
        finally:
            os.unlink(temp_path)


class TestModelErrorHandling:
    """Test error handling in the MicroLM model class."""
    
    def test_invalid_vocab_size(self):
        """Test error with invalid vocab_size parameter."""
        with pytest.raises(ModelConfigurationError) as exc_info:
            MicroLM(vocab_size=-10, embedding_dim=128, hidden_dim=256, num_layers=2)
        
        assert "vocab_size must be a positive integer" in str(exc_info.value)
        assert exc_info.value.context["parameter"] == "vocab_size"
        assert exc_info.value.context["value"] == -10
    
    def test_invalid_embedding_dim(self):
        """Test error with invalid embedding_dim parameter."""
        with pytest.raises(ModelConfigurationError) as exc_info:
            MicroLM(vocab_size=50, embedding_dim=0, hidden_dim=256, num_layers=2)
        
        assert "embedding_dim must be a positive integer" in str(exc_info.value)
        assert exc_info.value.context["parameter"] == "embedding_dim"
    
    def test_invalid_hidden_dim(self):
        """Test error with invalid hidden_dim parameter."""
        with pytest.raises(ModelConfigurationError) as exc_info:
            MicroLM(vocab_size=50, embedding_dim=128, hidden_dim=-5, num_layers=2)
        
        assert "hidden_dim must be a positive integer" in str(exc_info.value)
        assert exc_info.value.context["parameter"] == "hidden_dim"
    
    def test_invalid_num_layers(self):
        """Test error with invalid num_layers parameter."""
        with pytest.raises(ModelConfigurationError) as exc_info:
            MicroLM(vocab_size=50, embedding_dim=128, hidden_dim=256, num_layers=0)
        
        assert "num_layers must be a positive integer" in str(exc_info.value)
        assert exc_info.value.context["parameter"] == "num_layers"
    
    def test_invalid_dropout(self):
        """Test error with invalid dropout parameter."""
        with pytest.raises(ModelConfigurationError) as exc_info:
            MicroLM(vocab_size=50, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=1.5)
        
        assert "dropout must be a number between 0 and 1" in str(exc_info.value)
        assert exc_info.value.context["parameter"] == "dropout"
    
    def test_forward_invalid_input_type(self):
        """Test error with invalid input type in forward pass."""
        model = MicroLM(vocab_size=50, embedding_dim=32, hidden_dim=64, num_layers=1)
        
        with pytest.raises(ModelConfigurationError) as exc_info:
            model.forward("not a tensor")
        
        assert "Input must be a torch.Tensor" in str(exc_info.value)
        assert exc_info.value.context["parameter"] == "input"
    
    def test_forward_invalid_input_shape(self):
        """Test error with invalid input shape in forward pass."""
        model = MicroLM(vocab_size=50, embedding_dim=32, hidden_dim=64, num_layers=1)
        
        with pytest.raises(ModelConfigurationError) as exc_info:
            model.forward(torch.randn(10))  # 1D tensor instead of 2D
        
        assert "Input tensor must be 2-dimensional" in str(exc_info.value)
        assert exc_info.value.context["parameter"] == "input_shape"
    
    def test_forward_invalid_input_dtype(self):
        """Test error with invalid input dtype in forward pass."""
        model = MicroLM(vocab_size=50, embedding_dim=32, hidden_dim=64, num_layers=1)
        
        with pytest.raises(ModelConfigurationError) as exc_info:
            model.forward(torch.randn(2, 10))  # Float tensor instead of integer
        
        assert "Input tensor must have integer dtype" in str(exc_info.value)
        assert exc_info.value.context["parameter"] == "input_dtype"
    
    def test_forward_invalid_token_indices(self):
        """Test error with invalid token indices in forward pass."""
        model = MicroLM(vocab_size=50, embedding_dim=32, hidden_dim=64, num_layers=1)
        
        with pytest.raises(ModelConfigurationError) as exc_info:
            model.forward(torch.tensor([[0, 1, 100]]))  # 100 is out of vocab range
        
        assert "Input contains invalid token indices" in str(exc_info.value)
        assert exc_info.value.context["parameter"] == "token_indices"
    
    def test_init_hidden_invalid_batch_size(self):
        """Test error with invalid batch_size in init_hidden."""
        model = MicroLM(vocab_size=50, embedding_dim=32, hidden_dim=64, num_layers=1)
        
        with pytest.raises(ModelConfigurationError) as exc_info:
            model.init_hidden(batch_size=-1)
        
        assert "batch_size must be a positive integer" in str(exc_info.value)
        assert exc_info.value.context["parameter"] == "batch_size"
    
    def test_init_hidden_invalid_device(self):
        """Test error with invalid device in init_hidden."""
        model = MicroLM(vocab_size=50, embedding_dim=32, hidden_dim=64, num_layers=1)
        
        with pytest.raises(CudaError) as exc_info:
            model.init_hidden(batch_size=1, device="invalid_device")
        
        assert "Invalid device specification" in str(exc_info.value)


class TestTrainerErrorHandling:
    """Test error handling in the ModelTrainer class."""
    
    def test_invalid_model_type(self):
        """Test error with invalid model type."""
        tokenizer = CharacterTokenizer("hello world")
        
        with pytest.raises(ModelConfigurationError) as exc_info:
            ModelTrainer("not a model", tokenizer)
        
        assert "model must be MicroLM instance" in str(exc_info.value)
        assert exc_info.value.context["parameter"] == "model"
    
    def test_invalid_tokenizer_type(self):
        """Test error with invalid tokenizer type."""
        model = MicroLM(vocab_size=50, embedding_dim=32, hidden_dim=64, num_layers=1)
        
        with pytest.raises(ModelConfigurationError) as exc_info:
            ModelTrainer(model, "not a tokenizer")
        
        assert "tokenizer must be CharacterTokenizer instance" in str(exc_info.value)
        assert exc_info.value.context["parameter"] == "tokenizer"
    
    def test_vocab_size_mismatch(self):
        """Test error with mismatched vocab sizes."""
        model = MicroLM(vocab_size=50, embedding_dim=32, hidden_dim=64, num_layers=1)
        tokenizer = CharacterTokenizer("hello")  # Different vocab size
        
        with pytest.raises(ModelConfigurationError) as exc_info:
            ModelTrainer(model, tokenizer)
        
        assert "Model vocab_size" in str(exc_info.value)
        assert "must match tokenizer vocab_size" in str(exc_info.value)
        assert exc_info.value.context["parameter"] == "vocab_size_mismatch"
    
    def test_prepare_data_invalid_text(self):
        """Test error with invalid text in prepare_data."""
        tokenizer = CharacterTokenizer("hello world")
        model = MicroLM(vocab_size=tokenizer.vocab_size(), embedding_dim=32, hidden_dim=64, num_layers=1)
        trainer = ModelTrainer(model, tokenizer)
        
        with pytest.raises(TrainingError) as exc_info:
            trainer.prepare_data("", sequence_length=5)
        
        assert "Text must be a non-empty string" in str(exc_info.value)
        assert "text_length" in exc_info.value.context
    
    def test_prepare_data_invalid_sequence_length(self):
        """Test error with invalid sequence_length in prepare_data."""
        tokenizer = CharacterTokenizer("hello world")
        model = MicroLM(vocab_size=tokenizer.vocab_size(), embedding_dim=32, hidden_dim=64, num_layers=1)
        trainer = ModelTrainer(model, tokenizer)
        
        with pytest.raises(TrainingError) as exc_info:
            trainer.prepare_data("hello world", sequence_length=-5)
        
        assert "sequence_length must be a positive integer" in str(exc_info.value)
        assert exc_info.value.context["sequence_length"] == -5
    
    def test_train_invalid_epochs(self):
        """Test error with invalid epochs in train method."""
        tokenizer = CharacterTokenizer("hello world")
        model = MicroLM(vocab_size=tokenizer.vocab_size(), embedding_dim=32, hidden_dim=64, num_layers=1)
        trainer = ModelTrainer(model, tokenizer)
        data_loader = trainer.prepare_data("hello world", sequence_length=3, batch_size=1)
        
        with pytest.raises(TrainingError) as exc_info:
            trainer.train(data_loader, epochs=0)
        
        assert "epochs must be a positive integer" in str(exc_info.value)
        assert exc_info.value.context["epochs"] == 0
    
    def test_train_invalid_learning_rate(self):
        """Test error with invalid learning_rate in train method."""
        tokenizer = CharacterTokenizer("hello world")
        model = MicroLM(vocab_size=tokenizer.vocab_size(), embedding_dim=32, hidden_dim=64, num_layers=1)
        trainer = ModelTrainer(model, tokenizer)
        data_loader = trainer.prepare_data("hello world", sequence_length=3, batch_size=1)
        
        with pytest.raises(TrainingError) as exc_info:
            trainer.train(data_loader, epochs=1, learning_rate=-0.001)
        
        assert "learning_rate must be a positive number" in str(exc_info.value)
        assert exc_info.value.context["learning_rate"] == -0.001
    
    def test_save_model_invalid_filepath(self):
        """Test error with invalid filepath in save_model."""
        tokenizer = CharacterTokenizer("hello world")
        model = MicroLM(vocab_size=tokenizer.vocab_size(), embedding_dim=32, hidden_dim=64, num_layers=1)
        trainer = ModelTrainer(model, tokenizer)
        
        with pytest.raises(FileOperationError) as exc_info:
            trainer.save_model("")
        
        assert "Filepath must be a non-empty string" in str(exc_info.value)
        assert exc_info.value.context["operation"] == "save_model"
    
    def test_load_model_invalid_filepath(self):
        """Test error with invalid filepath in load_model."""
        tokenizer = CharacterTokenizer("hello world")
        model = MicroLM(vocab_size=tokenizer.vocab_size(), embedding_dim=32, hidden_dim=64, num_layers=1)
        trainer = ModelTrainer(model, tokenizer)
        
        with pytest.raises(FileOperationError) as exc_info:
            trainer.load_model("")
        
        assert "Filepath must be a non-empty string" in str(exc_info.value)
        assert exc_info.value.context["operation"] == "load_model"
    
    def test_load_model_nonexistent_file(self):
        """Test error when loading nonexistent model file."""
        tokenizer = CharacterTokenizer("hello world")
        model = MicroLM(vocab_size=tokenizer.vocab_size(), embedding_dim=32, hidden_dim=64, num_layers=1)
        trainer = ModelTrainer(model, tokenizer)
        
        with pytest.raises(FileOperationError) as exc_info:
            trainer.load_model("/nonexistent/model.pt")
        
        assert "Model file not found" in str(exc_info.value)
        assert exc_info.value.context["operation"] == "load_model"


class TestGeneratorErrorHandling:
    """Test error handling in the TextGenerator class."""
    
    def test_invalid_model_type(self):
        """Test error with invalid model type."""
        tokenizer = CharacterTokenizer("hello world")
        
        with pytest.raises(ModelConfigurationError) as exc_info:
            TextGenerator("not a model", tokenizer)
        
        assert "model must be MicroLM instance" in str(exc_info.value)
        assert exc_info.value.context["parameter"] == "model"
    
    def test_invalid_tokenizer_type(self):
        """Test error with invalid tokenizer type."""
        model = MicroLM(vocab_size=50, embedding_dim=32, hidden_dim=64, num_layers=1)
        
        with pytest.raises(ModelConfigurationError) as exc_info:
            TextGenerator(model, "not a tokenizer")
        
        assert "tokenizer must be CharacterTokenizer instance" in str(exc_info.value)
        assert exc_info.value.context["parameter"] == "tokenizer"
    
    def test_vocab_size_mismatch(self):
        """Test error with mismatched vocab sizes."""
        model = MicroLM(vocab_size=50, embedding_dim=32, hidden_dim=64, num_layers=1)
        tokenizer = CharacterTokenizer("hello")  # Different vocab size
        
        with pytest.raises(ModelConfigurationError) as exc_info:
            TextGenerator(model, tokenizer)
        
        assert "Model vocab_size" in str(exc_info.value)
        assert "must match tokenizer vocab_size" in str(exc_info.value)
        assert exc_info.value.context["parameter"] == "vocab_size_mismatch"
    
    def test_generate_invalid_prompt_type(self):
        """Test error with invalid prompt type."""
        tokenizer = CharacterTokenizer("hello world")
        model = MicroLM(vocab_size=tokenizer.vocab_size(), embedding_dim=32, hidden_dim=64, num_layers=1)
        generator = TextGenerator(model, tokenizer)
        
        with pytest.raises(GenerationError) as exc_info:
            generator.generate(123, length=10)
        
        assert "Prompt must be a string" in str(exc_info.value)
        assert exc_info.value.context["prompt"] == 123
    
    def test_generate_empty_prompt(self):
        """Test error with empty prompt."""
        tokenizer = CharacterTokenizer("hello world")
        model = MicroLM(vocab_size=tokenizer.vocab_size(), embedding_dim=32, hidden_dim=64, num_layers=1)
        generator = TextGenerator(model, tokenizer)
        
        with pytest.raises(GenerationError) as exc_info:
            generator.generate("", length=10)
        
        assert "Prompt cannot be empty" in str(exc_info.value)
        assert exc_info.value.context["prompt"] == ""
    
    def test_generate_invalid_length(self):
        """Test error with invalid length."""
        tokenizer = CharacterTokenizer("hello world")
        model = MicroLM(vocab_size=tokenizer.vocab_size(), embedding_dim=32, hidden_dim=64, num_layers=1)
        generator = TextGenerator(model, tokenizer)
        
        with pytest.raises(GenerationError) as exc_info:
            generator.generate("hello", length=-10)
        
        assert "Length must be a positive integer" in str(exc_info.value)
        assert exc_info.value.context["length"] == -10
    
    def test_generate_invalid_temperature(self):
        """Test error with invalid temperature."""
        tokenizer = CharacterTokenizer("hello world")
        model = MicroLM(vocab_size=tokenizer.vocab_size(), embedding_dim=32, hidden_dim=64, num_layers=1)
        generator = TextGenerator(model, tokenizer)
        
        with pytest.raises(GenerationError) as exc_info:
            generator.generate("hello", length=10, temperature=-0.5)
        
        assert "Temperature must be a non-negative number" in str(exc_info.value)
        assert exc_info.value.context["temperature"] == -0.5
    
    def test_generate_invalid_top_k(self):
        """Test error with invalid top_k."""
        tokenizer = CharacterTokenizer("hello world")
        model = MicroLM(vocab_size=tokenizer.vocab_size(), embedding_dim=32, hidden_dim=64, num_layers=1)
        generator = TextGenerator(model, tokenizer)
        
        with pytest.raises(GenerationError) as exc_info:
            generator.generate("hello", length=10, top_k=0)
        
        assert "top_k must be a positive integer" in str(exc_info.value)
        assert exc_info.value.context["top_k"] == 0
    
    def test_generate_invalid_top_p(self):
        """Test error with invalid top_p."""
        tokenizer = CharacterTokenizer("hello world")
        model = MicroLM(vocab_size=tokenizer.vocab_size(), embedding_dim=32, hidden_dim=64, num_layers=1)
        generator = TextGenerator(model, tokenizer)
        
        with pytest.raises(GenerationError) as exc_info:
            generator.generate("hello", length=10, top_p=1.5)
        
        assert "top_p must be a number between 0.0 and 1.0" in str(exc_info.value)
        assert exc_info.value.context["top_p"] == 1.5


class TestCudaErrorHandling:
    """Test CUDA-specific error handling."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_out_of_memory_simulation(self):
        """Test handling of CUDA out of memory errors."""
        # This test simulates CUDA OOM by creating a very large model
        # that would exceed GPU memory on most systems
        try:
            # Create an extremely large model that should cause OOM
            model = MicroLM(
                vocab_size=10000,
                embedding_dim=8192,
                hidden_dim=8192,
                num_layers=10
            )
            
            # Try to move to CUDA - this might cause OOM
            model.to('cuda')
            
            # Try to create very large batch
            large_input = torch.randint(0, 10000, (1000, 1000), device='cuda')
            model(large_input)
            
        except CudaError as e:
            # This is expected for OOM scenarios
            assert "CUDA" in str(e) or "out of memory" in str(e).lower()
            assert "suggestion" in e.context
        except RuntimeError as e:
            # PyTorch might raise RuntimeError for CUDA issues
            # Our code should catch and convert these
            if "CUDA" in str(e) or "out of memory" in str(e).lower():
                pytest.fail("CUDA RuntimeError should be caught and converted to CudaError")


if __name__ == "__main__":
    pytest.main([__file__])