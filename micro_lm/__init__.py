"""
MicroLSTM - A simple educational LSTM language model implementation.

This package provides a character-level LSTM language model designed for
learning purposes, with extensive documentation and clear code structure.

Components:
- CharacterTokenizer: Handles text-to-numbers conversion
- MicroLM: The LSTM-based language model
- ModelTrainer: Training utilities and progress tracking
- TextGenerator: Text generation with multiple sampling strategies
- Custom exceptions: Comprehensive error handling with helpful context
"""

__version__ = "0.1.0"
__author__ = "Micro LM Project"

# Import main classes for easy access
from .tokenizer import CharacterTokenizer
from .model import MicroLM
from .trainer import ModelTrainer  
from .generator import TextGenerator
from .inspection import ModelInspector, TrainingVisualizer, inspect_model, visualize_training, analyze_parameters
from .exceptions import (
    ModelError, TrainingError, GenerationError, TokenizationError,
    DataError, ModelConfigurationError, FileOperationError, CudaError
)

__all__ = [
    "CharacterTokenizer",
    "MicroLM", 
    "ModelTrainer",
    "TextGenerator",
    "ModelInspector",
    "TrainingVisualizer",
    "inspect_model",
    "visualize_training", 
    "analyze_parameters",
    "ModelError",
    "TrainingError", 
    "GenerationError",
    "TokenizationError",
    "DataError",
    "ModelConfigurationError",
    "FileOperationError",
    "CudaError"
]