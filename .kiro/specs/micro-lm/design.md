# Design Document

## Overview

The Micro Language Model is designed as an educational tool to demonstrate the fundamental concepts of neural language modeling. The system uses a character-level LSTM (Long Short-Term Memory) network, which provides a good balance between simplicity and effectiveness for learning purposes. The architecture prioritizes code readability and extensive documentation over performance optimization.

## Architecture

### High-Level Architecture

```
Text Input → Tokenizer → Model → Training Loop → Saved Model
                ↓           ↓
            Vocabulary   Loss Function
                ↓           ↓
            Character    Backpropagation
            Mapping         ↓
                        Updated Weights
```

### Core Components

1. **Tokenizer**: Converts text to character indices and vice versa
2. **Vocabulary**: Maps characters to unique integer IDs
3. **LSTM Model**: Neural network for sequence prediction
4. **Trainer**: Handles the training loop and optimization
5. **Generator**: Produces text from trained model
6. **Utils**: Helper functions for data processing and visualization

## Components and Interfaces

### 1. Tokenizer Class

```python
class CharacterTokenizer:
    """
    Handles conversion between text and numerical representations.
    Uses character-level tokenization for simplicity.
    """
    
    def __init__(self, text_corpus: str)
    def encode(self, text: str) -> List[int]
    def decode(self, indices: List[int]) -> str
    def vocab_size(self) -> int
    def get_vocab(self) -> Dict[str, int]
```

**Purpose**: Provides a clean interface for text-to-numbers conversion, essential for neural network processing.

### 2. MicroLM Model Class

```python
class MicroLM(nn.Module):
    """
    Simple LSTM-based language model.
    Predicts next character given a sequence of previous characters.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers: int)
    def forward(self, x: torch.Tensor) -> torch.Tensor
    def init_hidden(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]
    def get_model_info(self) -> Dict[str, Any]
```

**Purpose**: Core neural network that learns character patterns and relationships.

### 3. Trainer Class

```python
class ModelTrainer:
    """
    Handles the training process with clear progress reporting.
    Includes utilities for monitoring and early stopping.
    """
    
    def __init__(self, model: MicroLM, tokenizer: CharacterTokenizer)
    def prepare_data(self, text: str, sequence_length: int) -> DataLoader
    def train(self, data_loader: DataLoader, epochs: int, learning_rate: float) -> Dict[str, List[float]]
    def save_model(self, filepath: str) -> None
    def load_model(self, filepath: str) -> None
```

**Purpose**: Encapsulates training logic with clear progress tracking and model persistence.

### 4. TextGenerator Class

```python
class TextGenerator:
    """
    Generates text using the trained model.
    Supports different sampling strategies for variety.
    """
    
    def __init__(self, model: MicroLM, tokenizer: CharacterTokenizer)
    def generate(self, prompt: str, length: int, temperature: float = 1.0) -> str
    def sample_next_char(self, probabilities: torch.Tensor, temperature: float) -> int
```

**Purpose**: Provides text generation capabilities with controllable randomness.

## Data Models

### Training Data Structure

```python
@dataclass
class TrainingBatch:
    """Represents a batch of training sequences"""
    input_sequences: torch.Tensor  # Shape: (batch_size, sequence_length)
    target_sequences: torch.Tensor # Shape: (batch_size, sequence_length)
    
@dataclass
class ModelConfig:
    """Configuration for model architecture"""
    vocab_size: int
    embedding_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 2
    sequence_length: int = 100
    
@dataclass
class TrainingConfig:
    """Configuration for training process"""
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    save_every: int = 10
```

### Model State

The model maintains several key pieces of state:
- **Embedding Layer**: Maps character indices to dense vectors
- **LSTM Layers**: Process sequences and maintain hidden states
- **Output Layer**: Projects LSTM output to vocabulary probabilities
- **Hidden States**: Temporary memory for sequence processing

## Error Handling

### Input Validation
- **Text Input**: Validate that input text contains recognizable characters
- **Model Parameters**: Ensure dimensions are positive integers
- **File Operations**: Handle missing files and permission errors gracefully

### Training Errors
- **Memory Issues**: Catch CUDA out-of-memory errors and suggest smaller batch sizes
- **Convergence Problems**: Monitor for exploding/vanishing gradients
- **Data Issues**: Handle empty or corrupted training data

### Generation Errors
- **Invalid Prompts**: Handle prompts with unknown characters
- **Model State**: Ensure model is properly loaded before generation
- **Parameter Bounds**: Validate temperature and length parameters

```python
class ModelError(Exception):
    """Base exception for model-related errors"""
    pass

class TrainingError(ModelError):
    """Raised when training encounters issues"""
    pass

class GenerationError(ModelError):
    """Raised when text generation fails"""
    pass
```

## Testing Strategy

### Unit Tests
1. **Tokenizer Tests**
   - Test character encoding/decoding round-trips
   - Verify vocabulary creation from text corpus
   - Test handling of unknown characters

2. **Model Architecture Tests**
   - Verify model output shapes for different input sizes
   - Test forward pass with dummy data
   - Validate parameter initialization

3. **Training Tests**
   - Test data preparation and batching
   - Verify loss calculation and backpropagation
   - Test model saving and loading

4. **Generation Tests**
   - Test text generation with various prompts
   - Verify temperature effects on randomness
   - Test generation length limits

### Integration Tests
1. **End-to-End Training**
   - Train on small sample text and verify loss decreases
   - Test complete training pipeline from text to saved model

2. **Generation Pipeline**
   - Load trained model and generate coherent text
   - Test generation with different sampling strategies

### Example Test Structure
```python
def test_tokenizer_round_trip():
    """Test that encode->decode returns original text"""
    tokenizer = CharacterTokenizer("hello world")
    original = "hello"
    encoded = tokenizer.encode(original)
    decoded = tokenizer.decode(encoded)
    assert decoded == original

def test_model_forward_pass():
    """Test model produces correct output shape"""
    model = MicroLM(vocab_size=50, embedding_dim=32, hidden_dim=64, num_layers=1)
    input_tensor = torch.randint(0, 50, (2, 10))  # batch_size=2, seq_len=10
    output = model(input_tensor)
    assert output.shape == (2, 10, 50)  # (batch, seq, vocab)
```

This design emphasizes educational value through:
- Clear separation of concerns with dedicated classes
- Extensive documentation and comments
- Simple, understandable architecture choices
- Comprehensive error handling and testing
- Focus on demonstrating core concepts rather than optimization