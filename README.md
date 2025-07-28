# Micro Language Model 🧠

A simple, educational character-level language model implementation designed for learning and understanding the fundamentals of neural language modeling.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Purpose

This project provides a clean, well-documented implementation of a character-level LSTM language model. It's designed specifically for:

- **Learning**: Understanding how neural language models work at a fundamental level
- **Education**: Teaching the core concepts with extensive comments and explanations
- **Experimentation**: Providing a simple platform for trying different approaches
- **Research**: Serving as a baseline for more complex implementations

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd micro-lm

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Configuration Guide

Before diving into training, check out our comprehensive setup guide to choose the right model configuration for your system and use case:

```bash
# Run the interactive setup guide
python setup/setup_guide.py
```

This will analyze your system capabilities and recommend optimal model configurations based on your available memory, processing power, and intended use case.

### Basic Usage

```python
from micro_lm import CharacterTokenizer, MicroLM, ModelTrainer, TextGenerator

# 1. Prepare your data
text_data = "Hello world! This is a simple example of text data for training."

# 2. Create tokenizer
tokenizer = CharacterTokenizer(text_data)
print(f"Vocabulary size: {tokenizer.vocab_size()}")

# 3. Create model
model = MicroLM(
    vocab_size=tokenizer.vocab_size(),
    embedding_dim=128,
    hidden_dim=256,
    num_layers=2
)

# 4. Train the model
trainer = ModelTrainer(model, tokenizer)
data_loader = trainer.prepare_data(text_data, sequence_length=50)
training_stats = trainer.train(data_loader, epochs=50, learning_rate=0.001)

# 5. Generate text
generator = TextGenerator(model, tokenizer)
generated_text = generator.generate("Hello", length=100, temperature=0.8)
print(f"Generated: {generated_text}")
```

## � CProject Structure

```
micro-lm/
├── micro_lm/              # Core package
│   ├── __init__.py        # Package exports
│   ├── tokenizer.py       # Character tokenization
│   ├── model.py           # LSTM language model
│   ├── trainer.py         # Training utilities
│   ├── generator.py       # Text generation
│   └── inspection.py      # Model analysis tools
├── setup/                 # Configuration and setup
│   └── setup_guide.py     # Interactive configuration guide
├── examples/              # Usage demonstrations
│   ├── training_demo.py   # Complete training example
│   ├── text_generation_demo.py  # Generation examples
│   └── model_inspection_demo.py # Model analysis demo
├── tests/                 # Test suite
├── setup.py              # Package installation
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## 🎮 Examples and Demos

Explore the `examples/` directory for comprehensive demonstrations:

```bash
# Complete training workflow
python examples/training_demo.py

# Text generation techniques
python examples/text_generation_demo.py

# Model analysis and inspection
python examples/model_inspection_demo.py
```

Each example includes detailed comments explaining the concepts and best practices.

## 📚 Core Components

### 1. CharacterTokenizer

Converts text to numerical indices and back, operating at the character level for simplicity.

```python
from micro_lm import CharacterTokenizer

# Create tokenizer from training text
tokenizer = CharacterTokenizer("Hello world!")

# Encode text to numbers
encoded = tokenizer.encode("Hello")
print(encoded)  # [7, 4, 11, 11, 14]

# Decode numbers back to text
decoded = tokenizer.decode(encoded)
print(decoded)  # "Hello"

# Inspect vocabulary
print(f"Vocab size: {tokenizer.vocab_size()}")
print(f"Vocabulary: {tokenizer.get_vocab()}")
```

**Key Features:**
- Character-level tokenization for simplicity
- Deterministic vocabulary building
- Round-trip encoding/decoding
- Unknown character handling

### 2. MicroLM (Language Model)

LSTM-based neural network that learns to predict the next character in a sequence.

```python
from micro_lm import MicroLM
import torch

# Create model
model = MicroLM(
    vocab_size=50,        # Size of character vocabulary
    embedding_dim=128,    # Character embedding dimension
    hidden_dim=256,       # LSTM hidden state dimension
    num_layers=2,         # Number of LSTM layers
    dropout=0.2           # Dropout for regularization
)

# Forward pass
input_sequence = torch.randint(0, 50, (32, 100))  # batch_size=32, seq_len=100
output = model(input_sequence)  # Shape: (32, 100, 50)

# Get model information
info = model.get_model_info()
print(f"Parameters: {info['total_parameters']:,}")
print(f"Model size: {info['model_size_mb']:.2f} MB")
```

**Architecture:**
- **Embedding Layer**: Maps character indices to dense vectors
- **LSTM Layers**: Capture sequential patterns and dependencies
- **Output Projection**: Projects to vocabulary probabilities
- **Dropout**: Prevents overfitting during training

### 3. ModelTrainer

Handles the training process with progress tracking and model persistence.

```python
from micro_lm import ModelTrainer

# Create trainer
trainer = ModelTrainer(model, tokenizer)

# Prepare training data
text = "Your training text here..."
data_loader = trainer.prepare_data(
    text, 
    sequence_length=100,  # Length of input sequences
    batch_size=32        # Training batch size
)

# Train the model
training_stats = trainer.train(
    data_loader=data_loader,
    epochs=100,           # Number of training epochs
    learning_rate=0.001,  # Learning rate for optimizer
    save_every=10,        # Save model every N epochs
    model_path="models/my_model.pt"
)

# Access training statistics
print(f"Final loss: {training_stats['losses'][-1]:.4f}")
print(f"Training time: {training_stats['training_time']:.2f} seconds")
```

**Features:**
- Automatic data preparation and batching
- Progress tracking with loss monitoring
- Model checkpointing and persistence
- Training statistics collection
- Early stopping support

### 4. TextGenerator

Generates text using the trained model with various sampling strategies.

```python
from micro_lm import TextGenerator

# Create generator
generator = TextGenerator(model, tokenizer)

# Greedy generation (deterministic)
text = generator.generate("The quick", length=50, temperature=0.0)

# Creative generation (higher randomness)
text = generator.generate("The quick", length=50, temperature=1.2)

# Balanced generation
text = generator.generate("The quick", length=50, temperature=0.8)

# Batch generation for comparison
prompts = ["Hello", "The cat", "In the beginning"]
results = generator.generate_batch(prompts, length=30, temperature=0.8)
```

**Sampling Strategies:**
- **Temperature = 0.0**: Greedy decoding (most likely character)
- **Temperature < 1.0**: Conservative sampling (more predictable)
- **Temperature = 1.0**: Neutral sampling (balanced)
- **Temperature > 1.0**: Creative sampling (more random)

## ⚙️ Configuration Options

### Choosing the Right Configuration

For detailed guidance on selecting model configurations, run our interactive setup guide:

```bash
python setup/setup_guide.py
```

The guide will analyze your system and recommend configurations from nano (ultra-fast) to xlarge (research-grade).

### Model Architecture Examples

```python
# Nano model (ultra-minimal, ~2K parameters)
model = MicroLM(
    vocab_size=tokenizer.vocab_size(),
    embedding_dim=16,
    hidden_dim=32,
    num_layers=1,
    dropout=0.1
)

# Small model (balanced, ~35K parameters)
model = MicroLM(
    vocab_size=tokenizer.vocab_size(),
    embedding_dim=64,
    hidden_dim=128,
    num_layers=2,
    dropout=0.2
)

# Large model (high quality, ~600K parameters)
model = MicroLM(
    vocab_size=tokenizer.vocab_size(),
    embedding_dim=256,
    hidden_dim=512,
    num_layers=3,
    dropout=0.3
)
```

**Configuration Selection Criteria:**
- **System Memory**: Larger models need more RAM
- **Training Time**: Bigger models train slower but produce better results
- **Use Case**: Research vs. prototyping vs. production
- **Data Size**: Larger models need more training data

### Training Parameters

```python
# Quick training (for testing)
trainer.train(
    data_loader=data_loader,
    epochs=20,
    learning_rate=0.003,
    batch_size=16
)

# Standard training (balanced)
trainer.train(
    data_loader=data_loader,
    epochs=100,
    learning_rate=0.001,
    batch_size=32
)

# Thorough training (best quality)
trainer.train(
    data_loader=data_loader,
    epochs=200,
    learning_rate=0.0005,
    batch_size=64
)
```

### Generation Parameters

| Parameter | Effect | Recommended Range |
|-----------|--------|-------------------|
| `temperature` | Controls randomness | 0.0 (deterministic) to 2.0 (very random) |
| `length` | Output length | 10-1000 characters |
| `prompt` | Starting text | Any string in training vocabulary |

## 🔧 Advanced Usage

### Custom Training Loop

```python
import torch
import torch.nn as nn
from torch.optim import Adam

# Manual training setup
model = MicroLM(vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2)
optimizer = Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
model.train()
for epoch in range(epochs):
    total_loss = 0
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(data_loader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
```

### Model Inspection

```python
from micro_lm import inspect_model, analyze_parameters

# Get detailed model information
info = inspect_model(model)
print(f"Architecture: {info['architecture']}")
print(f"Parameters: {info['parameter_count']:,}")
print(f"Memory usage: {info['memory_mb']:.2f} MB")

# Analyze parameter distributions
analysis = analyze_parameters(model)
print(f"Parameter health: {analysis['health_score']:.2f}")
print(f"Gradient flow: {analysis['gradient_flow']}")
```

### Batch Processing

```python
# Process multiple texts
texts = ["First text", "Second text", "Third text"]
results = []

for text in texts:
    tokenizer = CharacterTokenizer(text)
    # ... train model for each text
    results.append(trained_model)

# Batch generation
prompts = ["Hello", "The cat", "Once upon"]
generated = generator.generate_batch(prompts, length=50)
```

## 🎓 Language Modeling Concepts

### What is a Language Model?

A language model is a neural network that learns the probability distribution of sequences in a language. Given a sequence of characters (or words), it predicts what character is most likely to come next.

**Mathematical Foundation:**
```
P(c_t | c_1, c_2, ..., c_{t-1})
```
Where `c_t` is the character at position `t`, and the model predicts it based on all previous characters.

### Character-Level vs Word-Level

**Character-Level (used in this project):**
- ✅ Simpler vocabulary (typically 50-100 characters)
- ✅ Can generate any word, including made-up ones
- ✅ Better for learning fundamentals
- ❌ Slower generation
- ❌ May struggle with long-range dependencies

**Word-Level:**
- ✅ Faster generation
- ✅ Better long-range understanding
- ❌ Large vocabulary (10,000+ words)
- ❌ Cannot generate unknown words

### LSTM Architecture

**Why LSTM?**
Long Short-Term Memory networks solve the vanishing gradient problem of traditional RNNs, allowing them to learn long-range dependencies in sequences.

**Key Components:**
1. **Forget Gate**: Decides what information to discard
2. **Input Gate**: Decides what new information to store
3. **Output Gate**: Controls what parts of cell state to output
4. **Cell State**: Long-term memory of the network

### Training Process

1. **Forward Pass**: Input sequence → Model → Predictions
2. **Loss Calculation**: Compare predictions with actual next characters
3. **Backward Pass**: Calculate gradients using backpropagation
4. **Parameter Update**: Adjust weights using optimizer (Adam)
5. **Repeat**: Continue for multiple epochs until convergence

### Generation Strategies

**Greedy Decoding:**
```python
# Always pick the most likely character
next_char = torch.argmax(probabilities)
```

**Temperature Sampling:**
```python
# Adjust probability distribution
probabilities = probabilities / temperature
next_char = torch.multinomial(probabilities, 1)
```

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### 1. Memory Errors

**Problem**: `RuntimeError: CUDA out of memory` or system memory issues

**Solutions:**
```python
# Reduce batch size
data_loader = trainer.prepare_data(text, sequence_length=50, batch_size=8)

# Use smaller model
model = MicroLM(vocab_size, embedding_dim=64, hidden_dim=128, num_layers=1)

# Reduce sequence length
data_loader = trainer.prepare_data(text, sequence_length=25)
```

#### 2. Slow Training

**Problem**: Training takes too long

**Solutions:**
```python
# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Increase learning rate for faster convergence
trainer.train(data_loader, epochs=50, learning_rate=0.003)

# Use smaller dataset for testing
sample_text = text[:10000]  # Use first 10k characters
```

#### 3. Poor Generation Quality

**Problem**: Generated text is nonsensical or repetitive

**Solutions:**
```python
# Train for more epochs
trainer.train(data_loader, epochs=200)

# Use larger model
model = MicroLM(vocab_size, embedding_dim=256, hidden_dim=512, num_layers=3)

# Adjust generation temperature
text = generator.generate("prompt", length=100, temperature=0.8)

# Ensure sufficient training data (at least 10k characters)
```

#### 4. Import Errors

**Problem**: `ModuleNotFoundError: No module named 'micro_lm'`

**Solutions:**
```bash
# Install in development mode from project root
pip install -e .

# Or add to Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/micro-lm"

# Verify installation
python -c "import micro_lm; print('Success!')"
```

#### 5. Training Loss Not Decreasing

**Problem**: Loss stays high or increases during training

**Solutions:**
```python
# Lower learning rate
trainer.train(data_loader, epochs=100, learning_rate=0.0005)

# Check data quality
print(f"Unique characters: {tokenizer.vocab_size()}")
print(f"Text length: {len(text)}")

# Increase model capacity
model = MicroLM(vocab_size, embedding_dim=256, hidden_dim=512, num_layers=2)

# Add gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

#### 6. Generation Produces Repeated Characters

**Problem**: Output like "aaaaaaa" or "the the the"

**Solutions:**
```python
# Increase temperature
text = generator.generate("prompt", temperature=1.0)  # Instead of 0.5

# Train longer
trainer.train(data_loader, epochs=150)

# Use more diverse training data
# Ensure training text has varied patterns
```

### Debugging Tips

1. **Check vocabulary size**: Should be reasonable (20-200 characters)
2. **Monitor training loss**: Should decrease over time
3. **Validate data**: Ensure text is clean and representative
4. **Test small**: Start with small models and datasets
5. **Use examples**: Run provided examples first to verify setup

### Performance Optimization

```python
# Enable mixed precision training (if supported)
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = model(inputs)
    loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))

# Use DataLoader with multiple workers
data_loader = DataLoader(dataset, batch_size=32, num_workers=4)

# Compile model for faster inference (PyTorch 2.0+)
model = torch.compile(model)
```

## 📖 Learning Resources

### Recommended Reading

**Books:**
- "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville
- "Natural Language Processing with Python" by Steven Bird and Ewan Klein
- "Speech and Language Processing" by Dan Jurafsky and James H. Martin

**Papers:**
- "Understanding LSTM Networks" by Christopher Olah
- "The Unreasonable Effectiveness of Recurrent Neural Networks" by Andrej Karpathy
- "Attention Is All You Need" by Vaswani et al. (for understanding modern transformers)

### Online Courses

- **CS231n**: Convolutional Neural Networks for Visual Recognition (Stanford)
- **CS224n**: Natural Language Processing with Deep Learning (Stanford)
- **Fast.ai**: Practical Deep Learning for Coders
- **Coursera**: Deep Learning Specialization by Andrew Ng

### Tutorials and Blogs

- **Andrej Karpathy's Blog**: Excellent explanations of RNNs and language models
- **The Illustrated Transformer**: Visual guide to attention mechanisms
- **PyTorch Tutorials**: Official tutorials for deep learning with PyTorch
- **Towards Data Science**: Medium publication with ML articles

### Interactive Resources

- **Playground TensorFlow**: Visualize neural networks in your browser
- **ConvNetJS**: Interactive demos of neural networks
- **Distill.pub**: Interactive machine learning explanations

### Advanced Topics to Explore

1. **Transformer Architecture**: Modern approach to sequence modeling
2. **Attention Mechanisms**: How models focus on relevant parts of input
3. **BERT and GPT**: Pre-trained language models
4. **Fine-tuning**: Adapting pre-trained models to specific tasks
5. **Reinforcement Learning from Human Feedback (RLHF)**: Training models with human preferences

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Report Issues**: Found a bug? Open an issue with details
2. **Suggest Features**: Have an idea? Discuss it in issues first
3. **Improve Documentation**: Help make explanations clearer
4. **Add Examples**: Create new educational examples
5. **Optimize Code**: Improve performance while maintaining clarity

### Development Setup

```bash
# Clone and install in development mode
git clone <repository-url>
cd micro-lm
pip install -e .

# Install development dependencies
pip install pytest pytest-cov black flake8

# Run tests
pytest tests/

# Format code
black micro_lm/ examples/ setup/ tests/

# Check code style
flake8 micro_lm/ examples/ setup/ tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by Andrej Karpathy's educational materials on neural networks
- Built with PyTorch for its excellent educational value and flexibility
- Thanks to the open-source community for tools and inspiration

---

**Happy Learning!** 🚀

For questions, issues, or contributions, please visit our [GitHub repository](https://github.com/your-repo/micro-lm).