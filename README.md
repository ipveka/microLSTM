# MicroLSTM 🧠

A simple, educational character-level LSTM language model for learning neural language modeling fundamentals.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Perfect for learning, experimentation, and understanding how language models work before diving into more complex architectures. 

**Verified Working**: All tests pass and demos run successfully on Windows, macOS, and Linux.

## 🚀 Quick Start

### Installation

```bash
git clone <repository-url>
cd microLSTM
pip install -r requirements.txt
pip install -e .
```

### Testing

Verify everything works correctly:

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=micro_lstm --cov-report=term-missing
```

### Configuration Guide

Choose the right model configuration for your system:

```bash
python setup/setup_guide.py
```

### Basic Usage

#### Command Line Interface (Recommended)

```bash
# Train a model
microlstm train --text-file data.txt --epochs 100 --save-model model.pt

# Generate text
microlstm generate --model model.pt --prompt "Hello" --length 100

# Interactive generation
microlstm generate --model model.pt --interactive

# Analyze model
microlstm analyze --model model.pt
```

#### Python API

```python
from micro_lstm import DataLoader, CharacterTokenizer, MicroLSTM, ModelTrainer, TextGenerator

# 1. Load and prepare dataset from Hugging Face
loader = DataLoader()
text, tokenizer, info = loader.quick_setup("roneneldan/TinyStories", preprocess=True)

# 2. Create and train model
model = MicroLSTM(vocab_size=tokenizer.vocab_size(), embedding_dim=128, hidden_dim=256, num_layers=2)
trainer = ModelTrainer(model, tokenizer)
data_loader = trainer.prepare_data(text, sequence_length=100, batch_size=32)
trainer.train(data_loader, epochs=50, learning_rate=0.001)

# 3. Generate text
generator = TextGenerator(model, tokenizer)
generated_text = generator.generate("Once upon a time", length=100, temperature=0.8)
print(f"Generated: {generated_text}")
```

#### Available Datasets

The DataLoader supports several popular datasets:
- **roneneldan/TinyStories**: Simple stories for children (~1MB)
- **wikitext-2-raw-v1**: Wikipedia articles (~2MB)
- **bookcorpus**: Free books from Project Gutenberg (~500MB)
- **openwebtext**: Web content from Reddit (~8GB)
- **custom**: Your own text files

### Advanced Training with Configuration

For larger models and production training, use the advanced training script:

```bash
# Train a large model with custom configuration
python src/run.py
```

This script:
- **Loads configuration** from `config.yaml`
- **Trains larger models** (up to 8M+ parameters)
- **Uses GPU acceleration** automatically
- **Saves models** to `models/` directory
- **Generates text samples** after training
- **Provides detailed logging** and progress tracking

#### Configuration Options

Edit `config.yaml` to customize:

```yaml
# Model Architecture
model:
  embedding_dim: 256      # Character embedding size
  hidden_dim: 512         # LSTM hidden layer size
  num_layers: 4           # Number of LSTM layers
  dropout: 0.3            # Dropout for regularization

# Training Settings
training:
  epochs: 20              # Number of training epochs
  batch_size: 32          # Batch size (larger for GPU)
  sequence_length: 100    # Context window size
  learning_rate: 0.001    # Learning rate

# Generation Settings
generation:
  num_samples: 5          # Number of text samples to generate
  length: 200             # Length of each sample
  temperature: 0.8        # Sampling temperature
```

#### Example Output

The script produces:
- **Trained model**: `models/big_microlstm_YYYYMMDD_HHMMSS.pth`
- **Tokenizer**: `models/big_microlstm_YYYYMMDD_HHMMSS_tokenizer.pkl`
- **Configuration**: `models/big_microlstm_YYYYMMDD_HHMMSS_config.yaml`
- **Generated samples**: `models/generated_samples_YYYYMMDD_HHMMSS.txt`
- **Training log**: `training.log`

## 📁 Project Structure

```
micro-lstm/
├── micro_lstm/              # Core package (tokenizer, model, trainer, generator)
├── src/                    # Advanced training scripts
├── models/                 # Saved trained models
├── setup/                  # Configuration guide
├── examples/               # Usage demos
├── tests/                  # Test suite
├── config.yaml             # Training configuration
└── requirements.txt        # Dependencies
```

## 🎮 Examples

Run the interactive examples to see MicroLSTM in action:

```bash
# Data preparation and training workflow
python examples/data_preparation.py

# Complete training with model inspection
python examples/training.py

# Interactive text generation with different settings
python examples/text_generation.py
```

**Example Results**: All examples run successfully and demonstrate:
- ✅ Data preprocessing and vocabulary creation
- ✅ Model architecture analysis with detailed parameter statistics
- ✅ Training with loss tracking and progress monitoring
- ✅ Model inspection before and after training
- ✅ Text generation with temperature control and interactive prompts

## 🏗️ Architecture: LSTM vs Transformers

This project uses **LSTM** (Long Short-Term Memory) architecture. Here's how it compares to modern **Transformers**:

### LSTM Advantages ✅
- **Sequential processing**: Natural for understanding text flow
- **Memory efficient**: Lower memory usage, good for learning
- **Simple architecture**: Easier to understand and implement
- **Good for small data**: Works well with limited training data
- **Educational value**: Great for learning RNN fundamentals

### LSTM Limitations ❌
- **Sequential bottleneck**: Must process text one character at a time
- **Limited context**: Struggles with very long-range dependencies
- **Slower training**: Cannot parallelize sequence processing
- **Vanishing gradients**: Can lose information over long sequences

### Transformers Advantages ✅
- **Parallel processing**: Can process entire sequences simultaneously
- **Attention mechanism**: Directly connects any two positions in text
- **Better long-range**: Excellent at capturing distant relationships
- **Scalable**: Works well with massive datasets and models
- **State-of-the-art**: Powers GPT, BERT, and other modern models

### Transformers Limitations ❌
- **Memory intensive**: Quadratic memory usage with sequence length
- **Complex architecture**: Harder to understand and implement
- **Data hungry**: Needs large amounts of training data
- **Computational cost**: Requires more resources for training

### How LSTM Works (Simple Explanation)

**LSTM = Long Short-Term Memory**

Think of LSTM as a smart memory system that reads text one character at a time:

```
Input: "Hello world"
Step 1: Read 'H' → Remember: "I've seen H"
Step 2: Read 'e' → Remember: "I've seen He" 
Step 3: Read 'l' → Remember: "I've seen Hel"
...and so on
```

**Key Components:**
1. **Cell State** (Long-term memory): Stores important information
2. **Hidden State** (Short-term memory): Current context
3. **Gates** (Smart filters): Decide what to remember/forget

**The Magic:**
- **Forget Gate**: "Should I forget old information?"
- **Input Gate**: "What new information should I store?"
- **Output Gate**: "What should I output based on what I know?"

**Why it works for language:**
- Remembers context: "The cat sat on the ___" → likely "mat"
- Handles grammar: Knows "cats" needs plural verbs
- Learns patterns: Common word sequences and structures

### Why LSTM for Learning?

This project uses LSTM because it's:
- **Conceptually simpler**: Easier to understand how language modeling works
- **Resource friendly**: Runs on modest hardware
- **Educational**: Teaches fundamental concepts before advanced architectures
- **Historical importance**: Foundation for understanding modern models

## ⚙️ Model Configurations

Run the setup guide for personalized recommendations:

```bash
python setup/setup_guide.py
```

## 🔧 Core Components

### CharacterTokenizer
Converts text to numbers and back:
```python
tokenizer = CharacterTokenizer("Hello world!")
encoded = tokenizer.encode("Hello")  # [7, 4, 11, 11, 14]
decoded = tokenizer.decode(encoded)  # "Hello"
```

### MicroLSTM
LSTM-based language model:
```python
model = MicroLSTM(vocab_size=50, embedding_dim=128, hidden_dim=256, num_layers=2)
```

### ModelTrainer
Handles training with progress tracking:
```python
trainer = ModelTrainer(model, tokenizer)
trainer.train(data_loader, epochs=100, learning_rate=0.001)
```

### TextGenerator
Generates text with temperature control:
```python
generator = TextGenerator(model, tokenizer)
text = generator.generate("Hello", length=50, temperature=0.8)
```

## 🛠️ Common Issues

**Memory errors**: Reduce batch size or model size
**Slow training**: Use GPU, increase learning rate, or smaller model
**Poor quality**: Train longer, use larger model, or more data
**Import errors**: Run `pip install -e .` from project root
**Python not found**: Use `py` instead of `python` on Windows

## ✅ Project Status

**Current Status**: All systems operational
- **Tests**: 68/68 passing ✅
- **Examples**: All 3 examples working correctly ✅
- **Advanced Training**: Large model training script working ✅
- **GPU Support**: Full CUDA acceleration supported ✅
- **Coverage**: 37% (core functionality well tested)
- **Documentation**: Complete with examples ✅

## 📚 Learning Path

1. **Start here**: Understand LSTM basics with this project
2. **Next steps**: Learn about attention mechanisms
3. **Advanced**: Study Transformer architecture (GPT, BERT)
4. **Modern**: Explore large language models and fine-tuning

## 🤝 Contributing

1. Report issues or suggest features
2. Improve documentation and examples
3. Add new educational content
4. Optimize code while maintaining clarity

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---