# Micro Language Model 🧠

A simple, educational character-level LSTM language model for learning neural language modeling fundamentals.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Perfect for learning, experimentation, and understanding how language models work before diving into more complex architectures.

## 🚀 Quick Start

### Installation

```bash
git clone <repository-url>
cd micro-lm
pip install -r requirements.txt
pip install -e .
```

### Configuration Guide

Choose the right model configuration for your system:

```bash
python setup/setup_guide.py
```

### Basic Usage

```python
from micro_lm import CharacterTokenizer, MicroLM, ModelTrainer, TextGenerator

# 1. Prepare data and tokenizer
text_data = "Hello world! This is sample training text."
tokenizer = CharacterTokenizer(text_data)

# 2. Create and train model
model = MicroLM(vocab_size=tokenizer.vocab_size(), embedding_dim=128, hidden_dim=256, num_layers=2)
trainer = ModelTrainer(model, tokenizer)
data_loader = trainer.prepare_data(text_data, sequence_length=50)
trainer.train(data_loader, epochs=50, learning_rate=0.001)

# 3. Generate text
generator = TextGenerator(model, tokenizer)
generated_text = generator.generate("Hello", length=100, temperature=0.8)
print(f"Generated: {generated_text}")
```

## 📁 Project Structure

```
micro-lm/
├── micro_lm/              # Core package (tokenizer, model, trainer, generator)
├── setup/                 # Configuration guide
├── examples/              # Usage demos
└── tests/                 # Test suite
```

## 🎮 Examples

```bash
python examples/training_demo.py          # Complete workflow
python examples/text_generation_demo.py   # Generation examples  
python examples/model_inspection_demo.py  # Model analysis
```

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

**Quick reference:**

| Size | Parameters | Memory | Use Case |
|------|------------|--------|----------|
| Nano | ~2K | 50MB | Learning/Testing |
| Small | ~35K | 200MB | Prototyping |
| Large | ~600K | 1GB | Research/Quality |

## 🔧 Core Components

### CharacterTokenizer
Converts text to numbers and back:
```python
tokenizer = CharacterTokenizer("Hello world!")
encoded = tokenizer.encode("Hello")  # [7, 4, 11, 11, 14]
decoded = tokenizer.decode(encoded)  # "Hello"
```

### MicroLM
LSTM-based language model:
```python
model = MicroLM(vocab_size=50, embedding_dim=128, hidden_dim=256, num_layers=2)
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

**Happy Learning!** 🚀

For questions and contributions, visit our [GitHub repository](https://github.com/your-repo/micro-lm).