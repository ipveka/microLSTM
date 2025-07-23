# Micro Language Model - Example Scripts

This directory contains comprehensive example scripts that demonstrate all aspects of the Micro Language Model package. Each script is designed to be both educational and practical, with extensive comments explaining the concepts and best practices.

## 📚 Available Examples

### 1. Training Demo (`training_demo.py`)
**Complete training pipeline demonstration**

```bash
python examples/training_demo.py
```

**What it demonstrates:**
- Complete training workflow from data to trained model
- Data preparation and analysis techniques
- Model configuration options and trade-offs
- Training process with progress monitoring
- Text generation validation
- Results saving and analysis

**Key learning objectives:**
- Understanding the complete training pipeline
- Model configuration and hyperparameter selection
- Training loop implementation and monitoring
- Best practices for neural language model training

**Estimated runtime:** 5-15 minutes (depending on configuration)

---

### 2. Text Generation Demo (`text_generation_demo.py`)
**Interactive text generation with multiple sampling strategies**

```bash
python examples/text_generation_demo.py
```

**What it demonstrates:**
- Multiple text sampling strategies (greedy, temperature, top-k, top-p)
- Interactive text generation session
- Batch generation for comparison
- Advanced generation controls
- Generation quality analysis

**Key learning objectives:**
- Understanding different sampling techniques
- Trade-offs between creativity and coherence
- Interactive experimentation with generation parameters
- Real-time parameter adjustment effects

**Estimated runtime:** 10-30 minutes (interactive)

---

### 3. Model Inspection Demo (`model_inspection_demo.py`)
**Comprehensive model analysis and visualization**

```bash
python examples/model_inspection_demo.py
```

**What it demonstrates:**
- Model architecture analysis and visualization
- Parameter distribution and health analysis
- Activation inspection and visualization
- Performance profiling and optimization
- Training progress analysis
- Utility functions for quick analysis

**Key learning objectives:**
- Understanding model internals and architecture
- Identifying potential training issues
- Performance optimization techniques
- Training dynamics analysis

**Estimated runtime:** 5-10 minutes

**Output:** Creates visualization plots in `inspection_demo_output/` directory

---

### 4. Configuration Examples (`configuration_examples.py`)
**Model configuration guidance for different use cases**

```bash
python examples/configuration_examples.py
```

**What it demonstrates:**
- System capability analysis
- Model configuration options (nano to xlarge)
- Configuration recommendations based on resources
- Interactive configuration builder
- Comprehensive configuration comparison
- Best practices for different scenarios

**Key learning objectives:**
- Choosing appropriate model configurations
- Understanding resource requirements
- Balancing performance vs computational constraints
- Configuration optimization for specific use cases

**Estimated runtime:** 5-15 minutes (interactive)

**Output:** Creates `configuration_guide.json` and `CONFIGURATION_GUIDE.md`

---

### 5. Data Preparation Demo (`data_preparation_demo.py`)
**Comprehensive data preparation techniques**

```bash
python examples/data_preparation_demo.py
```

**What it demonstrates:**
- Text data loading and cleaning techniques
- Character-level tokenization strategies
- Sequence preparation and batching
- Data quality analysis and validation
- Train/validation/test splitting
- Memory optimization techniques
- Efficient data loading strategies

**Key learning objectives:**
- Data preprocessing best practices
- Understanding tokenization trade-offs
- Efficient data pipeline implementation
- Memory optimization for large datasets

**Estimated runtime:** 5-10 minutes

---

## 🚀 Quick Start Guide

### Prerequisites
Ensure you have the Micro Language Model package installed:

```bash
pip install torch numpy matplotlib
# Install the micro_lm package (from project root)
pip install -e .
```

### Running Examples

1. **Start with Configuration Examples** to understand your system capabilities:
   ```bash
   python examples/configuration_examples.py
   ```

2. **Learn Data Preparation** to understand how to prepare your own data:
   ```bash
   python examples/data_preparation_demo.py
   ```

3. **Train Your First Model** with the training demo:
   ```bash
   python examples/training_demo.py
   ```

4. **Experiment with Text Generation**:
   ```bash
   python examples/text_generation_demo.py
   ```

5. **Analyze Your Model** with inspection tools:
   ```bash
   python examples/model_inspection_demo.py
   ```

## 📖 Learning Path

### For Beginners
1. **Configuration Examples** - Understand system requirements
2. **Training Demo** - Learn the complete workflow
3. **Text Generation Demo** - See the results in action

### For Intermediate Users
1. **Data Preparation Demo** - Master data preprocessing
2. **Model Inspection Demo** - Understand model internals
3. **Training Demo** - Experiment with different configurations

### For Advanced Users
- Use all examples as reference implementations
- Modify examples for your specific use cases
- Combine techniques from different examples

## 🛠️ Customization

### Using Your Own Data
Replace the sample text in any example with your own data:

```python
# Instead of using sample data
sample_text = get_sample_training_data()

# Use your own text file
with open('your_text_file.txt', 'r', encoding='utf-8') as f:
    your_text = f.read()
```

### Adjusting Model Size
Modify model configurations in any example:

```python
# Smaller model for quick testing
model = MicroLM(
    vocab_size=tokenizer.vocab_size(),
    embedding_dim=32,    # Smaller
    hidden_dim=64,       # Smaller
    num_layers=1,        # Fewer layers
    dropout=0.1
)

# Larger model for better quality
model = MicroLM(
    vocab_size=tokenizer.vocab_size(),
    embedding_dim=256,   # Larger
    hidden_dim=512,      # Larger
    num_layers=3,        # More layers
    dropout=0.3
)
```

### Modifying Training Parameters
Adjust training settings based on your needs:

```python
# Quick training for testing
trainer.train(
    data_loader=data_loader,
    epochs=10,           # Fewer epochs
    learning_rate=0.003, # Higher learning rate
    batch_size=16        # Smaller batch
)

# Thorough training for quality
trainer.train(
    data_loader=data_loader,
    epochs=100,          # More epochs
    learning_rate=0.001, # Lower learning rate
    batch_size=32        # Larger batch
)
```

## 🔧 Troubleshooting

### Common Issues

**Memory Errors:**
- Reduce batch size in training examples
- Use smaller model configurations
- Try the 'nano' or 'tiny' configurations

**Slow Training:**
- Use GPU if available (`torch.cuda.is_available()`)
- Reduce sequence length
- Use smaller datasets for testing

**Import Errors:**
- Ensure you're running from the project root directory
- Install all required dependencies
- Check that micro_lm package is properly installed

**Generation Quality Issues:**
- Train for more epochs
- Use larger model configurations
- Ensure sufficient training data
- Experiment with different sampling parameters

### Getting Help

1. **Check the comments** in each example script for detailed explanations
2. **Run examples step by step** to understand each component
3. **Modify parameters gradually** to see their effects
4. **Use the model inspection tools** to debug issues

## 📊 Expected Outputs

### Training Demo
- Model training progress with loss curves
- Generated text samples
- Training statistics and analysis
- Saved model files in `models/` directory

### Text Generation Demo
- Comparison of different sampling strategies
- Interactive generation session
- Quality analysis of generated text

### Model Inspection Demo
- Architecture analysis and recommendations
- Parameter distribution plots
- Activation visualizations
- Performance profiling results
- Training dynamics analysis

### Configuration Examples
- System capability analysis
- Configuration recommendations
- Comparison tables
- Configuration guide files

### Data Preparation Demo
- Text cleaning and analysis results
- Tokenization strategy comparisons
- Data splitting and validation
- Memory optimization recommendations

## 🎯 Next Steps

After running the examples:

1. **Experiment with your own data** using the techniques learned
2. **Try different model configurations** for your specific use case
3. **Combine techniques** from different examples
4. **Build your own applications** using the micro_lm package
5. **Contribute improvements** to the examples or package

## 📝 Notes

- All examples include extensive comments explaining the concepts
- Examples are designed to be educational first, practical second
- Runtime estimates are for reference and may vary based on your system
- Examples create output files/directories that can be safely deleted
- Each example is self-contained and can be run independently

Happy learning with the Micro Language Model! 🚀