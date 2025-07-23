#!/usr/bin/env python3
"""
Interactive Text Generation Demo for Micro Language Model

This script provides a comprehensive demonstration of text generation capabilities
using the Micro Language Model. It showcases different sampling strategies,
interactive generation modes, and advanced generation techniques.

Key Features Demonstrated:
1. Multiple sampling strategies (greedy, temperature, top-k, top-p)
2. Interactive text generation with user input
3. Batch generation for comparing different approaches
4. Advanced generation controls (repetition penalty, stop tokens)
5. Generation quality analysis and comparison
6. Real-time parameter adjustment and experimentation

This demo is designed to be both educational and practical, helping users
understand how different generation parameters affect output quality and style.
"""

import torch
from pathlib import Path
import json
import time
from typing import Dict, List, Any, Optional

# Import Micro LM components
from micro_lm import (
    CharacterTokenizer, MicroLM, ModelTrainer, TextGenerator,
    GenerationError, ModelConfigurationError
)


def load_or_create_model() -> tuple[MicroLM, CharacterTokenizer]:
    """
    Load a pre-trained model or create and train a simple one for demonstration.
    
    This function first attempts to load a previously trained model. If none
    exists, it creates and trains a small model on sample data for demonstration
    purposes.
    
    Returns:
        tuple: (model, tokenizer) ready for text generation
    """
    print("🔍 LOADING OR CREATING MODEL FOR GENERATION")
    print("="*60)
    
    # Check for existing trained models
    models_dir = Path("./models")
    model_files = list(models_dir.glob("*/model_final.pt")) if models_dir.exists() else []
    
    if model_files:
        print("Found existing trained models:")
        for i, model_file in enumerate(model_files):
            print(f"  {i+1}. {model_file.parent.name}")
        
        # For demo purposes, use the first available model
        chosen_model = model_files[0]
        print(f"\nUsing model: {chosen_model.parent.name}")
        
        try:
            # Load model configuration and tokenizer
            config_file = chosen_model.parent / "config.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                # Recreate tokenizer (in practice, you'd save/load this too)
                sample_text = get_demo_text()
                tokenizer = CharacterTokenizer(sample_text)
                
                # Recreate and load model
                model = MicroLM(**config['model_params'])
                model.load_state_dict(torch.load(chosen_model, map_location='cpu'))
                model.eval()
                
                print(f"✓ Successfully loaded trained model")
                return model, tokenizer
                
            else:
                print("⚠ Model config not found, creating new model...")
                
        except Exception as e:
            print(f"⚠ Failed to load model: {e}")
            print("Creating new model for demonstration...")
    
    # Create and train a simple model for demonstration
    print("Creating and training a simple model for demonstration...")
    print("(This will take a few minutes)")
    
    # Get sample text
    sample_text = get_demo_text()
    tokenizer = CharacterTokenizer(sample_text)
    
    # Create a small, fast-training model
    model = MicroLM(
        vocab_size=tokenizer.vocab_size(),
        embedding_dim=64,
        hidden_dim=128,
        num_layers=2,
        dropout=0.2
    )
    
    # Quick training for demonstration
    trainer = ModelTrainer(model, tokenizer)
    data_loader = trainer.prepare_data(
        text=sample_text,
        sequence_length=25,
        batch_size=8,
        shuffle=True
    )
    
    print("Training model (this may take a few minutes)...")
    trainer.train(
        data_loader=data_loader,
        epochs=15,  # Quick training for demo
        learning_rate=0.002,
        save_path="./models/demo_generation"
    )
    
    print("✓ Model training completed")
    return model, tokenizer


def get_demo_text() -> str:
    """
    Provide sample text for model training/demonstration.
    
    Returns:
        str: Sample text with diverse patterns for generation
    """
    return """
    Welcome to the world of artificial intelligence and machine learning.
    Neural networks have revolutionized how we approach complex problems in computer science.
    
    Language models are particularly fascinating because they learn to understand and generate human language.
    These models work by analyzing patterns in text data and learning statistical relationships between words and characters.
    
    The process begins with tokenization, where text is converted into numerical representations.
    Then, neural networks process these numbers to learn patterns and relationships.
    During training, the model adjusts its parameters to better predict the next word or character in a sequence.
    
    Once trained, language models can generate new text that follows similar patterns to their training data.
    This capability has led to applications in creative writing, code generation, translation, and conversation.
    
    The key to successful text generation lies in balancing creativity with coherence.
    Too much randomness produces nonsensical text, while too little creates repetitive, boring output.
    Modern techniques like temperature sampling, top-k filtering, and nucleus sampling help achieve this balance.
    
    As you experiment with different generation parameters, you'll discover how each setting affects the style and quality of generated text.
    This hands-on exploration is the best way to understand how language models work and how to use them effectively.
    
    Remember that language modeling is both an art and a science.
    The technical aspects involve neural network architectures, training algorithms, and optimization techniques.
    The artistic aspects involve choosing the right parameters, crafting good prompts, and interpreting the results.
    
    Whether you're interested in creative applications, research, or practical tools, understanding language models opens up a world of possibilities.
    The field continues to evolve rapidly, with new techniques and applications emerging regularly.
    
    This demo will help you explore these concepts through hands-on experimentation with text generation.
    Try different prompts, adjust parameters, and observe how the model responds to various inputs.
    """


def demonstrate_sampling_strategies(generator: TextGenerator) -> Dict[str, str]:
    """
    Demonstrate different text sampling strategies with the same prompt.
    
    This function shows how different sampling parameters affect the style
    and quality of generated text, helping users understand the trade-offs
    between creativity and coherence.
    
    Args:
        generator (TextGenerator): Initialized text generator
    
    Returns:
        Dict[str, str]: Generated texts for each strategy
    """
    print("\n🎯 SAMPLING STRATEGIES DEMONSTRATION")
    print("="*60)
    print("Comparing different sampling strategies with the same prompt...")
    
    # Test prompt that should work well with our training data
    prompt = "Neural networks are"
    generation_length = 100
    
    print(f"Prompt: '{prompt}'")
    print(f"Generation length: {generation_length} characters")
    print("-" * 60)
    
    # Define sampling strategies to demonstrate
    strategies = [
        {
            'name': 'Greedy Decoding',
            'params': {'temperature': 0.0},
            'description': 'Always selects most probable character (deterministic)'
        },
        {
            'name': 'Low Temperature',
            'params': {'temperature': 0.3},
            'description': 'Conservative sampling, focuses on likely continuations'
        },
        {
            'name': 'Medium Temperature',
            'params': {'temperature': 0.7},
            'description': 'Balanced creativity and coherence'
        },
        {
            'name': 'High Temperature',
            'params': {'temperature': 1.2},
            'description': 'Creative sampling, more diverse but potentially less coherent'
        },
        {
            'name': 'Top-K Sampling',
            'params': {'temperature': 0.8, 'top_k': 10},
            'description': 'Limits choices to 10 most probable characters'
        },
        {
            'name': 'Nucleus Sampling',
            'params': {'temperature': 0.8, 'top_p': 0.9},
            'description': 'Dynamic vocabulary based on cumulative probability'
        }
    ]
    
    results = {}
    
    for strategy in strategies:
        print(f"\n{strategy['name']}:")
        print(f"  Description: {strategy['description']}")
        print(f"  Parameters: {strategy['params']}")
        
        try:
            # Generate text with current strategy
            generated_text = generator.generate(
                prompt=prompt,
                length=generation_length,
                **strategy['params']
            )
            
            # Extract generated part (after prompt)
            generated_part = generated_text[len(prompt):]
            results[strategy['name']] = generated_text
            
            print(f"  Result: '{prompt}{generated_part}'")
            
        except Exception as e:
            print(f"  ❌ Generation failed: {e}")
            results[strategy['name']] = f"Error: {e}"
    
    return results


def interactive_generation_session(generator: TextGenerator):
    """
    Provide an interactive text generation session where users can experiment
    with different prompts and parameters in real-time.
    
    Args:
        generator (TextGenerator): Initialized text generator
    """
    print("\n🎮 INTERACTIVE GENERATION SESSION")
    print("="*60)
    print("Welcome to the interactive text generation session!")
    print("You can experiment with different prompts and parameters.")
    print("\nCommands:")
    print("  'help' - Show available commands")
    print("  'params' - Show/modify generation parameters")
    print("  'quit' - Exit interactive session")
    print("  Or just type a prompt to generate text")
    print("-" * 60)
    
    # Default generation parameters
    params = {
        'length': 80,
        'temperature': 0.7,
        'top_k': None,
        'top_p': None,
        'repetition_penalty': 1.0
    }
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                print("Goodbye! Thanks for experimenting with text generation.")
                break
            
            elif user_input.lower() == 'help':
                print("\nAvailable commands:")
                print("  help - Show this help message")
                print("  params - View/modify generation parameters")
                print("  examples - Show example prompts")
                print("  compare <prompt> - Compare multiple strategies")
                print("  quit - Exit session")
                print("\nOr enter any text as a prompt for generation")
            
            elif user_input.lower() == 'params':
                show_and_modify_params(params)
            
            elif user_input.lower() == 'examples':
                show_example_prompts()
            
            elif user_input.lower().startswith('compare '):
                prompt = user_input[8:].strip()
                if prompt:
                    compare_strategies_for_prompt(generator, prompt, params['length'])
                else:
                    print("Please provide a prompt after 'compare'")
            
            else:
                # Treat input as a generation prompt
                print(f"\nGenerating text for prompt: '{user_input}'")
                print(f"Parameters: {params}")
                print("-" * 40)
                
                start_time = time.time()
                generated_text = generator.generate(
                    prompt=user_input,
                    **params
                )
                generation_time = time.time() - start_time
                
                # Extract generated part
                generated_part = generated_text[len(user_input):]
                
                print(f"Result: {user_input}{generated_part}")
                print(f"Generation time: {generation_time:.2f} seconds")
                
        except KeyboardInterrupt:
            print("\n\nSession interrupted. Type 'quit' to exit properly.")
        except Exception as e:
            print(f"❌ Error: {e}")


def show_and_modify_params(params: Dict[str, Any]):
    """
    Display current parameters and allow user to modify them.
    
    Args:
        params (Dict): Current generation parameters
    """
    print("\nCurrent generation parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    print("\nTo modify a parameter, type: <parameter_name> <new_value>")
    print("Or press Enter to keep current settings")
    
    user_input = input("Modify parameter: ").strip()
    if not user_input:
        return
    
    try:
        parts = user_input.split()
        if len(parts) != 2:
            print("Format: <parameter_name> <new_value>")
            return
        
        param_name, param_value = parts
        
        if param_name not in params:
            print(f"Unknown parameter: {param_name}")
            print(f"Available parameters: {list(params.keys())}")
            return
        
        # Convert value to appropriate type
        if param_name == 'length':
            params[param_name] = int(param_value)
        elif param_name in ['temperature', 'top_p', 'repetition_penalty']:
            params[param_name] = float(param_value)
        elif param_name == 'top_k':
            params[param_name] = int(param_value) if param_value.lower() != 'none' else None
        
        print(f"✓ Updated {param_name} to {params[param_name]}")
        
    except ValueError as e:
        print(f"❌ Invalid value: {e}")


def show_example_prompts():
    """Display example prompts that work well with the model."""
    print("\nExample prompts to try:")
    examples = [
        "Neural networks are",
        "The process of machine learning",
        "Language models can",
        "Training a model involves",
        "Artificial intelligence is",
        "The future of technology",
        "Deep learning algorithms",
        "Text generation works by"
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"  {i}. \"{example}\"")


def compare_strategies_for_prompt(generator: TextGenerator, prompt: str, length: int):
    """
    Compare multiple generation strategies for a specific prompt.
    
    Args:
        generator (TextGenerator): Text generator
        prompt (str): Prompt to use for comparison
        length (int): Length of text to generate
    """
    print(f"\nComparing strategies for prompt: '{prompt}'")
    print("-" * 50)
    
    strategies = [
        {'name': 'Greedy', 'params': {'temperature': 0.0}},
        {'name': 'Conservative', 'params': {'temperature': 0.3}},
        {'name': 'Balanced', 'params': {'temperature': 0.7}},
        {'name': 'Creative', 'params': {'temperature': 1.0}}
    ]
    
    for strategy in strategies:
        try:
            generated_text = generator.generate(
                prompt=prompt,
                length=length,
                **strategy['params']
            )
            generated_part = generated_text[len(prompt):]
            
            print(f"\n{strategy['name']} (T={strategy['params']['temperature']}):")
            print(f"  {prompt}{generated_part}")
            
        except Exception as e:
            print(f"\n{strategy['name']}: ❌ Error: {e}")


def batch_generation_demo(generator: TextGenerator):
    """
    Demonstrate batch generation for comparing multiple outputs.
    
    Args:
        generator (TextGenerator): Text generator
    """
    print("\n📦 BATCH GENERATION DEMONSTRATION")
    print("="*60)
    print("Generating multiple variations of the same prompt...")
    
    prompt = "The future of artificial intelligence"
    num_variations = 5
    length = 60
    
    print(f"Prompt: '{prompt}'")
    print(f"Generating {num_variations} variations with temperature=0.8")
    print("-" * 60)
    
    for i in range(num_variations):
        try:
            generated_text = generator.generate(
                prompt=prompt,
                length=length,
                temperature=0.8,
                seed=i  # Different seed for each variation
            )
            generated_part = generated_text[len(prompt):]
            
            print(f"\nVariation {i+1}:")
            print(f"  {prompt}{generated_part}")
            
        except Exception as e:
            print(f"\nVariation {i+1}: ❌ Error: {e}")


def advanced_generation_features(generator: TextGenerator):
    """
    Demonstrate advanced generation features like repetition penalty and stop tokens.
    
    Args:
        generator (TextGenerator): Text generator
    """
    print("\n🚀 ADVANCED GENERATION FEATURES")
    print("="*60)
    
    prompt = "Machine learning is"
    
    # Demonstrate repetition penalty
    print("1. Repetition Penalty Demonstration:")
    print(f"   Prompt: '{prompt}'")
    print("-" * 40)
    
    penalties = [1.0, 1.2, 1.5]
    for penalty in penalties:
        try:
            generated_text = generator.generate(
                prompt=prompt,
                length=80,
                temperature=0.8,
                repetition_penalty=penalty
            )
            generated_part = generated_text[len(prompt):]
            
            print(f"\nRepetition penalty {penalty}:")
            print(f"  {prompt}{generated_part}")
            
        except Exception as e:
            print(f"\nRepetition penalty {penalty}: ❌ Error: {e}")
    
    # Demonstrate stop tokens
    print("\n\n2. Stop Tokens Demonstration:")
    print(f"   Prompt: '{prompt}'")
    print("   Stop tokens: ['.', '!', '?']")
    print("-" * 40)
    
    try:
        generated_text = generator.generate(
            prompt=prompt,
            length=200,  # Long length, but should stop at punctuation
            temperature=0.7,
            stop_tokens=['.', '!', '?']
        )
        generated_part = generated_text[len(prompt):]
        
        print(f"\nResult (stopped at punctuation):")
        print(f"  {prompt}{generated_part}")
        
    except Exception as e:
        print(f"\nStop tokens demo: ❌ Error: {e}")


def generation_quality_analysis(results: Dict[str, str]):
    """
    Analyze and compare the quality of different generation results.
    
    Args:
        results (Dict): Generation results from different strategies
    """
    print("\n📊 GENERATION QUALITY ANALYSIS")
    print("="*60)
    print("Analyzing characteristics of different generation strategies...")
    
    for strategy_name, generated_text in results.items():
        if generated_text.startswith("Error:"):
            continue
        
        print(f"\n{strategy_name}:")
        
        # Basic statistics
        length = len(generated_text)
        word_count = len(generated_text.split())
        avg_word_length = length / word_count if word_count > 0 else 0
        
        # Character diversity
        unique_chars = len(set(generated_text.lower()))
        char_diversity = unique_chars / length if length > 0 else 0
        
        # Repetition analysis (simple)
        words = generated_text.lower().split()
        unique_words = len(set(words))
        word_diversity = unique_words / len(words) if words else 0
        
        print(f"  Length: {length} characters, {word_count} words")
        print(f"  Average word length: {avg_word_length:.1f}")
        print(f"  Character diversity: {char_diversity:.3f}")
        print(f"  Word diversity: {word_diversity:.3f}")
        
        # Simple coherence check (presence of common patterns)
        coherence_indicators = [' the ', ' and ', ' of ', ' to ', ' in ']
        coherence_score = sum(1 for indicator in coherence_indicators if indicator in generated_text.lower())
        print(f"  Coherence indicators: {coherence_score}/5")


def main():
    """
    Main function that orchestrates the text generation demonstration.
    
    This function provides a comprehensive tour of text generation capabilities,
    from basic sampling strategies to advanced interactive features.
    """
    print("MICRO LANGUAGE MODEL - TEXT GENERATION DEMO")
    print("="*80)
    print("This demo showcases the text generation capabilities of the Micro Language Model.")
    print("You'll explore different sampling strategies, interactive generation, and advanced features.")
    print("="*80)
    
    try:
        # Step 1: Load or create model
        model, tokenizer = load_or_create_model()
        
        # Step 2: Initialize generator
        print("\n🔧 INITIALIZING TEXT GENERATOR")
        generator = TextGenerator(model, tokenizer)
        print("✓ Text generator ready")
        
        # Step 3: Demonstrate sampling strategies
        print("\n" + "="*80)
        results = demonstrate_sampling_strategies(generator)
        
        # Step 4: Quality analysis
        generation_quality_analysis(results)
        
        # Step 5: Batch generation demo
        batch_generation_demo(generator)
        
        # Step 6: Advanced features
        advanced_generation_features(generator)
        
        # Step 7: Interactive session
        print("\n" + "="*80)
        print("Ready for interactive generation session!")
        print("This is where you can experiment freely with prompts and parameters.")
        
        response = input("\nStart interactive session? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            interactive_generation_session(generator)
        
        print("\n" + "="*80)
        print("TEXT GENERATION DEMO COMPLETED!")
        print("="*80)
        print("Key insights from this demonstration:")
        print("• Temperature controls the creativity vs coherence trade-off")
        print("• Top-k and top-p sampling provide fine-grained control")
        print("• Repetition penalty helps avoid repetitive text")
        print("• Stop tokens enable controlled generation length")
        print("• Interactive experimentation is key to understanding generation")
        print("\nNext steps to explore:")
        print("• Try your own prompts and see how the model responds")
        print("• Experiment with different parameter combinations")
        print("• Train models on different types of text data")
        print("• Explore prompt engineering techniques")
        print("• Compare outputs from models trained on different data")
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\nDemo failed with error: {e}")
        print("This might be due to:")
        print("• No trained model available (run training_demo.py first)")
        print("• Insufficient memory for text generation")
        print("• Model compatibility issues")
        raise


if __name__ == "__main__":
    main()