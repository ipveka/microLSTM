#!/usr/bin/env python3
"""
Model Inspection and Visualization Demo

This script demonstrates the comprehensive model inspection and visualization
utilities provided by the Micro Language Model package. It shows how to:

1. Analyze model architecture and parameters
2. Visualize parameter distributions and health
3. Inspect intermediate activations
4. Profile model performance
5. Visualize training progress and dynamics
6. Generate optimization recommendations

The inspection utilities are designed to help understand how neural language
models work internally and identify potential issues or optimization opportunities.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import Micro LM components
from micro_lm import (
    MicroLM, CharacterTokenizer, ModelTrainer, 
    ModelInspector, TrainingVisualizer,
    inspect_model, visualize_training, analyze_parameters
)


def create_sample_model():
    """Create a sample model for demonstration."""
    print("Creating sample model...")
    
    # Create a moderately sized model for demonstration
    model = MicroLM(
        vocab_size=100,      # 100 character vocabulary
        embedding_dim=128,   # 128-dimensional embeddings
        hidden_dim=256,      # 256-dimensional LSTM hidden state
        num_layers=2,        # 2 LSTM layers
        dropout=0.2          # 20% dropout for regularization
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model


def create_sample_tokenizer():
    """Create a sample tokenizer with diverse text."""
    print("Creating sample tokenizer...")
    
    # Sample text corpus with various characters
    sample_text = """
    The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.
    Neural networks learn patterns in data through gradient descent optimization.
    Language models predict the next word or character in a sequence.
    Deep learning has revolutionized natural language processing and computer vision.
    Transformers and attention mechanisms have become the foundation of modern AI.
    
    Here are some numbers: 1234567890
    And some punctuation: !@#$%^&*()_+-=[]{}|;:,.<>?
    
    This text provides a diverse set of characters for tokenization and training.
    The model will learn to predict characters based on previous context.
    """
    
    tokenizer = CharacterTokenizer(sample_text)
    print(f"Tokenizer created with vocabulary size: {tokenizer.vocab_size()}")
    return tokenizer, sample_text


def demonstrate_architecture_analysis(model):
    """Demonstrate comprehensive architecture analysis."""
    print("\n" + "="*80)
    print("ARCHITECTURE ANALYSIS DEMONSTRATION")
    print("="*80)
    
    # Create inspector
    inspector = ModelInspector(model)
    
    # Print comprehensive model summary
    print("\n1. Comprehensive Model Summary:")
    print("-" * 50)
    inspector.print_model_summary(detailed=True)
    
    # Get detailed architecture summary
    print("\n2. Detailed Architecture Analysis:")
    print("-" * 50)
    summary = inspector.get_architecture_summary()
    
    # Show architectural insights
    insights = summary['architectural_insights']
    if insights['strengths']:
        print("\nModel Strengths:")
        for strength in insights['strengths']:
            print(f"  ✓ {strength}")
    
    if insights['potential_improvements']:
        print("\nPotential Improvements:")
        for improvement in insights['potential_improvements']:
            print(f"  → {improvement}")
    
    if insights['recommendations']:
        print("\nArchitecture Recommendations:")
        for rec in insights['recommendations']:
            print(f"  • {rec}")
    
    return inspector


def demonstrate_parameter_analysis(inspector, save_dir):
    """Demonstrate parameter analysis and visualization."""
    print("\n" + "="*80)
    print("PARAMETER ANALYSIS DEMONSTRATION")
    print("="*80)
    
    # Get parameter statistics
    print("\n1. Parameter Statistics Analysis:")
    print("-" * 50)
    stats = inspector.get_parameter_statistics()
    
    # Print overall statistics
    overall = stats['overall_stats']
    print(f"Total Parameters: {overall['total_parameters']:,}")
    print(f"Parameter Mean: {overall['mean']:.6f}")
    print(f"Parameter Std: {overall['std']:.6f}")
    print(f"Parameter Range: [{overall['min']:.6f}, {overall['max']:.6f}]")
    print(f"Zero Parameters: {overall['zero_fraction']:.2%}")
    
    # Print layer-wise statistics
    print("\nLayer-wise Parameter Statistics:")
    for layer_name, layer_stat in stats['layer_stats'].items():
        print(f"  {layer_name}:")
        print(f"    Shape: {layer_stat['shape']}")
        print(f"    Parameters: {layer_stat['total_params']:,}")
        print(f"    Mean: {layer_stat['mean']:.6f}")
        print(f"    Std: {layer_stat['std']:.6f}")
    
    # Print health indicators
    print("\nModel Health Indicators:")
    health = stats['health_indicators']
    for indicator, info in health.items():
        status_symbol = "✓" if info['status'] == 'healthy' else "⚠" if info['status'] == 'warning' else "✗"
        print(f"  {status_symbol} {indicator.replace('_', ' ').title()}: {info['message']}")
    
    # Visualize parameter distribution
    print("\n2. Parameter Distribution Visualization:")
    print("-" * 50)
    param_plot_path = save_dir / "parameter_distribution.png"
    inspector.visualize_parameter_distribution(save_path=str(param_plot_path), show_plot=False)
    print(f"Parameter distribution plot saved to: {param_plot_path}")
    
    return stats


def demonstrate_activation_inspection(inspector, tokenizer, save_dir):
    """Demonstrate activation inspection and visualization."""
    print("\n" + "="*80)
    print("ACTIVATION INSPECTION DEMONSTRATION")
    print("="*80)
    
    # Create sample input
    sample_text = "The quick brown fox"
    input_tokens = tokenizer.encode(sample_text)
    input_tensor = torch.tensor([input_tokens])  # Add batch dimension
    
    print(f"\n1. Inspecting activations for input: '{sample_text}'")
    print(f"Input tokens: {input_tokens}")
    print(f"Input tensor shape: {input_tensor.shape}")
    
    # Inspect activations
    print("\n2. Activation Analysis:")
    print("-" * 50)
    activations = inspector.inspect_activations(input_tensor)
    
    for layer_name, activation_info in activations.items():
        print(f"\n{layer_name.upper()} Layer:")
        print(f"  Shape: {activation_info['shape']}")
        print(f"  Mean: {activation_info['mean']:.6f}")
        print(f"  Std: {activation_info['std']:.6f}")
        print(f"  Range: [{activation_info['min']:.6f}, {activation_info['max']:.6f}]")
        print(f"  Zero Fraction: {activation_info['zero_fraction']:.2%}")
        print(f"  Activation Norm: {activation_info['activation_norm']:.6f}")
    
    # Visualize activations
    print("\n3. Activation Visualization:")
    print("-" * 50)
    activation_plot_path = save_dir / "activation_analysis.png"
    inspector.visualize_activations(input_tensor, save_path=str(activation_plot_path), show_plot=False)
    print(f"Activation visualization saved to: {activation_plot_path}")


def demonstrate_performance_profiling(inspector, tokenizer, save_dir):
    """Demonstrate model performance profiling."""
    print("\n" + "="*80)
    print("PERFORMANCE PROFILING DEMONSTRATION")
    print("="*80)
    
    # Create different sized inputs for profiling
    test_cases = [
        {"name": "Small Batch", "batch_size": 4, "seq_length": 20},
        {"name": "Medium Batch", "batch_size": 16, "seq_length": 50},
        {"name": "Large Batch", "batch_size": 32, "seq_length": 100}
    ]
    
    print("\n1. Performance Profiling Results:")
    print("-" * 50)
    
    for test_case in test_cases:
        print(f"\n{test_case['name']} ({test_case['batch_size']} x {test_case['seq_length']}):")
        
        # Create input tensor
        input_tensor = torch.randint(0, tokenizer.vocab_size(), 
                                   (test_case['batch_size'], test_case['seq_length']))
        
        # Profile performance
        profile_results = inspector.profile_model_performance(input_tensor, num_runs=20)
        
        # Print results
        total_time = profile_results['total_time']
        throughput = profile_results['throughput']
        memory = profile_results['memory_usage']
        
        print(f"  Forward Pass Time: {total_time['mean']*1000:.2f} ± {total_time['std']*1000:.2f} ms")
        print(f"  Throughput: {throughput['tokens_per_second']:.0f} tokens/sec")
        print(f"  Memory Usage: {memory['allocated_mb']:.1f} MB allocated")
        
        # Print optimization suggestions
        if profile_results['optimization_suggestions']:
            print("  Optimization Suggestions:")
            for suggestion in profile_results['optimization_suggestions'][:3]:  # Show top 3
                print(f"    • {suggestion}")


def simulate_training_history():
    """Simulate realistic training history for demonstration."""
    print("\n" + "="*80)
    print("SIMULATING TRAINING HISTORY")
    print("="*80)
    
    # Simulate 50 epochs of training with realistic loss curve
    epochs = 50
    
    # Create realistic loss curve (exponential decay with noise)
    base_loss = 3.0
    decay_rate = 0.05
    noise_level = 0.1
    
    train_losses = []
    epoch_times = []
    learning_rates = []
    
    for epoch in range(epochs):
        # Exponential decay with noise
        loss = base_loss * np.exp(-decay_rate * epoch) + np.random.normal(0, noise_level)
        loss = max(loss, 0.5)  # Minimum loss floor
        train_losses.append(loss)
        
        # Simulate varying epoch times
        base_time = 15.0  # 15 seconds base
        time_variation = np.random.normal(0, 1.0)
        epoch_time = max(base_time + time_variation, 5.0)
        epoch_times.append(epoch_time)
        
        # Learning rate schedule (decay every 20 epochs)
        if epoch < 20:
            lr = 0.001
        elif epoch < 40:
            lr = 0.0005
        else:
            lr = 0.0001
        learning_rates.append(lr)
    
    history = {
        'train_loss': train_losses,
        'epoch_times': epoch_times,
        'learning_rates': learning_rates
    }
    
    print(f"Simulated {epochs} epochs of training")
    print(f"Initial loss: {train_losses[0]:.4f}")
    print(f"Final loss: {train_losses[-1]:.4f}")
    print(f"Total improvement: {train_losses[0] - train_losses[-1]:.4f}")
    
    return history


def demonstrate_training_visualization(training_history, save_dir):
    """Demonstrate training progress visualization and analysis."""
    print("\n" + "="*80)
    print("TRAINING VISUALIZATION DEMONSTRATION")
    print("="*80)
    
    # Create training visualizer
    visualizer = TrainingVisualizer(training_history)
    
    # Plot training progress
    print("\n1. Training Progress Visualization:")
    print("-" * 50)
    training_plot_path = save_dir / "training_progress.png"
    visualizer.plot_training_progress(save_path=str(training_plot_path), show_plot=False)
    print(f"Training progress plot saved to: {training_plot_path}")
    
    # Analyze training dynamics
    print("\n2. Training Dynamics Analysis:")
    print("-" * 50)
    analysis = visualizer.analyze_training_dynamics()
    
    # Print convergence analysis
    convergence = analysis['convergence_analysis']
    print(f"\nConvergence Analysis:")
    print(f"  Total Improvement: {convergence['total_improvement']:.4f}")
    print(f"  Relative Improvement: {convergence['relative_improvement']:.2%}")
    print(f"  Convergence Rate: {convergence['convergence_rate']:.6f}")
    print(f"  Is Plateaued: {convergence['is_plateaued']}")
    print(f"  Best Loss: {convergence['best_loss']:.4f} (epoch {convergence['epochs_to_best']})")
    
    # Print stability metrics
    stability = analysis['stability_metrics']
    print(f"\nStability Metrics:")
    print(f"  Volatility: {stability['volatility']:.6f}")
    print(f"  Oscillation Rate: {stability['oscillation_rate']:.2%}")
    print(f"  Spike Count: {stability['spike_count']}")
    print(f"  Monotonicity: {stability['monotonicity']:.2%}")
    print(f"  Stability Score: {stability['stability_score']:.2%}")
    
    # Print efficiency metrics
    efficiency = analysis['efficiency_metrics']
    if 'time_metrics' in efficiency:
        time_metrics = efficiency['time_metrics']
        print(f"\nTime Efficiency:")
        print(f"  Mean Epoch Time: {time_metrics['mean_epoch_time']:.1f} seconds")
        print(f"  Total Training Time: {time_metrics['total_training_time']/60:.1f} minutes")
        print(f"  Time Stability: {time_metrics['time_stability']:.2%}")
    
    if 'loss_efficiency' in efficiency:
        loss_eff = efficiency['loss_efficiency']
        print(f"\nLoss Efficiency:")
        print(f"  Improvement per Epoch: {loss_eff['improvement_per_epoch']:.6f}")
        print(f"  Efficiency Score: {loss_eff['efficiency_score']:.4f}")
    
    # Print recommendations
    recommendations = analysis['recommendations']
    print(f"\nTraining Recommendations:")
    for rec in recommendations:
        print(f"  • {rec}")


def demonstrate_utility_functions(model, training_history, save_dir):
    """Demonstrate the convenient utility functions."""
    print("\n" + "="*80)
    print("UTILITY FUNCTIONS DEMONSTRATION")
    print("="*80)
    
    # Quick model inspection
    print("\n1. Quick Model Inspection:")
    print("-" * 50)
    inspect_model(model, detailed=False)
    
    # Quick training visualization
    print("\n2. Quick Training Visualization:")
    print("-" * 50)
    training_util_path = save_dir / "quick_training_viz.png"
    visualize_training(training_history, save_path=str(training_util_path))
    print(f"Quick training visualization saved to: {training_util_path}")
    
    # Quick parameter analysis
    print("\n3. Quick Parameter Analysis:")
    print("-" * 50)
    param_util_path = save_dir / "quick_param_analysis.png"
    param_stats = analyze_parameters(model, save_path=str(param_util_path))
    
    print(f"Parameter analysis completed:")
    print(f"  Total parameters: {param_stats['overall_stats']['total_parameters']:,}")
    print(f"  Parameter mean: {param_stats['overall_stats']['mean']:.6f}")
    print(f"  Parameter std: {param_stats['overall_stats']['std']:.6f}")
    print(f"  Visualization saved to: {param_util_path}")


def main():
    """Main demonstration function."""
    print("MICRO LANGUAGE MODEL - INSPECTION UTILITIES DEMO")
    print("="*80)
    print("This demo showcases the comprehensive model inspection and")
    print("visualization utilities for understanding neural language models.")
    print("="*80)
    
    # Create output directory for plots
    save_dir = Path("inspection_demo_output")
    save_dir.mkdir(exist_ok=True)
    print(f"\nSaving visualizations to: {save_dir.absolute()}")
    
    try:
        # 1. Create sample model and tokenizer
        model = create_sample_model()
        tokenizer, sample_text = create_sample_tokenizer()
        
        # 2. Demonstrate architecture analysis
        inspector = demonstrate_architecture_analysis(model)
        
        # 3. Demonstrate parameter analysis
        demonstrate_parameter_analysis(inspector, save_dir)
        
        # 4. Demonstrate activation inspection
        demonstrate_activation_inspection(inspector, tokenizer, save_dir)
        
        # 5. Demonstrate performance profiling
        demonstrate_performance_profiling(inspector, tokenizer, save_dir)
        
        # 6. Simulate and visualize training
        training_history = simulate_training_history()
        demonstrate_training_visualization(training_history, save_dir)
        
        # 7. Demonstrate utility functions
        demonstrate_utility_functions(model, training_history, save_dir)
        
        print("\n" + "="*80)
        print("DEMONSTRATION COMPLETE")
        print("="*80)
        print(f"All visualizations have been saved to: {save_dir.absolute()}")
        print("\nKey takeaways from this demonstration:")
        print("• Model inspection utilities provide deep insights into architecture")
        print("• Parameter analysis helps identify potential training issues")
        print("• Activation inspection reveals how information flows through the model")
        print("• Performance profiling identifies bottlenecks and optimization opportunities")
        print("• Training visualization helps monitor and debug the training process")
        print("• Utility functions provide quick access to common analysis tasks")
        print("\nThese tools are essential for understanding, debugging, and optimizing")
        print("neural language models in both research and production environments.")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        print("Make sure all dependencies are installed:")
        print("  pip install torch numpy matplotlib seaborn scipy")
        raise


if __name__ == "__main__":
    main()