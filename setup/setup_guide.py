#!/usr/bin/env python3
"""
Model Configuration Examples for MicroLSTM

This script demonstrates different model configurations for various use cases,
showing how to balance model capacity, training time, and resource requirements.
It provides practical examples of configuring models for different scenarios
and explains the trade-offs involved in each choice.

Key Configuration Aspects Covered:
1. Model size variations (tiny to large)
2. Training parameter optimization
3. Memory and computational requirements
4. Use case specific configurations
5. Performance vs resource trade-offs
6. Best practices for different scenarios

This demo helps users understand how to choose appropriate configurations
for their specific needs and computational constraints.
"""

import torch
import torch.nn as nn
from pathlib import Path
import json
import time
import psutil
import os
from typing import Dict, List, Any, Tuple

# Import Micro LM components
from micro_lstm import (
    CharacterTokenizer, MicroLSTM, ModelTrainer, TextGenerator,
    ModelConfigurationError
)


def get_system_info() -> Dict[str, Any]:
    """
    Get system information to help recommend appropriate configurations.
    
    Returns:
        Dict[str, Any]: System specifications and capabilities
    """
    print("🖥️  SYSTEM ANALYSIS")
    print("="*60)
    print("Analyzing system capabilities to recommend configurations...")
    
    # Memory information
    memory = psutil.virtual_memory()
    total_memory_gb = memory.total / (1024**3)
    available_memory_gb = memory.available / (1024**3)
    
    # CPU information
    cpu_count = psutil.cpu_count()
    cpu_freq = psutil.cpu_freq()
    
    # GPU information (if available)
    gpu_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if gpu_available else 0
    gpu_memory = []
    
    if gpu_available:
        for i in range(gpu_count):
            gpu_props = torch.cuda.get_device_properties(i)
            gpu_memory.append(gpu_props.total_memory / (1024**3))
    
    system_info = {
        'memory': {
            'total_gb': round(total_memory_gb, 1),
            'available_gb': round(available_memory_gb, 1),
            'usage_percent': memory.percent
        },
        'cpu': {
            'count': cpu_count,
            'frequency_mhz': round(cpu_freq.current) if cpu_freq else 'Unknown'
        },
        'gpu': {
            'available': gpu_available,
            'count': gpu_count,
            'memory_gb': [round(mem, 1) for mem in gpu_memory]
        }
    }
    
    print(f"Memory: {system_info['memory']['total_gb']} GB total, "
          f"{system_info['memory']['available_gb']} GB available")
    print(f"CPU: {system_info['cpu']['count']} cores @ {system_info['cpu']['frequency_mhz']} MHz")
    
    if gpu_available:
        print(f"GPU: {gpu_count} device(s) available")
        for i, mem in enumerate(gpu_memory):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"  GPU {i}: {gpu_name} ({mem} GB)")
    else:
        print("GPU: Not available (CPU-only)")
    
    return system_info


def define_model_configurations() -> Dict[str, Dict[str, Any]]:
    """
    Define comprehensive model configurations for different use cases.
    
    Returns:
        Dict[str, Dict[str, Any]]: Named configurations with detailed specifications
    """
    configurations = {
        'nano': {
            'description': 'Ultra-minimal model for proof-of-concept and testing',
            'model_params': {
                'embedding_dim': 16,
                'hidden_dim': 32,
                'num_layers': 1,
                'dropout': 0.1
            },
            'training_params': {
                'sequence_length': 15,
                'batch_size': 32,
                'learning_rate': 0.005,
                'epochs': 10,
                'optimizer': 'adam'
            },
            'use_cases': [
                'Quick prototyping and testing',
                'Educational demonstrations',
                'Resource-constrained environments',
                'Proof-of-concept implementations'
            ],
            'requirements': {
                'min_memory_mb': 50,
                'training_time_minutes': 2,
                'parameters_approx': 2000,
                'recommended_text_size': 1000
            },
            'pros': [
                'Extremely fast training',
                'Minimal memory usage',
                'Good for learning basics',
                'Quick iteration cycles'
            ],
            'cons': [
                'Very limited generation quality',
                'Cannot learn complex patterns',
                'Suitable only for simple tasks',
                'Limited vocabulary handling'
            ]
        },
        
        'tiny': {
            'description': 'Minimal viable model for simple text generation',
            'model_params': {
                'embedding_dim': 32,
                'hidden_dim': 64,
                'num_layers': 1,
                'dropout': 0.1
            },
            'training_params': {
                'sequence_length': 20,
                'batch_size': 24,
                'learning_rate': 0.003,
                'epochs': 20,
                'optimizer': 'adam'
            },
            'use_cases': [
                'Simple text completion',
                'Educational projects',
                'Mobile applications',
                'Embedded systems'
            ],
            'requirements': {
                'min_memory_mb': 100,
                'training_time_minutes': 5,
                'parameters_approx': 8000,
                'recommended_text_size': 5000
            },
            'pros': [
                'Fast training and inference',
                'Low memory footprint',
                'Good for simple patterns',
                'Easy to understand and debug'
            ],
            'cons': [
                'Limited coherence in long text',
                'Simple vocabulary patterns only',
                'May struggle with complex grammar',
                'Repetitive output tendency'
            ]
        },
        
        'small': {
            'description': 'Compact model with reasonable performance for most tasks',
            'model_params': {
                'embedding_dim': 64,
                'hidden_dim': 128,
                'num_layers': 2,
                'dropout': 0.2
            },
            'training_params': {
                'sequence_length': 30,
                'batch_size': 16,
                'learning_rate': 0.002,
                'epochs': 30,
                'optimizer': 'adam'
            },
            'use_cases': [
                'General text generation',
                'Creative writing assistance',
                'Code completion (simple)',
                'Personal projects'
            ],
            'requirements': {
                'min_memory_mb': 200,
                'training_time_minutes': 15,
                'parameters_approx': 35000,
                'recommended_text_size': 20000
            },
            'pros': [
                'Good balance of speed and quality',
                'Handles moderate complexity well',
                'Reasonable training time',
                'Suitable for most applications'
            ],
            'cons': [
                'May struggle with very long contexts',
                'Limited handling of rare patterns',
                'Moderate memory requirements',
                'Needs decent amount of training data'
            ]
        },
        
        'medium': {
            'description': 'Well-balanced model for professional applications',
            'model_params': {
                'embedding_dim': 128,
                'hidden_dim': 256,
                'num_layers': 2,
                'dropout': 0.2
            },
            'training_params': {
                'sequence_length': 40,
                'batch_size': 12,
                'learning_rate': 0.001,
                'epochs': 50,
                'optimizer': 'adam'
            },
            'use_cases': [
                'Professional text generation',
                'Content creation tools',
                'Research applications',
                'Production systems (small scale)'
            ],
            'requirements': {
                'min_memory_mb': 500,
                'training_time_minutes': 45,
                'parameters_approx': 150000,
                'recommended_text_size': 100000
            },
            'pros': [
                'Good text quality and coherence',
                'Handles complex patterns well',
                'Suitable for production use',
                'Good generalization ability'
            ],
            'cons': [
                'Longer training time required',
                'Higher memory requirements',
                'Needs substantial training data',
                'More complex hyperparameter tuning'
            ]
        },
        
        'large': {
            'description': 'High-capacity model for demanding applications',
            'model_params': {
                'embedding_dim': 256,
                'hidden_dim': 512,
                'num_layers': 3,
                'dropout': 0.3
            },
            'training_params': {
                'sequence_length': 50,
                'batch_size': 8,
                'learning_rate': 0.0008,
                'epochs': 75,
                'optimizer': 'adam'
            },
            'use_cases': [
                'High-quality text generation',
                'Research and experimentation',
                'Large-scale applications',
                'Complex pattern learning'
            ],
            'requirements': {
                'min_memory_mb': 1000,
                'training_time_minutes': 120,
                'parameters_approx': 600000,
                'recommended_text_size': 500000
            },
            'pros': [
                'Excellent text quality',
                'Handles complex patterns and contexts',
                'Good for research applications',
                'High-quality creative generation'
            ],
            'cons': [
                'Long training time',
                'High memory requirements',
                'Needs large datasets',
                'Risk of overfitting on small data'
            ]
        },
        
        'xlarge': {
            'description': 'Maximum capacity model for research and specialized applications',
            'model_params': {
                'embedding_dim': 512,
                'hidden_dim': 1024,
                'num_layers': 4,
                'dropout': 0.3
            },
            'training_params': {
                'sequence_length': 60,
                'batch_size': 4,
                'learning_rate': 0.0005,
                'epochs': 100,
                'optimizer': 'adam'
            },
            'use_cases': [
                'Research applications',
                'Specialized high-quality generation',
                'Large-scale production systems',
                'Complex domain-specific tasks'
            ],
            'requirements': {
                'min_memory_mb': 2000,
                'training_time_minutes': 300,
                'parameters_approx': 2000000,
                'recommended_text_size': 1000000
            },
            'pros': [
                'State-of-the-art text quality',
                'Excellent pattern recognition',
                'Handles very complex contexts',
                'Research-grade performance'
            ],
            'cons': [
                'Very long training time',
                'High computational requirements',
                'Needs very large datasets',
                'Potential for overfitting'
            ]
        }
    }
    
    return configurations


def recommend_configuration(system_info: Dict[str, Any], use_case: str = None) -> List[str]:
    """
    Recommend appropriate configurations based on system capabilities and use case.
    
    Args:
        system_info (Dict): System specifications
        use_case (str, optional): Specific use case description
    
    Returns:
        List[str]: Recommended configuration names
    """
    print("\n🎯 CONFIGURATION RECOMMENDATIONS")
    print("="*60)
    
    available_memory_mb = system_info['memory']['available_gb'] * 1024
    has_gpu = system_info['gpu']['available']
    cpu_cores = system_info['cpu']['count']
    
    print(f"Based on your system capabilities:")
    print(f"  Available memory: {available_memory_mb:.0f} MB")
    print(f"  GPU acceleration: {'Yes' if has_gpu else 'No'}")
    print(f"  CPU cores: {cpu_cores}")
    
    configurations = define_model_configurations()
    recommendations = []
    
    # Memory-based filtering
    for config_name, config in configurations.items():
        min_memory = config['requirements']['min_memory_mb']
        if available_memory_mb >= min_memory * 2:  # 2x safety margin
            recommendations.append(config_name)
    
    if not recommendations:
        recommendations = ['nano']  # Fallback for very limited systems
    
    # Performance-based recommendations
    if has_gpu and available_memory_mb > 1000:
        print("\n✅ Your system can handle larger models efficiently")
        if 'xlarge' not in recommendations:
            recommendations.extend(['large', 'xlarge'])
    elif available_memory_mb > 500:
        print("\n✅ Your system is suitable for medium-sized models")
        if 'large' not in recommendations:
            recommendations.append('large')
    else:
        print("\n⚠️  Your system is best suited for smaller models")
    
    # Use case specific recommendations
    if use_case:
        print(f"\nFor use case '{use_case}':")
        use_case_lower = use_case.lower()
        
        if any(word in use_case_lower for word in ['research', 'production', 'quality']):
            print("  → Recommending higher-capacity models")
            recommendations = [r for r in recommendations if r in ['medium', 'large', 'xlarge']]
        elif any(word in use_case_lower for word in ['mobile', 'embedded', 'quick', 'demo']):
            print("  → Recommending lightweight models")
            recommendations = [r for r in recommendations if r in ['nano', 'tiny', 'small']]
    
    # Remove duplicates and sort by size
    size_order = ['nano', 'tiny', 'small', 'medium', 'large', 'xlarge']
    recommendations = sorted(list(set(recommendations)), key=lambda x: size_order.index(x))
    
    print(f"\n🎯 Recommended configurations: {', '.join(recommendations)}")
    return recommendations


def demonstrate_configuration(config_name: str, config: Dict[str, Any], vocab_size: int):
    """
    Demonstrate a specific model configuration with detailed analysis.
    
    Args:
        config_name (str): Configuration name
        config (Dict): Configuration details
        vocab_size (int): Vocabulary size for the model
    """
    print(f"\n📋 CONFIGURATION: {config_name.upper()}")
    print("="*60)
    
    print(f"Description: {config['description']}")
    
    # Model architecture details
    print(f"\n🏗️  Model Architecture:")
    model_params = config['model_params'].copy()
    model_params['vocab_size'] = vocab_size
    
    for param, value in model_params.items():
        print(f"  {param}: {value}")
    
    # Create model to get actual parameter count
    try:
        model = MicroLSTM(**model_params)
        actual_params = sum(p.numel() for p in model.parameters())
        model_size_mb = (actual_params * 4) / (1024 * 1024)  # Assuming float32
        
        print(f"\n📊 Model Statistics:")
        print(f"  Total parameters: {actual_params:,}")
        print(f"  Model size: {model_size_mb:.2f} MB")
        print(f"  Estimated parameters: {config['requirements']['parameters_approx']:,}")
        
    except Exception as e:
        print(f"\n❌ Could not create model: {e}")
        actual_params = config['requirements']['parameters_approx']
        model_size_mb = (actual_params * 4) / (1024 * 1024)
    
    # Training configuration
    print(f"\n🎓 Training Configuration:")
    for param, value in config['training_params'].items():
        print(f"  {param}: {value}")
    
    # Requirements and expectations
    print(f"\n💾 Resource Requirements:")
    req = config['requirements']
    print(f"  Minimum memory: {req['min_memory_mb']} MB")
    print(f"  Training time: ~{req['training_time_minutes']} minutes")
    print(f"  Recommended text size: {req['recommended_text_size']:,} characters")
    
    # Use cases
    print(f"\n🎯 Suitable Use Cases:")
    for use_case in config['use_cases']:
        print(f"  • {use_case}")
    
    # Pros and cons
    print(f"\n✅ Advantages:")
    for pro in config['pros']:
        print(f"  • {pro}")
    
    print(f"\n⚠️  Limitations:")
    for con in config['cons']:
        print(f"  • {con}")


def create_configuration_comparison_table(configurations: Dict[str, Dict[str, Any]]):
    """
    Create a comparison table of all configurations.
    
    Args:
        configurations (Dict): All model configurations
    """
    print("\n📊 CONFIGURATION COMPARISON TABLE")
    print("="*80)
    
    # Table headers
    headers = ['Config', 'Embed', 'Hidden', 'Layers', 'Params', 'Memory', 'Time', 'Quality']
    col_widths = [8, 6, 6, 6, 8, 8, 8, 8]
    
    # Print header
    header_row = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header_row)
    print("-" * len(header_row))
    
    # Print each configuration
    for config_name, config in configurations.items():
        model_params = config['model_params']
        req = config['requirements']
        
        # Estimate quality score (subjective)
        quality_scores = {'nano': 1, 'tiny': 2, 'small': 3, 'medium': 4, 'large': 5, 'xlarge': 5}
        quality = '★' * quality_scores.get(config_name, 3)
        
        row_data = [
            config_name,
            str(model_params['embedding_dim']),
            str(model_params['hidden_dim']),
            str(model_params['num_layers']),
            f"{req['parameters_approx']//1000}K",
            f"{req['min_memory_mb']}MB",
            f"{req['training_time_minutes']}min",
            quality
        ]
        
        row = " | ".join(data.ljust(width) for data, width in zip(row_data, col_widths))
        print(row)


def interactive_configuration_builder():
    """
    Interactive tool to help users build custom configurations.
    """
    print("\n🛠️  INTERACTIVE CONFIGURATION BUILDER")
    print("="*60)
    print("Let's build a custom configuration based on your needs!")
    
    # Gather requirements
    print("\nPlease answer a few questions:")
    
    # Use case
    print("\n1. What's your primary use case?")
    use_cases = [
        "Learning/Education",
        "Prototyping/Experimentation", 
        "Creative Writing",
        "Production Application",
        "Research Project"
    ]
    
    for i, use_case in enumerate(use_cases, 1):
        print(f"   {i}. {use_case}")
    
    try:
        use_case_choice = int(input("Enter choice (1-5): ")) - 1
        selected_use_case = use_cases[use_case_choice]
    except (ValueError, IndexError):
        selected_use_case = "General Purpose"
    
    # Quality vs Speed preference
    print("\n2. What's more important to you?")
    print("   1. Fast training and inference (lower quality)")
    print("   2. Balanced approach")
    print("   3. High quality output (slower training)")
    
    try:
        quality_choice = int(input("Enter choice (1-3): "))
    except ValueError:
        quality_choice = 2
    
    # Memory constraints
    print("\n3. What are your memory constraints?")
    print("   1. Very limited (<500MB)")
    print("   2. Moderate (500MB-2GB)")
    print("   3. High (2GB+)")
    
    try:
        memory_choice = int(input("Enter choice (1-3): "))
    except ValueError:
        memory_choice = 2
    
    # Build recommendation
    print(f"\n🎯 CUSTOM RECOMMENDATION")
    print("="*40)
    print(f"Use case: {selected_use_case}")
    print(f"Quality preference: {['Speed', 'Balanced', 'Quality'][quality_choice-1]}")
    print(f"Memory constraint: {['Limited', 'Moderate', 'High'][memory_choice-1]}")
    
    # Logic for recommendation
    if memory_choice == 1:  # Limited memory
        if quality_choice == 1:
            recommended = 'nano'
        else:
            recommended = 'tiny'
    elif memory_choice == 2:  # Moderate memory
        if quality_choice == 1:
            recommended = 'small'
        elif quality_choice == 2:
            recommended = 'medium'
        else:
            recommended = 'medium'
    else:  # High memory
        if quality_choice == 1:
            recommended = 'medium'
        elif quality_choice == 2:
            recommended = 'large'
        else:
            recommended = 'xlarge'
    
    print(f"\n✅ Recommended configuration: {recommended.upper()}")
    
    # Show the recommended configuration
    configurations = define_model_configurations()
    if recommended in configurations:
        demonstrate_configuration(recommended, configurations[recommended], 100)  # Assume vocab_size=100
    
    return recommended


def save_configuration_guide():
    """
    Save a comprehensive configuration guide to a file.
    """
    print("\n💾 SAVING CONFIGURATION GUIDE")
    print("="*60)
    
    configurations = define_model_configurations()
    
    guide = {
        'title': 'MicroLSTM Configuration Guide',
        'description': 'Comprehensive guide to model configurations for different use cases',
        'configurations': configurations,
        'selection_criteria': {
            'memory_requirements': 'Choose based on available system memory',
            'training_time': 'Consider your time constraints for training',
            'use_case': 'Match configuration to your specific application',
            'data_size': 'Larger models need more training data',
            'quality_needs': 'Higher capacity models produce better text'
        },
        'best_practices': [
            'Start with smaller configurations and scale up',
            'Monitor memory usage during training',
            'Use GPU acceleration when available',
            'Adjust batch size based on memory constraints',
            'Consider sequence length impact on memory',
            'Save models regularly during training',
            'Validate on held-out data to avoid overfitting'
        ]
    }
    
    # Save to file
    guide_file = Path("configuration_guide.json")
    with open(guide_file, 'w') as f:
        json.dump(guide, f, indent=2)
    
    print(f"✅ Configuration guide saved to: {guide_file}")
    
    # Also create a markdown version
    md_file = Path("CONFIGURATION_GUIDE.md")
    with open(md_file, 'w') as f:
        f.write("# MicroLSTM Configuration Guide\n\n")
        f.write("This guide helps you choose the right model configuration for your needs.\n\n")
        
        f.write("## Available Configurations\n\n")
        for config_name, config in configurations.items():
            f.write(f"### {config_name.upper()}\n\n")
            f.write(f"**Description:** {config['description']}\n\n")
            
            f.write("**Model Parameters:**\n")
            for param, value in config['model_params'].items():
                f.write(f"- {param}: {value}\n")
            f.write("\n")
            
            f.write("**Use Cases:**\n")
            for use_case in config['use_cases']:
                f.write(f"- {use_case}\n")
            f.write("\n")
            
            f.write("**Requirements:**\n")
            req = config['requirements']
            f.write(f"- Memory: {req['min_memory_mb']} MB\n")
            f.write(f"- Training time: ~{req['training_time_minutes']} minutes\n")
            f.write(f"- Parameters: ~{req['parameters_approx']:,}\n")
            f.write("\n")
    
    print(f"✅ Markdown guide saved to: {md_file}")


def main():
    """
    Main function demonstrating model configuration options and recommendations.
    """
    print("MICROLSTM - CONFIGURATION EXAMPLES")
    print("="*80)
    print("This demo helps you understand and choose the right model configuration")
    print("for your specific needs, system capabilities, and use cases.")
    print("="*80)
    
    try:
        # Step 1: Analyze system capabilities
        system_info = get_system_info()
        
        # Step 2: Show all available configurations
        print("\n📋 AVAILABLE CONFIGURATIONS")
        print("="*60)
        configurations = define_model_configurations()
        
        # Create sample tokenizer for parameter calculations
        sample_text = "Hello world! This is a sample text for tokenizer creation."
        tokenizer = CharacterTokenizer(sample_text)
        vocab_size = tokenizer.vocab_size()
        
        print(f"Using vocabulary size: {vocab_size} (from sample text)")
        
        # Step 3: Show comparison table
        create_configuration_comparison_table(configurations)
        
        # Step 4: Get recommendations
        recommendations = recommend_configuration(system_info)
        
        # Step 5: Demonstrate recommended configurations
        print("\n🔍 DETAILED CONFIGURATION ANALYSIS")
        print("="*60)
        
        for config_name in recommendations[:3]:  # Show top 3 recommendations
            if config_name in configurations:
                demonstrate_configuration(config_name, configurations[config_name], vocab_size)
        
        # Step 6: Interactive configuration builder
        print("\n" + "="*80)
        response = input("Would you like to use the interactive configuration builder? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            custom_config = interactive_configuration_builder()
        
        # Step 7: Save configuration guide
        print("\n" + "="*80)
        response = input("Save configuration guide to files? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            save_configuration_guide()
        
        print("\n" + "="*80)
        print("CONFIGURATION EXAMPLES COMPLETED!")
        print("="*80)
        print("Key takeaways:")
        print("• Model size should match your system capabilities")
        print("• Larger models need more data and training time")
        print("• Start small and scale up based on results")
        print("• Consider your specific use case requirements")
        print("• Balance quality needs with resource constraints")
        print("\nNext steps:")
        print("• Try training with your recommended configuration")
        print("• Experiment with different hyperparameters")
        print("• Monitor resource usage during training")
        print("• Compare results from different configurations")
        print("• Adjust based on your specific data and needs")
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\nDemo failed with error: {e}")
        print("This might be due to:")
        print("• System resource detection issues")
        print("• Missing dependencies")
        print("• Insufficient permissions for system analysis")
        raise


if __name__ == "__main__":
    main()