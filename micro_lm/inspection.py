"""
Model inspection and visualization utilities for the Micro Language Model.

This module provides comprehensive tools for analyzing and visualizing the
language model's architecture, parameters, training progress, and internal
representations. These utilities are designed for educational purposes to
help understand how neural language models work internally.

The module includes:
- Model architecture summary and parameter analysis
- Memory usage estimation and optimization suggestions
- Intermediate activation inspection and visualization
- Training progress visualization and analysis
- Weight distribution analysis and health monitoring
- Performance profiling and bottleneck identification

All utilities include detailed explanations of what each metric means and
how to interpret the results for model debugging and optimization.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import json
from pathlib import Path
import time
from collections import defaultdict, OrderedDict
import warnings

from .model import MicroLM
from .trainer import ModelTrainer
from .exceptions import ModelError, ModelConfigurationError


class ModelInspector:
    """
    Comprehensive model inspection and analysis utilities.
    
    This class provides tools to analyze model architecture, parameters,
    memory usage, and internal representations. It's designed to help
    understand and debug neural language models.
    
    Args:
        model (MicroLM): The language model to inspect
        
    Attributes:
        model (MicroLM): The model being inspected
        device (torch.device): Device the model is on
        
    Example:
        >>> model = MicroLM(vocab_size=50, embedding_dim=128, hidden_dim=256, num_layers=2)
        >>> inspector = ModelInspector(model)
        >>> summary = inspector.get_architecture_summary()
        >>> inspector.print_model_summary()
    """
    
    def __init__(self, model: MicroLM):
        """
        Initialize the model inspector.
        
        Args:
            model (MicroLM): The language model to inspect
            
        Raises:
            TypeError: If model is not a MicroLM instance
        """
        if not isinstance(model, MicroLM):
            raise ModelConfigurationError(
                f"model must be MicroLM instance, got {type(model)}",
                parameter="model",
                value=type(model)
            )
        
        self.model = model
        self.device = next(model.parameters()).device
        
        # Cache for activation hooks
        self._activation_hooks = {}
        self._activations = {}
        
    def get_architecture_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive architecture summary with detailed parameter analysis.
        
        This method provides a complete overview of the model architecture,
        including layer-by-layer parameter counts, memory usage estimates,
        and architectural insights.
        
        Returns:
            Dict[str, Any]: Comprehensive architecture information including:
                - basic_info: Model configuration and total parameters
                - layer_details: Detailed breakdown of each layer
                - parameter_analysis: Parameter distribution and statistics
                - memory_analysis: Memory usage estimates and optimization tips
                - architectural_insights: Analysis of model design choices
                
        The parameter analysis includes:
        - Total trainable parameters
        - Parameter distribution across layers
        - Memory footprint estimates
        - Computational complexity estimates
        """
        # Get basic model info
        basic_info = self.model.get_model_info()
        
        # Detailed layer analysis
        layer_details = self._analyze_layers()
        
        # Parameter distribution analysis
        param_analysis = self._analyze_parameters()
        
        # Memory usage analysis
        memory_analysis = self._analyze_memory_usage()
        
        # Architectural insights
        insights = self._generate_architectural_insights()
        
        return {
            'basic_info': basic_info,
            'layer_details': layer_details,
            'parameter_analysis': param_analysis,
            'memory_analysis': memory_analysis,
            'architectural_insights': insights,
            'generation_timestamp': time.time()
        }
    
    def _analyze_layers(self) -> Dict[str, Any]:
        """
        Analyze each layer in detail.
        
        Returns:
            Dict[str, Any]: Detailed layer analysis
        """
        layer_info = {}
        
        # Embedding layer analysis
        embedding = self.model.embedding
        layer_info['embedding'] = {
            'type': 'Embedding',
            'input_dim': embedding.num_embeddings,
            'output_dim': embedding.embedding_dim,
            'parameters': embedding.num_embeddings * embedding.embedding_dim,
            'memory_mb': (embedding.num_embeddings * embedding.embedding_dim * 4) / (1024**2),
            'description': 'Converts character indices to dense vector representations',
            'purpose': 'Maps discrete tokens to continuous space for neural processing'
        }
        
        # LSTM layer analysis
        lstm = self.model.lstm
        # LSTM parameter calculation: 4 * (input_size + hidden_size + 1) * hidden_size per layer
        lstm_params_per_layer = 4 * (lstm.input_size * lstm.hidden_size + 
                                    lstm.hidden_size * lstm.hidden_size + 
                                    lstm.hidden_size)
        total_lstm_params = lstm_params_per_layer * lstm.num_layers
        
        layer_info['lstm'] = {
            'type': 'LSTM',
            'input_size': lstm.input_size,
            'hidden_size': lstm.hidden_size,
            'num_layers': lstm.num_layers,
            'parameters': total_lstm_params,
            'parameters_per_layer': lstm_params_per_layer,
            'memory_mb': (total_lstm_params * 4) / (1024**2),
            'description': 'Processes sequences and captures temporal dependencies',
            'purpose': 'Learns patterns in character sequences for next-character prediction',
            'gates': {
                'forget_gate': 'Decides what information to discard from cell state',
                'input_gate': 'Decides what new information to store in cell state', 
                'output_gate': 'Controls what parts of cell state to output',
                'candidate_values': 'New candidate values to add to cell state'
            }
        }
        
        # Output projection layer analysis
        output_proj = self.model.output_projection
        layer_info['output_projection'] = {
            'type': 'Linear',
            'input_dim': output_proj.in_features,
            'output_dim': output_proj.out_features,
            'parameters': output_proj.in_features * output_proj.out_features + output_proj.out_features,
            'memory_mb': ((output_proj.in_features * output_proj.out_features + output_proj.out_features) * 4) / (1024**2),
            'description': 'Projects LSTM output to vocabulary probabilities',
            'purpose': 'Converts hidden representations to character probability distributions'
        }
        
        return layer_info
    
    def _analyze_parameters(self) -> Dict[str, Any]:
        """
        Analyze parameter distribution and statistics.
        
        Returns:
            Dict[str, Any]: Parameter analysis results
        """
        param_stats = {}
        
        # Count parameters by type
        total_params = 0
        trainable_params = 0
        param_by_layer = {}
        
        for name, param in self.model.named_parameters():
            layer_name = name.split('.')[0]
            param_count = param.numel()
            
            if layer_name not in param_by_layer:
                param_by_layer[layer_name] = 0
            param_by_layer[layer_name] += param_count
            
            total_params += param_count
            if param.requires_grad:
                trainable_params += param_count
        
        # Calculate parameter distribution percentages
        param_distribution = {}
        for layer, count in param_by_layer.items():
            param_distribution[layer] = {
                'count': count,
                'percentage': (count / total_params) * 100
            }
        
        # Parameter efficiency metrics
        vocab_size = self.model.vocab_size
        embedding_dim = self.model.embedding_dim
        hidden_dim = self.model.hidden_dim
        
        # Calculate parameter efficiency ratios
        params_per_vocab_item = total_params / vocab_size
        embedding_efficiency = (vocab_size * embedding_dim) / total_params
        lstm_efficiency = param_by_layer.get('lstm', 0) / total_params
        
        param_stats = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params,
            'parameter_distribution': param_distribution,
            'efficiency_metrics': {
                'params_per_vocab_item': params_per_vocab_item,
                'embedding_efficiency': embedding_efficiency,
                'lstm_efficiency': lstm_efficiency,
                'model_complexity': self._calculate_model_complexity()
            },
            'size_comparisons': {
                'vs_gpt2_small': total_params / 124_000_000,  # GPT-2 small has ~124M params
                'vs_bert_base': total_params / 110_000_000,   # BERT base has ~110M params
                'classification': self._classify_model_size(total_params)
            }
        }
        
        return param_stats
    
    def _analyze_memory_usage(self) -> Dict[str, Any]:
        """
        Analyze memory usage and provide optimization suggestions.
        
        Returns:
            Dict[str, Any]: Memory analysis results
        """
        # Calculate model memory usage
        total_params = sum(p.numel() for p in self.model.parameters())
        
        # Memory usage estimates (in MB)
        # Assuming float32 (4 bytes per parameter)
        model_memory = (total_params * 4) / (1024**2)
        
        # Gradient memory (same as model for backprop)
        gradient_memory = model_memory
        
        # Optimizer memory (Adam uses 2x model params for momentum and variance)
        optimizer_memory = model_memory * 2
        
        # Activation memory estimation (depends on batch size and sequence length)
        def estimate_activation_memory(batch_size: int, seq_length: int) -> float:
            """Estimate activation memory for given batch and sequence size."""
            # Embedding activations
            embedding_mem = (batch_size * seq_length * self.model.embedding_dim * 4) / (1024**2)
            
            # LSTM activations (hidden states for each layer)
            lstm_mem = (batch_size * seq_length * self.model.hidden_dim * self.model.num_layers * 4) / (1024**2)
            
            # Output activations
            output_mem = (batch_size * seq_length * self.model.vocab_size * 4) / (1024**2)
            
            return embedding_mem + lstm_mem + output_mem
        
        # Memory usage for different scenarios
        memory_scenarios = {
            'small_batch': {
                'batch_size': 8,
                'seq_length': 50,
                'activation_memory': estimate_activation_memory(8, 50)
            },
            'medium_batch': {
                'batch_size': 32,
                'seq_length': 100,
                'activation_memory': estimate_activation_memory(32, 100)
            },
            'large_batch': {
                'batch_size': 64,
                'seq_length': 200,
                'activation_memory': estimate_activation_memory(64, 200)
            }
        }
        
        # Total memory estimates
        for scenario in memory_scenarios.values():
            scenario['total_memory'] = (model_memory + gradient_memory + 
                                      optimizer_memory + scenario['activation_memory'])
        
        # Memory optimization suggestions
        optimization_tips = self._generate_memory_optimization_tips(
            model_memory, total_params
        )
        
        return {
            'model_memory_mb': model_memory,
            'gradient_memory_mb': gradient_memory,
            'optimizer_memory_mb': optimizer_memory,
            'memory_scenarios': memory_scenarios,
            'optimization_tips': optimization_tips,
            'memory_breakdown': {
                'embedding': (self.model.vocab_size * self.model.embedding_dim * 4) / (1024**2),
                'lstm': (sum(p.numel() for p in self.model.lstm.parameters()) * 4) / (1024**2),
                'output': (self.model.hidden_dim * self.model.vocab_size * 4) / (1024**2)
            }
        }
    
    def _generate_architectural_insights(self) -> Dict[str, Any]:
        """
        Generate insights about the model architecture.
        
        Returns:
            Dict[str, Any]: Architectural insights and recommendations
        """
        insights = {
            'strengths': [],
            'potential_improvements': [],
            'design_analysis': {},
            'recommendations': []
        }
        
        # Analyze embedding dimension
        embed_ratio = self.model.embedding_dim / self.model.hidden_dim
        if embed_ratio < 0.5:
            insights['potential_improvements'].append(
                "Embedding dimension is much smaller than hidden dimension. "
                "Consider increasing embedding_dim for richer character representations."
            )
        elif embed_ratio > 1.5:
            insights['potential_improvements'].append(
                "Embedding dimension is much larger than hidden dimension. "
                "This may lead to information bottleneck in LSTM layers."
            )
        else:
            insights['strengths'].append(
                "Good balance between embedding and hidden dimensions."
            )
        
        # Analyze model depth
        if self.model.num_layers == 1:
            insights['potential_improvements'].append(
                "Single LSTM layer may limit model capacity. "
                "Consider adding more layers for complex patterns."
            )
        elif self.model.num_layers > 4:
            insights['potential_improvements'].append(
                "Many LSTM layers may cause vanishing gradients. "
                "Consider using residual connections or reducing depth."
            )
        else:
            insights['strengths'].append(
                f"Good depth with {self.model.num_layers} LSTM layers."
            )
        
        # Analyze vocabulary size impact
        vocab_ratio = self.model.vocab_size / self.model.embedding_dim
        insights['design_analysis']['vocab_embedding_ratio'] = {
            'ratio': vocab_ratio,
            'interpretation': (
                "High ratio suggests many characters per embedding dimension. "
                "Low ratio suggests rich representations per character."
            )
        }
        
        # Model complexity analysis
        total_params = sum(p.numel() for p in self.model.parameters())
        complexity_score = self._calculate_model_complexity()
        
        insights['design_analysis']['complexity'] = {
            'score': complexity_score,
            'interpretation': self._interpret_complexity_score(complexity_score),
            'parameter_efficiency': total_params / (self.model.vocab_size ** 2)
        }
        
        # Generate specific recommendations
        insights['recommendations'] = self._generate_architecture_recommendations()
        
        return insights
    
    def print_model_summary(self, detailed: bool = True) -> None:
        """
        Print a comprehensive model summary to console.
        
        This method provides a human-readable summary of the model architecture,
        parameters, and key metrics. It's designed to give a quick overview
        of the model's structure and characteristics.
        
        Args:
            detailed (bool): Whether to include detailed layer analysis
            
        The summary includes:
        - Model architecture overview
        - Parameter counts and distribution
        - Memory usage estimates
        - Performance characteristics
        - Optimization suggestions
        """
        summary = self.get_architecture_summary()
        
        print("=" * 80)
        print("MICRO LANGUAGE MODEL - ARCHITECTURE SUMMARY")
        print("=" * 80)
        
        # Basic information
        basic = summary['basic_info']
        print(f"Model Configuration:")
        print(f"  Vocabulary Size:    {basic['architecture']['vocab_size']:,}")
        print(f"  Embedding Dim:      {basic['architecture']['embedding_dim']:,}")
        print(f"  Hidden Dim:         {basic['architecture']['hidden_dim']:,}")
        print(f"  LSTM Layers:        {basic['architecture']['num_layers']}")
        print(f"  Dropout:            {basic['architecture']['dropout']}")
        print()
        
        # Parameter information
        params = summary['parameter_analysis']
        print(f"Parameter Analysis:")
        print(f"  Total Parameters:   {params['total_parameters']:,}")
        print(f"  Trainable Params:   {params['trainable_parameters']:,}")
        print(f"  Model Size:         {basic['model_size_mb']:.2f} MB")
        print()
        
        # Parameter distribution
        print(f"Parameter Distribution:")
        for layer, info in params['parameter_distribution'].items():
            print(f"  {layer.capitalize():15} {info['count']:>10,} ({info['percentage']:5.1f}%)")
        print()
        
        # Memory analysis
        memory = summary['memory_analysis']
        print(f"Memory Usage Estimates:")
        print(f"  Model Memory:       {memory['model_memory_mb']:.2f} MB")
        print(f"  Gradient Memory:    {memory['gradient_memory_mb']:.2f} MB")
        print(f"  Optimizer Memory:   {memory['optimizer_memory_mb']:.2f} MB")
        print()
        
        print(f"Training Memory (batch_size=32, seq_len=100):")
        medium_scenario = memory['memory_scenarios']['medium_batch']
        print(f"  Activation Memory:  {medium_scenario['activation_memory']:.2f} MB")
        print(f"  Total Memory:       {medium_scenario['total_memory']:.2f} MB")
        print()
        
        if detailed:
            # Layer details
            layers = summary['layer_details']
            print("Layer Details:")
            print("-" * 60)
            
            for layer_name, layer_info in layers.items():
                print(f"{layer_name.upper().replace('_', ' ')}:")
                print(f"  Type:        {layer_info['type']}")
                print(f"  Parameters:  {layer_info['parameters']:,}")
                print(f"  Memory:      {layer_info['memory_mb']:.2f} MB")
                print(f"  Purpose:     {layer_info['purpose']}")
                print()
            
            # Architectural insights
            insights = summary['architectural_insights']
            if insights['strengths']:
                print("Model Strengths:")
                for strength in insights['strengths']:
                    print(f"  ✓ {strength}")
                print()
            
            if insights['potential_improvements']:
                print("Potential Improvements:")
                for improvement in insights['potential_improvements']:
                    print(f"  → {improvement}")
                print()
        
        # Model classification
        size_class = params['size_comparisons']['classification']
        print(f"Model Classification: {size_class['category']} ({size_class['description']})")
        
        print("=" * 80)
    
    def get_parameter_statistics(self) -> Dict[str, Any]:
        """
        Get detailed statistics about model parameters.
        
        This method analyzes the distribution of parameter values across
        different layers, providing insights into model initialization,
        training progress, and potential issues like vanishing/exploding gradients.
        
        Returns:
            Dict[str, Any]: Parameter statistics including:
                - layer_stats: Statistics for each layer
                - overall_stats: Overall parameter statistics
                - gradient_stats: Gradient statistics (if available)
                - health_indicators: Model health indicators
        """
        stats = {
            'layer_stats': {},
            'overall_stats': {},
            'gradient_stats': {},
            'health_indicators': {}
        }
        
        all_params = []
        all_grads = []
        
        # Analyze each layer
        for name, param in self.model.named_parameters():
            param_data = param.data.cpu().numpy().flatten()
            all_params.extend(param_data)
            
            layer_stats = {
                'shape': list(param.shape),
                'total_params': param.numel(),
                'mean': float(np.mean(param_data)),
                'std': float(np.std(param_data)),
                'min': float(np.min(param_data)),
                'max': float(np.max(param_data)),
                'abs_mean': float(np.mean(np.abs(param_data))),
                'zero_fraction': float(np.mean(param_data == 0)),
                'small_fraction': float(np.mean(np.abs(param_data) < 1e-6))
            }
            
            # Add gradient statistics if available
            if param.grad is not None:
                grad_data = param.grad.data.cpu().numpy().flatten()
                all_grads.extend(grad_data)
                
                layer_stats['gradient'] = {
                    'mean': float(np.mean(grad_data)),
                    'std': float(np.std(grad_data)),
                    'min': float(np.min(grad_data)),
                    'max': float(np.max(grad_data)),
                    'abs_mean': float(np.mean(np.abs(grad_data))),
                    'norm': float(np.linalg.norm(grad_data))
                }
            
            stats['layer_stats'][name] = layer_stats
        
        # Overall parameter statistics
        all_params = np.array(all_params)
        stats['overall_stats'] = {
            'total_parameters': len(all_params),
            'mean': float(np.mean(all_params)),
            'std': float(np.std(all_params)),
            'min': float(np.min(all_params)),
            'max': float(np.max(all_params)),
            'abs_mean': float(np.mean(np.abs(all_params))),
            'zero_fraction': float(np.mean(all_params == 0)),
            'percentiles': {
                '1%': float(np.percentile(all_params, 1)),
                '5%': float(np.percentile(all_params, 5)),
                '25%': float(np.percentile(all_params, 25)),
                '50%': float(np.percentile(all_params, 50)),
                '75%': float(np.percentile(all_params, 75)),
                '95%': float(np.percentile(all_params, 95)),
                '99%': float(np.percentile(all_params, 99))
            }
        }
        
        # Overall gradient statistics
        if all_grads:
            all_grads = np.array(all_grads)
            stats['gradient_stats'] = {
                'mean': float(np.mean(all_grads)),
                'std': float(np.std(all_grads)),
                'min': float(np.min(all_grads)),
                'max': float(np.max(all_grads)),
                'abs_mean': float(np.mean(np.abs(all_grads))),
                'norm': float(np.linalg.norm(all_grads)),
                'percentiles': {
                    '1%': float(np.percentile(all_grads, 1)),
                    '5%': float(np.percentile(all_grads, 5)),
                    '25%': float(np.percentile(all_grads, 25)),
                    '50%': float(np.percentile(all_grads, 50)),
                    '75%': float(np.percentile(all_grads, 75)),
                    '95%': float(np.percentile(all_grads, 95)),
                    '99%': float(np.percentile(all_grads, 99))
                }
            }
        
        # Health indicators
        stats['health_indicators'] = self._calculate_health_indicators(stats)
        
        return stats
    
    def visualize_parameter_distribution(self, save_path: Optional[str] = None, 
                                       show_plot: bool = True) -> None:
        """
        Visualize the distribution of model parameters.
        
        Creates histograms and distribution plots for model parameters,
        helping to identify potential issues with initialization or training.
        
        Args:
            save_path (str, optional): Path to save the plot
            show_plot (bool): Whether to display the plot
            
        The visualization includes:
        - Overall parameter distribution histogram
        - Layer-wise parameter distributions
        - Gradient distributions (if available)
        - Statistical summaries and health indicators
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            raise ModelError(
                "matplotlib and seaborn are required for visualization. "
                "Install with: pip install matplotlib seaborn"
            )
        
        stats = self.get_parameter_statistics()
        
        # Set up the plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Parameter Distribution Analysis', fontsize=16, fontweight='bold')
        
        # Overall parameter distribution
        all_params = []
        for name, param in self.model.named_parameters():
            all_params.extend(param.data.cpu().numpy().flatten())
        
        axes[0, 0].hist(all_params, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_title('Overall Parameter Distribution')
        axes[0, 0].set_xlabel('Parameter Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(np.mean(all_params), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(all_params):.4f}')
        axes[0, 0].legend()
        
        # Layer-wise parameter means
        layer_names = []
        layer_means = []
        layer_stds = []
        
        for name, layer_stat in stats['layer_stats'].items():
            layer_names.append(name.replace('model.', '').replace('.weight', '').replace('.bias', ''))
            layer_means.append(layer_stat['abs_mean'])
            layer_stds.append(layer_stat['std'])
        
        x_pos = np.arange(len(layer_names))
        axes[0, 1].bar(x_pos, layer_means, alpha=0.7, color='green')
        axes[0, 1].set_title('Layer-wise Parameter Magnitudes')
        axes[0, 1].set_xlabel('Layer')
        axes[0, 1].set_ylabel('Mean Absolute Value')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(layer_names, rotation=45, ha='right')
        
        # Parameter standard deviations
        axes[1, 0].bar(x_pos, layer_stds, alpha=0.7, color='orange')
        axes[1, 0].set_title('Layer-wise Parameter Standard Deviations')
        axes[1, 0].set_xlabel('Layer')
        axes[1, 0].set_ylabel('Standard Deviation')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(layer_names, rotation=45, ha='right')
        
        # Gradient distribution (if available)
        if stats['gradient_stats']:
            all_grads = []
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    all_grads.extend(param.grad.data.cpu().numpy().flatten())
            
            if all_grads:
                axes[1, 1].hist(all_grads, bins=50, alpha=0.7, color='purple', edgecolor='black')
                axes[1, 1].set_title('Gradient Distribution')
                axes[1, 1].set_xlabel('Gradient Value')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].axvline(np.mean(all_grads), color='red', linestyle='--',
                                  label=f'Mean: {np.mean(all_grads):.6f}')
                axes[1, 1].legend()
            else:
                axes[1, 1].text(0.5, 0.5, 'No gradients available\n(Run backward pass first)', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Gradient Distribution')
        else:
            axes[1, 1].text(0.5, 0.5, 'No gradients available\n(Run backward pass first)', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Gradient Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Parameter distribution plot saved to: {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def inspect_activations(self, input_data: torch.Tensor, 
                          layer_names: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """
        Inspect intermediate activations for given input.
        
        This method captures and returns the intermediate activations from
        specified layers, helping to understand how information flows through
        the model and identify potential issues like vanishing activations.
        
        Args:
            input_data (torch.Tensor): Input tensor to process
            layer_names (List[str], optional): Specific layers to inspect.
                                             If None, inspects all major layers.
        
        Returns:
            Dict[str, torch.Tensor]: Dictionary mapping layer names to activations
            
        The activations can be used to:
        - Visualize how information flows through the model
        - Identify layers with vanishing or exploding activations
        - Debug model behavior on specific inputs
        - Analyze representation learning
        """
        if layer_names is None:
            layer_names = ['embedding', 'lstm', 'output_projection']
        
        # Clear previous activations
        self._activations.clear()
        
        # Register hooks for specified layers
        hooks = []
        for name in layer_names:
            if hasattr(self.model, name):
                layer = getattr(self.model, name)
                
                def create_hook(layer_name):
                    def hook(module, input, output):
                        # Handle different output types
                        if isinstance(output, tuple):
                            # For LSTM, take the first element (actual output)
                            activation = output[0].detach().clone()
                        else:
                            activation = output.detach().clone()
                        self._activations[layer_name] = activation
                    return hook
                
                hook = layer.register_forward_hook(create_hook(name))
                hooks.append(hook)
        
        try:
            # Forward pass to capture activations
            with torch.no_grad():
                self.model.eval()
                _ = self.model(input_data)
            
            # Analyze captured activations
            activation_analysis = {}
            for name, activation in self._activations.items():
                activation_analysis[name] = {
                    'tensor': activation,
                    'shape': list(activation.shape),
                    'mean': float(activation.mean()),
                    'std': float(activation.std()),
                    'min': float(activation.min()),
                    'max': float(activation.max()),
                    'zero_fraction': float((activation == 0).float().mean()),
                    'activation_norm': float(torch.norm(activation))
                }
            
            return activation_analysis
            
        finally:
            # Clean up hooks
            for hook in hooks:
                hook.remove()
    
    def visualize_activations(self, input_data: torch.Tensor, 
                            save_path: Optional[str] = None,
                            show_plot: bool = True) -> None:
        """
        Visualize activation patterns for given input.
        
        Creates visualizations of intermediate activations to help understand
        how the model processes information and identify potential issues.
        
        Args:
            input_data (torch.Tensor): Input tensor to analyze
            save_path (str, optional): Path to save the plot
            show_plot (bool): Whether to display the plot
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            raise ModelError(
                "matplotlib and seaborn are required for visualization. "
                "Install with: pip install matplotlib seaborn"
            )
        
        activations = self.inspect_activations(input_data)
        
        # Set up the plot
        num_layers = len(activations)
        fig, axes = plt.subplots(2, num_layers, figsize=(5*num_layers, 10))
        if num_layers == 1:
            axes = axes.reshape(2, 1)
        
        fig.suptitle('Activation Analysis', fontsize=16, fontweight='bold')
        
        for idx, (layer_name, activation_info) in enumerate(activations.items()):
            activation = activation_info['tensor']
            
            # Flatten activation for histogram
            flat_activation = activation.cpu().numpy().flatten()
            
            # Activation distribution
            axes[0, idx].hist(flat_activation, bins=50, alpha=0.7, color='blue', edgecolor='black')
            axes[0, idx].set_title(f'{layer_name.capitalize()} Activation Distribution')
            axes[0, idx].set_xlabel('Activation Value')
            axes[0, idx].set_ylabel('Frequency')
            axes[0, idx].axvline(activation_info['mean'], color='red', linestyle='--',
                               label=f"Mean: {activation_info['mean']:.4f}")
            axes[0, idx].legend()
            
            # Activation heatmap (for 2D or 3D tensors)
            if len(activation.shape) >= 2:
                # Take first batch and average over sequence dimension if 3D
                if len(activation.shape) == 3:
                    heatmap_data = activation[0].mean(dim=0).cpu().numpy()
                else:
                    heatmap_data = activation[0].cpu().numpy()
                
                # Reshape if 1D
                if len(heatmap_data.shape) == 1:
                    heatmap_data = heatmap_data.reshape(1, -1)
                
                im = axes[1, idx].imshow(heatmap_data, cmap='viridis', aspect='auto')
                axes[1, idx].set_title(f'{layer_name.capitalize()} Activation Heatmap')
                axes[1, idx].set_xlabel('Feature Dimension')
                axes[1, idx].set_ylabel('Sequence Position' if len(activation.shape) == 3 else 'Batch')
                plt.colorbar(im, ax=axes[1, idx])
            else:
                axes[1, idx].text(0.5, 0.5, f'Shape: {activation.shape}\nCannot visualize 1D tensor',
                                 ha='center', va='center', transform=axes[1, idx].transAxes)
                axes[1, idx].set_title(f'{layer_name.capitalize()} Activation Info')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Activation visualization saved to: {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def profile_model_performance(self, input_data: torch.Tensor, 
                                num_runs: int = 100) -> Dict[str, Any]:
        """
        Profile model performance and identify bottlenecks.
        
        This method measures the execution time of different model components
        to identify performance bottlenecks and optimization opportunities.
        
        Args:
            input_data (torch.Tensor): Input tensor for profiling
            num_runs (int): Number of runs for averaging (default: 100)
        
        Returns:
            Dict[str, Any]: Performance profiling results including:
                - layer_times: Execution time for each layer
                - total_time: Total forward pass time
                - throughput: Tokens processed per second
                - memory_usage: Memory usage during execution
                - bottlenecks: Identified performance bottlenecks
        """
        self.model.eval()
        
        # Warm up
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(input_data)
        
        # Profile layer-wise execution times
        layer_times = {}
        
        def time_hook(name):
            def hook(module, input, output):
                torch.cuda.synchronize() if self.device.type == 'cuda' else None
                layer_times[name] = time.time()
            return hook
        
        # Register timing hooks
        hooks = []
        start_times = {}
        
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                hook = module.register_forward_hook(time_hook(f"{name}_end"))
                hooks.append(hook)
                
                pre_hook = module.register_forward_pre_hook(
                    lambda module, input, module_name=name: 
                    start_times.update({module_name: time.time()})
                )
                hooks.append(pre_hook)
        
        try:
            # Time multiple runs
            total_times = []
            
            for _ in range(num_runs):
                start_times.clear()
                layer_times.clear()
                
                torch.cuda.synchronize() if self.device.type == 'cuda' else None
                start_time = time.time()
                
                with torch.no_grad():
                    _ = self.model(input_data)
                
                torch.cuda.synchronize() if self.device.type == 'cuda' else None
                end_time = time.time()
                
                total_times.append(end_time - start_time)
            
            # Calculate statistics
            avg_total_time = np.mean(total_times)
            std_total_time = np.std(total_times)
            
            # Calculate throughput
            batch_size, seq_length = input_data.shape
            tokens_per_second = (batch_size * seq_length) / avg_total_time
            
            # Memory usage
            if self.device.type == 'cuda':
                memory_allocated = torch.cuda.memory_allocated(self.device) / (1024**2)  # MB
                memory_reserved = torch.cuda.memory_reserved(self.device) / (1024**2)   # MB
            else:
                memory_allocated = 0
                memory_reserved = 0
            
            # Identify bottlenecks
            bottlenecks = self._identify_bottlenecks(layer_times, avg_total_time)
            
            return {
                'total_time': {
                    'mean': avg_total_time,
                    'std': std_total_time,
                    'min': min(total_times),
                    'max': max(total_times)
                },
                'throughput': {
                    'tokens_per_second': tokens_per_second,
                    'sequences_per_second': batch_size / avg_total_time,
                    'batch_size': batch_size,
                    'sequence_length': seq_length
                },
                'memory_usage': {
                    'allocated_mb': memory_allocated,
                    'reserved_mb': memory_reserved,
                    'device': str(self.device)
                },
                'bottlenecks': bottlenecks,
                'optimization_suggestions': self._generate_performance_suggestions(
                    avg_total_time, tokens_per_second, memory_allocated
                )
            }
            
        finally:
            # Clean up hooks
            for hook in hooks:
                hook.remove()
    
    def _calculate_model_complexity(self) -> float:
        """Calculate a complexity score for the model."""
        total_params = sum(p.numel() for p in self.model.parameters())
        vocab_size = self.model.vocab_size
        hidden_dim = self.model.hidden_dim
        num_layers = self.model.num_layers
        
        # Complexity score based on parameters, depth, and vocabulary
        complexity = (total_params / 1000) * (num_layers / 2) * (vocab_size / 100)
        return complexity
    
    def _interpret_complexity_score(self, score: float) -> str:
        """Interpret the complexity score."""
        if score < 10:
            return "Very simple model, suitable for basic tasks and learning"
        elif score < 100:
            return "Simple model, good for educational purposes and small datasets"
        elif score < 1000:
            return "Moderate complexity, suitable for medium-scale tasks"
        elif score < 10000:
            return "Complex model, requires significant computational resources"
        else:
            return "Very complex model, suitable for large-scale applications"
    
    def _classify_model_size(self, total_params: int) -> Dict[str, str]:
        """Classify model size based on parameter count."""
        if total_params < 1_000:
            return {
                'category': 'Tiny',
                'description': 'Suitable for learning and experimentation'
            }
        elif total_params < 10_000:
            return {
                'category': 'Very Small',
                'description': 'Good for simple tasks and educational purposes'
            }
        elif total_params < 100_000:
            return {
                'category': 'Small',
                'description': 'Suitable for basic language modeling tasks'
            }
        elif total_params < 1_000_000:
            return {
                'category': 'Medium',
                'description': 'Good balance of capability and efficiency'
            }
        elif total_params < 10_000_000:
            return {
                'category': 'Large',
                'description': 'Capable model requiring moderate resources'
            }
        else:
            return {
                'category': 'Very Large',
                'description': 'High-capacity model requiring significant resources'
            }
    
    def _generate_memory_optimization_tips(self, model_memory: float, 
                                         total_params: int) -> List[str]:
        """Generate memory optimization suggestions."""
        tips = []
        
        if model_memory > 100:  # > 100 MB
            tips.append("Consider reducing hidden_dim or num_layers to decrease memory usage")
        
        if self.model.vocab_size > 1000:
            tips.append("Large vocabulary increases embedding memory. Consider subword tokenization")
        
        if self.model.embedding_dim > self.model.hidden_dim:
            tips.append("Embedding dimension larger than hidden dimension may be inefficient")
        
        tips.extend([
            "Use gradient checkpointing for very long sequences",
            "Consider mixed precision training (fp16) to reduce memory usage",
            "Reduce batch size if running out of memory during training"
        ])
        
        return tips
    
    def _generate_architecture_recommendations(self) -> List[str]:
        """Generate architecture improvement recommendations."""
        recommendations = []
        
        # Embedding dimension recommendations
        embed_hidden_ratio = self.model.embedding_dim / self.model.hidden_dim
        if embed_hidden_ratio < 0.3:
            recommendations.append(
                f"Consider increasing embedding_dim (currently {self.model.embedding_dim}) "
                f"to better match hidden_dim ({self.model.hidden_dim})"
            )
        
        # Layer depth recommendations
        if self.model.num_layers == 1:
            recommendations.append("Consider adding more LSTM layers for increased model capacity")
        elif self.model.num_layers > 3:
            recommendations.append("Consider adding residual connections for deep LSTM networks")
        
        # Dropout recommendations
        if self.model.dropout == 0:
            recommendations.append("Consider adding dropout for regularization")
        elif self.model.dropout > 0.5:
            recommendations.append("High dropout may hurt model capacity. Consider reducing it")
        
        return recommendations
    
    def _calculate_health_indicators(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate model health indicators from parameter statistics."""
        indicators = {}
        
        # Parameter magnitude health
        overall_stats = stats['overall_stats']
        abs_mean = overall_stats['abs_mean']
        
        if abs_mean < 1e-6:
            indicators['parameter_magnitude'] = {
                'status': 'warning',
                'message': 'Very small parameter magnitudes may indicate vanishing gradients'
            }
        elif abs_mean > 10:
            indicators['parameter_magnitude'] = {
                'status': 'warning', 
                'message': 'Large parameter magnitudes may indicate exploding gradients'
            }
        else:
            indicators['parameter_magnitude'] = {
                'status': 'healthy',
                'message': 'Parameter magnitudes are in healthy range'
            }
        
        # Gradient health (if available)
        if stats['gradient_stats']:
            grad_stats = stats['gradient_stats']
            grad_norm = grad_stats['norm']
            
            if grad_norm < 1e-6:
                indicators['gradient_health'] = {
                    'status': 'warning',
                    'message': 'Very small gradients may indicate vanishing gradient problem'
                }
            elif grad_norm > 100:
                indicators['gradient_health'] = {
                    'status': 'warning',
                    'message': 'Large gradients may indicate exploding gradient problem'
                }
            else:
                indicators['gradient_health'] = {
                    'status': 'healthy',
                    'message': 'Gradient magnitudes are in healthy range'
                }
        
        # Zero parameter fraction
        zero_fraction = overall_stats['zero_fraction']
        if zero_fraction > 0.1:
            indicators['sparsity'] = {
                'status': 'info',
                'message': f'High zero fraction ({zero_fraction:.1%}) indicates sparse parameters'
            }
        
        return indicators
    
    def _identify_bottlenecks(self, layer_times: Dict[str, float], 
                            total_time: float) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks from timing data."""
        bottlenecks = []
        
        # This is a simplified bottleneck analysis
        # In practice, you'd need more sophisticated profiling
        if total_time > 0.1:  # > 100ms for forward pass
            bottlenecks.append({
                'type': 'slow_forward_pass',
                'severity': 'medium',
                'description': f'Forward pass takes {total_time*1000:.1f}ms, consider model optimization'
            })
        
        return bottlenecks
    
    def _generate_performance_suggestions(self, avg_time: float, 
                                        tokens_per_sec: float,
                                        memory_mb: float) -> List[str]:
        """Generate performance optimization suggestions."""
        suggestions = []
        
        if tokens_per_sec < 1000:
            suggestions.append("Low throughput. Consider reducing model size or using GPU acceleration")
        
        if memory_mb > 1000:  # > 1GB
            suggestions.append("High memory usage. Consider reducing batch size or model dimensions")
        
        if avg_time > 0.1:  # > 100ms
            suggestions.append("Slow inference. Consider model quantization or pruning")
        
        suggestions.extend([
            "Use torch.jit.script() for faster inference",
            "Consider ONNX export for deployment optimization",
            "Profile with torch.profiler for detailed bottleneck analysis"
        ])
        
        return suggestions


class TrainingVisualizer:
    """
    Visualization utilities for training progress and metrics.
    
    This class provides comprehensive visualization tools for monitoring
    and analyzing the training process of language models. It helps
    identify training issues, track progress, and optimize hyperparameters.
    
    Args:
        training_history (Dict[str, List[float]]): Training history from ModelTrainer
        
    Example:
        >>> trainer = ModelTrainer(model, tokenizer)
        >>> history = trainer.train(data_loader, epochs=50)
        >>> visualizer = TrainingVisualizer(history)
        >>> visualizer.plot_training_progress()
    """
    
    def __init__(self, training_history: Optional[Dict[str, List[float]]] = None):
        """
        Initialize the training visualizer.
        
        Args:
            training_history (Dict[str, List[float]], optional): Training history data
        """
        self.training_history = training_history or {}
    
    def update_history(self, new_history: Dict[str, List[float]]) -> None:
        """
        Update the training history with new data.
        
        Args:
            new_history (Dict[str, List[float]]): New training history data
        """
        self.training_history = new_history
    
    def plot_training_progress(self, save_path: Optional[str] = None,
                             show_plot: bool = True, figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Plot comprehensive training progress visualization.
        
        Creates a multi-panel plot showing various aspects of training progress
        including loss curves, learning rate schedules, and training metrics.
        
        Args:
            save_path (str, optional): Path to save the plot
            show_plot (bool): Whether to display the plot
            figsize (Tuple[int, int]): Figure size (width, height)
        """
        if not self.training_history:
            raise ModelError("No training history available. Train the model first.")
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            raise ModelError(
                "matplotlib and seaborn are required for visualization. "
                "Install with: pip install matplotlib seaborn"
            )
        
        # Set up the plot
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Training Progress Analysis', fontsize=16, fontweight='bold')
        
        epochs = range(1, len(self.training_history['train_loss']) + 1)
        
        # Loss curve
        axes[0, 0].plot(epochs, self.training_history['train_loss'], 'b-', linewidth=2, label='Training Loss')
        axes[0, 0].set_title('Training Loss Over Time')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Add trend line
        if len(epochs) > 1:
            z = np.polyfit(epochs, self.training_history['train_loss'], 1)
            p = np.poly1d(z)
            axes[0, 0].plot(epochs, p(epochs), "r--", alpha=0.8, label='Trend')
            axes[0, 0].legend()
        
        # Learning rate schedule (if available)
        if 'learning_rates' in self.training_history:
            axes[0, 1].plot(epochs, self.training_history['learning_rates'], 'g-', linewidth=2)
            axes[0, 1].set_title('Learning Rate Schedule')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Learning Rate')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_yscale('log')
        else:
            axes[0, 1].text(0.5, 0.5, 'Learning rate data\nnot available', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Learning Rate Schedule')
        
        # Training time per epoch
        if 'epoch_times' in self.training_history:
            axes[1, 0].plot(epochs, self.training_history['epoch_times'], 'orange', linewidth=2)
            axes[1, 0].set_title('Training Time per Epoch')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Time (seconds)')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Add average line
            avg_time = np.mean(self.training_history['epoch_times'])
            axes[1, 0].axhline(y=avg_time, color='red', linestyle='--', 
                              label=f'Average: {avg_time:.2f}s')
            axes[1, 0].legend()
        else:
            axes[1, 0].text(0.5, 0.5, 'Timing data\nnot available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Training Time per Epoch')
        
        # Loss improvement analysis
        if len(self.training_history['train_loss']) > 1:
            loss_improvements = []
            for i in range(1, len(self.training_history['train_loss'])):
                improvement = self.training_history['train_loss'][i-1] - self.training_history['train_loss'][i]
                loss_improvements.append(improvement)
            
            axes[1, 1].plot(range(2, len(epochs) + 1), loss_improvements, 'purple', linewidth=2)
            axes[1, 1].set_title('Loss Improvement per Epoch')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss Improvement')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        else:
            axes[1, 1].text(0.5, 0.5, 'Insufficient data\nfor improvement analysis', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Loss Improvement per Epoch')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training progress plot saved to: {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def analyze_training_dynamics(self) -> Dict[str, Any]:
        """
        Analyze training dynamics and provide insights.
        
        Returns:
            Dict[str, Any]: Analysis results including:
                - convergence_analysis: Information about training convergence
                - stability_metrics: Training stability indicators
                - efficiency_metrics: Training efficiency analysis
                - recommendations: Suggestions for improvement
        """
        if not self.training_history or 'train_loss' not in self.training_history:
            raise ModelError("No training loss data available for analysis.")
        
        losses = np.array(self.training_history['train_loss'])
        
        # Convergence analysis
        convergence_analysis = self._analyze_convergence(losses)
        
        # Stability metrics
        stability_metrics = self._analyze_stability(losses)
        
        # Efficiency metrics
        efficiency_metrics = self._analyze_efficiency()
        
        # Generate recommendations
        recommendations = self._generate_training_recommendations(
            convergence_analysis, stability_metrics, efficiency_metrics
        )
        
        return {
            'convergence_analysis': convergence_analysis,
            'stability_metrics': stability_metrics,
            'efficiency_metrics': efficiency_metrics,
            'recommendations': recommendations
        }
    
    def _analyze_convergence(self, losses: np.ndarray) -> Dict[str, Any]:
        """Analyze training convergence patterns."""
        if len(losses) < 2:
            return {'status': 'insufficient_data'}
        
        # Calculate improvement metrics
        total_improvement = losses[0] - losses[-1]
        relative_improvement = total_improvement / losses[0] if losses[0] != 0 else 0
        
        # Check for convergence plateau
        recent_window = min(10, len(losses) // 4)
        if len(losses) > recent_window:
            recent_losses = losses[-recent_window:]
            recent_std = np.std(recent_losses)
            plateau_threshold = 0.001  # Adjust based on typical loss scale
            
            is_plateaued = recent_std < plateau_threshold
        else:
            is_plateaued = False
        
        # Convergence rate (exponential decay fit)
        epochs = np.arange(len(losses))
        try:
            # Fit exponential decay: loss = a * exp(-b * epoch) + c
            from scipy.optimize import curve_fit
            
            def exp_decay(x, a, b, c):
                return a * np.exp(-b * x) + c
            
            popt, _ = curve_fit(exp_decay, epochs, losses, maxfev=1000)
            convergence_rate = popt[1]  # decay constant
        except:
            # Fallback to linear fit
            convergence_rate = -np.polyfit(epochs, losses, 1)[0]
        
        return {
            'total_improvement': float(total_improvement),
            'relative_improvement': float(relative_improvement),
            'convergence_rate': float(convergence_rate),
            'is_plateaued': is_plateaued,
            'final_loss': float(losses[-1]),
            'best_loss': float(np.min(losses)),
            'epochs_to_best': int(np.argmin(losses)) + 1
        }
    
    def _analyze_stability(self, losses: np.ndarray) -> Dict[str, Any]:
        """Analyze training stability."""
        if len(losses) < 3:
            return {'status': 'insufficient_data'}
        
        # Calculate loss volatility
        loss_changes = np.diff(losses)
        volatility = np.std(loss_changes)
        
        # Count oscillations (sign changes in loss difference)
        sign_changes = np.sum(np.diff(np.sign(loss_changes)) != 0)
        oscillation_rate = sign_changes / len(loss_changes) if len(loss_changes) > 0 else 0
        
        # Identify loss spikes (sudden increases)
        spike_threshold = np.mean(losses) + 2 * np.std(losses)
        spikes = np.sum(losses > spike_threshold)
        
        # Monotonicity check (generally decreasing)
        monotonic_decreases = np.sum(loss_changes < 0)
        monotonicity = monotonic_decreases / len(loss_changes) if len(loss_changes) > 0 else 0
        
        return {
            'volatility': float(volatility),
            'oscillation_rate': float(oscillation_rate),
            'spike_count': int(spikes),
            'monotonicity': float(monotonicity),
            'stability_score': float(1.0 - oscillation_rate)  # Higher is more stable
        }
    
    def _analyze_efficiency(self) -> Dict[str, Any]:
        """Analyze training efficiency."""
        efficiency = {}
        
        # Time efficiency
        if 'epoch_times' in self.training_history:
            times = np.array(self.training_history['epoch_times'])
            efficiency['time_metrics'] = {
                'mean_epoch_time': float(np.mean(times)),
                'total_training_time': float(np.sum(times)),
                'time_stability': float(1.0 / (1.0 + np.std(times) / np.mean(times)))
            }
        
        # Loss efficiency (improvement per epoch)
        if 'train_loss' in self.training_history:
            losses = np.array(self.training_history['train_loss'])
            if len(losses) > 1:
                total_improvement = losses[0] - losses[-1]
                improvement_per_epoch = total_improvement / len(losses)
                efficiency['loss_efficiency'] = {
                    'improvement_per_epoch': float(improvement_per_epoch),
                    'efficiency_score': float(improvement_per_epoch * len(losses))
                }
        
        return efficiency
    
    def _generate_training_recommendations(self, convergence: Dict[str, Any],
                                         stability: Dict[str, Any],
                                         efficiency: Dict[str, Any]) -> List[str]:
        """Generate training improvement recommendations."""
        recommendations = []
        
        # Convergence recommendations
        if convergence.get('is_plateaued', False):
            recommendations.append("Training has plateaued. Consider reducing learning rate or early stopping.")
        
        if convergence.get('relative_improvement', 0) < 0.1:
            recommendations.append("Low improvement rate. Consider increasing learning rate or model capacity.")
        
        # Stability recommendations
        if stability.get('oscillation_rate', 0) > 0.3:
            recommendations.append("High loss oscillation. Consider reducing learning rate for more stable training.")
        
        if stability.get('spike_count', 0) > 0:
            recommendations.append("Loss spikes detected. Consider gradient clipping or learning rate scheduling.")
        
        # Efficiency recommendations
        if 'time_metrics' in efficiency:
            if efficiency['time_metrics'].get('time_stability', 1.0) < 0.8:
                recommendations.append("Inconsistent epoch times. Check for system resource issues.")
        
        if not recommendations:
            recommendations.append("Training appears healthy. Continue monitoring progress.")
        
        return recommendations


# Utility functions for easy access
def inspect_model(model: MicroLM, detailed: bool = True) -> Dict[str, Any]:
    """
    Quick model inspection utility.
    
    Args:
        model (MicroLM): Model to inspect
        detailed (bool): Whether to show detailed analysis
        
    Returns:
        Dict[str, Any]: Model inspection results
    """
    inspector = ModelInspector(model)
    if detailed:
        inspector.print_model_summary(detailed=detailed)
    return inspector.get_architecture_summary()


def visualize_training(training_history: Dict[str, List[float]], 
                      save_path: Optional[str] = None) -> None:
    """
    Quick training visualization utility.
    
    Args:
        training_history (Dict[str, List[float]]): Training history from trainer
        save_path (str, optional): Path to save the plot
    """
    visualizer = TrainingVisualizer(training_history)
    visualizer.plot_training_progress(save_path=save_path)


def analyze_parameters(model: MicroLM, save_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Quick parameter analysis utility.
    
    Args:
        model (MicroLM): Model to analyze
        save_path (str, optional): Path to save visualization
        
    Returns:
        Dict[str, Any]: Parameter statistics
    """
    inspector = ModelInspector(model)
    stats = inspector.get_parameter_statistics()
    
    if save_path:
        inspector.visualize_parameter_distribution(save_path=save_path, show_plot=False)
    
    return stats