"""
Tests for model inspection and visualization utilities.

This module tests the comprehensive inspection and visualization tools
for MicroLSTM, ensuring they work correctly and provide
accurate analysis of model architecture, parameters, and training progress.
"""

import pytest
import torch
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from micro_lm.model import MicroLM
from micro_lm.tokenizer import CharacterTokenizer
from micro_lm.trainer import ModelTrainer
from micro_lm.inspection import (
    ModelInspector, TrainingVisualizer, inspect_model, 
    visualize_training, analyze_parameters
)
from micro_lm.exceptions import ModelError, ModelConfigurationError


class TestModelInspector:
    """Test cases for ModelInspector class."""
    
    @pytest.fixture
    def sample_model(self):
        """Create a sample model for testing."""
        return MicroLM(
            vocab_size=50,
            embedding_dim=32,
            hidden_dim=64,
            num_layers=2,
            dropout=0.1
        )
    
    @pytest.fixture
    def inspector(self, sample_model):
        """Create a ModelInspector instance."""
        return ModelInspector(sample_model)
    
    def test_inspector_initialization(self, sample_model):
        """Test ModelInspector initialization."""
        inspector = ModelInspector(sample_model)
        assert inspector.model is sample_model
        assert inspector.device == next(sample_model.parameters()).device
        assert inspector._activation_hooks == {}
        assert inspector._activations == {}
    
    def test_inspector_initialization_invalid_model(self):
        """Test ModelInspector initialization with invalid model."""
        with pytest.raises(ModelConfigurationError) as exc_info:
            ModelInspector("not_a_model")
        
        assert "model must be MicroLM instance" in str(exc_info.value)
    
    def test_get_architecture_summary(self, inspector):
        """Test comprehensive architecture summary generation."""
        summary = inspector.get_architecture_summary()
        
        # Check main sections
        assert 'basic_info' in summary
        assert 'layer_details' in summary
        assert 'parameter_analysis' in summary
        assert 'memory_analysis' in summary
        assert 'architectural_insights' in summary
        assert 'generation_timestamp' in summary
        
        # Check basic info structure
        basic_info = summary['basic_info']
        assert 'architecture' in basic_info
        assert 'parameters' in basic_info
        assert 'model_size_mb' in basic_info
        
        # Check layer details
        layer_details = summary['layer_details']
        assert 'embedding' in layer_details
        assert 'lstm' in layer_details
        assert 'output_projection' in layer_details
        
        # Verify embedding layer info
        embedding_info = layer_details['embedding']
        assert embedding_info['type'] == 'Embedding'
        assert embedding_info['input_dim'] == 50
        assert embedding_info['output_dim'] == 32
        assert embedding_info['parameters'] == 50 * 32
        assert 'description' in embedding_info
        assert 'purpose' in embedding_info
        
        # Verify LSTM layer info
        lstm_info = layer_details['lstm']
        assert lstm_info['type'] == 'LSTM'
        assert lstm_info['input_size'] == 32
        assert lstm_info['hidden_size'] == 64
        assert lstm_info['num_layers'] == 2
        assert 'gates' in lstm_info
        assert len(lstm_info['gates']) == 4  # forget, input, output, candidate
        
        # Verify output projection info
        output_info = layer_details['output_projection']
        assert output_info['type'] == 'Linear'
        assert output_info['input_dim'] == 64
        assert output_info['output_dim'] == 50
    
    def test_parameter_analysis(self, inspector):
        """Test parameter analysis functionality."""
        summary = inspector.get_architecture_summary()
        param_analysis = summary['parameter_analysis']
        
        # Check parameter counts
        assert 'total_parameters' in param_analysis
        assert 'trainable_parameters' in param_analysis
        assert 'parameter_distribution' in param_analysis
        assert 'efficiency_metrics' in param_analysis
        assert 'size_comparisons' in param_analysis
        
        # Verify parameter distribution
        distribution = param_analysis['parameter_distribution']
        assert 'embedding' in distribution
        assert 'lstm' in distribution
        assert 'output_projection' in distribution
        
        # Check that percentages sum to approximately 100%
        total_percentage = sum(layer['percentage'] for layer in distribution.values())
        assert abs(total_percentage - 100.0) < 0.1
        
        # Check efficiency metrics
        efficiency = param_analysis['efficiency_metrics']
        assert 'params_per_vocab_item' in efficiency
        assert 'embedding_efficiency' in efficiency
        assert 'lstm_efficiency' in efficiency
        assert 'model_complexity' in efficiency
        
        # Check size comparisons
        size_comp = param_analysis['size_comparisons']
        assert 'vs_gpt2_small' in size_comp
        assert 'vs_bert_base' in size_comp
        assert 'classification' in size_comp
        
        classification = size_comp['classification']
        assert 'category' in classification
        assert 'description' in classification
    
    def test_memory_analysis(self, inspector):
        """Test memory usage analysis."""
        summary = inspector.get_architecture_summary()
        memory_analysis = summary['memory_analysis']
        
        # Check memory metrics
        assert 'model_memory_mb' in memory_analysis
        assert 'gradient_memory_mb' in memory_analysis
        assert 'optimizer_memory_mb' in memory_analysis
        assert 'memory_scenarios' in memory_analysis
        assert 'optimization_tips' in memory_analysis
        assert 'memory_breakdown' in memory_analysis
        
        # Check memory scenarios
        scenarios = memory_analysis['memory_scenarios']
        assert 'small_batch' in scenarios
        assert 'medium_batch' in scenarios
        assert 'large_batch' in scenarios
        
        for scenario in scenarios.values():
            assert 'batch_size' in scenario
            assert 'seq_length' in scenario
            assert 'activation_memory' in scenario
            assert 'total_memory' in scenario
        
        # Check memory breakdown
        breakdown = memory_analysis['memory_breakdown']
        assert 'embedding' in breakdown
        assert 'lstm' in breakdown
        assert 'output' in breakdown
        
        # Verify optimization tips are provided
        tips = memory_analysis['optimization_tips']
        assert isinstance(tips, list)
        assert len(tips) > 0
    
    def test_architectural_insights(self, inspector):
        """Test architectural insights generation."""
        summary = inspector.get_architecture_summary()
        insights = summary['architectural_insights']
        
        # Check insight categories
        assert 'strengths' in insights
        assert 'potential_improvements' in insights
        assert 'design_analysis' in insights
        assert 'recommendations' in insights
        
        # Check that insights are lists/dicts
        assert isinstance(insights['strengths'], list)
        assert isinstance(insights['potential_improvements'], list)
        assert isinstance(insights['design_analysis'], dict)
        assert isinstance(insights['recommendations'], list)
        
        # Check design analysis components
        design_analysis = insights['design_analysis']
        assert 'vocab_embedding_ratio' in design_analysis
        assert 'complexity' in design_analysis
        
        vocab_ratio = design_analysis['vocab_embedding_ratio']
        assert 'ratio' in vocab_ratio
        assert 'interpretation' in vocab_ratio
        
        complexity = design_analysis['complexity']
        assert 'score' in complexity
        assert 'interpretation' in complexity
        assert 'parameter_efficiency' in complexity
    
    def test_print_model_summary(self, inspector, capsys):
        """Test model summary printing."""
        # Test basic summary
        inspector.print_model_summary(detailed=False)
        captured = capsys.readouterr()
        
        assert "MICROLSTM - ARCHITECTURE SUMMARY" in captured.out
        assert "Model Configuration:" in captured.out
        assert "Parameter Analysis:" in captured.out
        assert "Memory Usage Estimates:" in captured.out
        assert "Model Classification:" in captured.out
        
        # Test detailed summary
        inspector.print_model_summary(detailed=True)
        captured = capsys.readouterr()
        
        assert "Layer Details:" in captured.out
        assert "EMBEDDING:" in captured.out
        assert "LSTM:" in captured.out
        assert "OUTPUT PROJECTION:" in captured.out
    
    def test_get_parameter_statistics(self, inspector):
        """Test parameter statistics calculation."""
        stats = inspector.get_parameter_statistics()
        
        # Check main sections
        assert 'layer_stats' in stats
        assert 'overall_stats' in stats
        assert 'gradient_stats' in stats
        assert 'health_indicators' in stats
        
        # Check layer statistics
        layer_stats = stats['layer_stats']
        assert len(layer_stats) > 0
        
        for layer_name, layer_stat in layer_stats.items():
            assert 'shape' in layer_stat
            assert 'total_params' in layer_stat
            assert 'mean' in layer_stat
            assert 'std' in layer_stat
            assert 'min' in layer_stat
            assert 'max' in layer_stat
            assert 'abs_mean' in layer_stat
            assert 'zero_fraction' in layer_stat
            assert 'small_fraction' in layer_stat
        
        # Check overall statistics
        overall_stats = stats['overall_stats']
        assert 'total_parameters' in overall_stats
        assert 'mean' in overall_stats
        assert 'std' in overall_stats
        assert 'percentiles' in overall_stats
        
        percentiles = overall_stats['percentiles']
        expected_percentiles = ['1%', '5%', '25%', '50%', '75%', '95%', '99%']
        for p in expected_percentiles:
            assert p in percentiles
        
        # Check health indicators
        health = stats['health_indicators']
        assert 'parameter_magnitude' in health
        
        param_health = health['parameter_magnitude']
        assert 'status' in param_health
        assert 'message' in param_health
        assert param_health['status'] in ['healthy', 'warning', 'error']
    
    def test_parameter_statistics_with_gradients(self, inspector):
        """Test parameter statistics with gradient information."""
        # Create dummy gradients
        for param in inspector.model.parameters():
            param.grad = torch.randn_like(param) * 0.01
        
        stats = inspector.get_parameter_statistics()
        
        # Check that gradient stats are included
        assert 'gradient_stats' in stats
        gradient_stats = stats['gradient_stats']
        
        if gradient_stats:  # If gradients were available
            assert 'mean' in gradient_stats
            assert 'std' in gradient_stats
            assert 'norm' in gradient_stats
            assert 'percentiles' in gradient_stats
        
        # Check layer-level gradient stats
        layer_stats = stats['layer_stats']
        for layer_stat in layer_stats.values():
            if 'gradient' in layer_stat:
                grad_stat = layer_stat['gradient']
                assert 'mean' in grad_stat
                assert 'std' in grad_stat
                assert 'norm' in grad_stat
        
        # Check gradient health indicator
        health = stats['health_indicators']
        if 'gradient_health' in health:
            grad_health = health['gradient_health']
            assert 'status' in grad_health
            assert 'message' in grad_health
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_visualize_parameter_distribution(self, mock_savefig, mock_show, inspector):
        """Test parameter distribution visualization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "param_dist.png")
            
            # Test visualization without saving
            inspector.visualize_parameter_distribution(show_plot=False)
            mock_show.assert_not_called()
            
            # Test visualization with saving
            inspector.visualize_parameter_distribution(save_path=save_path, show_plot=False)
            mock_savefig.assert_called_with(save_path, dpi=300, bbox_inches='tight')
    
    def test_visualize_parameter_distribution_missing_deps(self, inspector):
        """Test parameter visualization with missing dependencies."""
        with patch.dict('sys.modules', {'matplotlib': None, 'seaborn': None}):
            with pytest.raises(ModelError) as exc_info:
                inspector.visualize_parameter_distribution()
            
            assert "matplotlib and seaborn are required" in str(exc_info.value)
    
    def test_inspect_activations(self, inspector):
        """Test activation inspection functionality."""
        # Create sample input
        batch_size, seq_length = 2, 10
        input_data = torch.randint(0, 50, (batch_size, seq_length))
        
        # Inspect activations
        activations = inspector.inspect_activations(input_data)
        
        # Check that activations were captured
        assert isinstance(activations, dict)
        assert len(activations) > 0
        
        # Check activation analysis
        for layer_name, activation_info in activations.items():
            assert 'tensor' in activation_info
            assert 'shape' in activation_info
            assert 'mean' in activation_info
            assert 'std' in activation_info
            assert 'min' in activation_info
            assert 'max' in activation_info
            assert 'zero_fraction' in activation_info
            assert 'activation_norm' in activation_info
            
            # Verify tensor properties
            tensor = activation_info['tensor']
            assert isinstance(tensor, torch.Tensor)
            assert tensor.shape[0] == batch_size  # Batch dimension should match
    
    def test_inspect_activations_specific_layers(self, inspector):
        """Test activation inspection for specific layers."""
        batch_size, seq_length = 2, 10
        input_data = torch.randint(0, 50, (batch_size, seq_length))
        
        # Inspect only embedding layer
        activations = inspector.inspect_activations(input_data, layer_names=['embedding'])
        
        assert len(activations) == 1
        assert 'embedding' in activations
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_visualize_activations(self, mock_savefig, mock_show, inspector):
        """Test activation visualization."""
        batch_size, seq_length = 2, 10
        input_data = torch.randint(0, 50, (batch_size, seq_length))
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "activations.png")
            
            # Test visualization
            inspector.visualize_activations(input_data, save_path=save_path, show_plot=False)
            mock_savefig.assert_called_with(save_path, dpi=300, bbox_inches='tight')
    
    def test_profile_model_performance(self, inspector):
        """Test model performance profiling."""
        batch_size, seq_length = 4, 20
        input_data = torch.randint(0, 50, (batch_size, seq_length))
        
        # Profile with small number of runs for testing
        profile_results = inspector.profile_model_performance(input_data, num_runs=5)
        
        # Check profile structure
        assert 'total_time' in profile_results
        assert 'throughput' in profile_results
        assert 'memory_usage' in profile_results
        assert 'bottlenecks' in profile_results
        assert 'optimization_suggestions' in profile_results
        
        # Check total time metrics
        total_time = profile_results['total_time']
        assert 'mean' in total_time
        assert 'std' in total_time
        assert 'min' in total_time
        assert 'max' in total_time
        assert total_time['mean'] > 0
        
        # Check throughput metrics
        throughput = profile_results['throughput']
        assert 'tokens_per_second' in throughput
        assert 'sequences_per_second' in throughput
        assert 'batch_size' in throughput
        assert 'sequence_length' in throughput
        assert throughput['batch_size'] == batch_size
        assert throughput['sequence_length'] == seq_length
        
        # Check memory usage
        memory_usage = profile_results['memory_usage']
        assert 'allocated_mb' in memory_usage
        assert 'reserved_mb' in memory_usage
        assert 'device' in memory_usage
        
        # Check optimization suggestions
        suggestions = profile_results['optimization_suggestions']
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0


class TestTrainingVisualizer:
    """Test cases for TrainingVisualizer class."""
    
    @pytest.fixture
    def sample_history(self):
        """Create sample training history."""
        return {
            'train_loss': [2.5, 2.2, 1.9, 1.7, 1.5, 1.4, 1.3, 1.25, 1.2, 1.18],
            'epoch_times': [10.5, 11.2, 10.8, 11.0, 10.9, 11.1, 10.7, 10.8, 11.0, 10.9],
            'learning_rates': [0.001] * 10
        }
    
    @pytest.fixture
    def visualizer(self, sample_history):
        """Create a TrainingVisualizer instance."""
        return TrainingVisualizer(sample_history)
    
    def test_visualizer_initialization(self):
        """Test TrainingVisualizer initialization."""
        # Test with no history
        visualizer = TrainingVisualizer()
        assert visualizer.training_history == {}
        
        # Test with history
        history = {'train_loss': [1.0, 0.8, 0.6]}
        visualizer = TrainingVisualizer(history)
        assert visualizer.training_history == history
    
    def test_update_history(self, visualizer):
        """Test updating training history."""
        new_history = {'train_loss': [3.0, 2.5, 2.0], 'epoch_times': [5.0, 5.1, 4.9]}
        visualizer.update_history(new_history)
        assert visualizer.training_history == new_history
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_plot_training_progress(self, mock_savefig, mock_show, visualizer):
        """Test training progress plotting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "training_progress.png")
            
            # Test plotting without saving
            visualizer.plot_training_progress(show_plot=False)
            mock_show.assert_not_called()
            
            # Test plotting with saving
            visualizer.plot_training_progress(save_path=save_path, show_plot=False)
            mock_savefig.assert_called_with(save_path, dpi=300, bbox_inches='tight')
    
    def test_plot_training_progress_no_history(self):
        """Test plotting with no training history."""
        visualizer = TrainingVisualizer()
        
        with pytest.raises(ModelError) as exc_info:
            visualizer.plot_training_progress()
        
        assert "No training history available" in str(exc_info.value)
    
    def test_plot_training_progress_missing_deps(self, visualizer):
        """Test plotting with missing dependencies."""
        with patch.dict('sys.modules', {'matplotlib': None, 'seaborn': None}):
            with pytest.raises(ModelError) as exc_info:
                visualizer.plot_training_progress()
            
            assert "matplotlib and seaborn are required" in str(exc_info.value)
    
    def test_analyze_training_dynamics(self, visualizer):
        """Test training dynamics analysis."""
        analysis = visualizer.analyze_training_dynamics()
        
        # Check main sections
        assert 'convergence_analysis' in analysis
        assert 'stability_metrics' in analysis
        assert 'efficiency_metrics' in analysis
        assert 'recommendations' in analysis
        
        # Check convergence analysis
        convergence = analysis['convergence_analysis']
        assert 'total_improvement' in convergence
        assert 'relative_improvement' in convergence
        assert 'convergence_rate' in convergence
        assert 'is_plateaued' in convergence
        assert 'final_loss' in convergence
        assert 'best_loss' in convergence
        assert 'epochs_to_best' in convergence
        
        # Verify convergence metrics make sense
        assert convergence['total_improvement'] > 0  # Loss should decrease
        assert convergence['relative_improvement'] > 0
        assert convergence['final_loss'] < 2.5  # Should be less than initial
        
        # Check stability metrics
        stability = analysis['stability_metrics']
        assert 'volatility' in stability
        assert 'oscillation_rate' in stability
        assert 'spike_count' in stability
        assert 'monotonicity' in stability
        assert 'stability_score' in stability
        
        # Check efficiency metrics
        efficiency = analysis['efficiency_metrics']
        assert 'time_metrics' in efficiency
        assert 'loss_efficiency' in efficiency
        
        time_metrics = efficiency['time_metrics']
        assert 'mean_epoch_time' in time_metrics
        assert 'total_training_time' in time_metrics
        assert 'time_stability' in time_metrics
        
        loss_efficiency = efficiency['loss_efficiency']
        assert 'improvement_per_epoch' in loss_efficiency
        assert 'efficiency_score' in loss_efficiency
        
        # Check recommendations
        recommendations = analysis['recommendations']
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
    
    def test_analyze_training_dynamics_no_loss_data(self):
        """Test analysis with no loss data."""
        visualizer = TrainingVisualizer({'epoch_times': [1.0, 1.1, 1.0]})
        
        with pytest.raises(ModelError) as exc_info:
            visualizer.analyze_training_dynamics()
        
        assert "No training loss data available" in str(exc_info.value)
    
    def test_analyze_training_dynamics_insufficient_data(self):
        """Test analysis with insufficient data."""
        visualizer = TrainingVisualizer({'train_loss': [1.0]})
        
        analysis = visualizer.analyze_training_dynamics()
        
        # Should handle insufficient data gracefully
        assert analysis['convergence_analysis']['status'] == 'insufficient_data'
        assert analysis['stability_metrics']['status'] == 'insufficient_data'


class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    @pytest.fixture
    def sample_model(self):
        """Create a sample model for testing."""
        return MicroLM(vocab_size=30, embedding_dim=16, hidden_dim=32, num_layers=1)
    
    @pytest.fixture
    def sample_history(self):
        """Create sample training history."""
        return {
            'train_loss': [2.0, 1.8, 1.6, 1.4, 1.2],
            'epoch_times': [5.0, 5.1, 4.9, 5.0, 5.2],
            'learning_rates': [0.001] * 5
        }
    
    def test_inspect_model_function(self, sample_model, capsys):
        """Test the inspect_model utility function."""
        # Test basic inspection
        inspect_model(sample_model, detailed=False)
        captured = capsys.readouterr()
        assert "MICROLSTM - ARCHITECTURE SUMMARY" in captured.out
        
        # Test detailed inspection
        inspect_model(sample_model, detailed=True)
        captured = capsys.readouterr()
        assert "Layer Details:" in captured.out
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_visualize_training_function(self, mock_savefig, mock_show, sample_history):
        """Test the visualize_training utility function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "training.png")
            
            # Test without saving
            visualize_training(sample_history)
            mock_show.assert_called_once()
            
            # Test with saving
            mock_show.reset_mock()
            visualize_training(sample_history, save_path=save_path)
            mock_savefig.assert_called_with(save_path, dpi=300, bbox_inches='tight')
    
    @patch('matplotlib.pyplot.show')
    def test_analyze_parameters_function(self, mock_show, sample_model):
        """Test the analyze_parameters utility function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "params.png")
            
            # Test without saving
            stats = analyze_parameters(sample_model)
            assert isinstance(stats, dict)
            assert 'layer_stats' in stats
            assert 'overall_stats' in stats
            
            # Test with saving
            stats = analyze_parameters(sample_model, save_path=save_path)
            assert isinstance(stats, dict)


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_very_small_model(self):
        """Test inspection of very small model."""
        model = MicroLM(vocab_size=5, embedding_dim=4, hidden_dim=8, num_layers=1)
        inspector = ModelInspector(model)
        
        summary = inspector.get_architecture_summary()
        assert summary['basic_info']['architecture']['vocab_size'] == 5
        
        # Should handle small model gracefully
        stats = inspector.get_parameter_statistics()
        assert stats['overall_stats']['total_parameters'] > 0
    
    def test_model_with_zero_parameters(self):
        """Test handling of edge case with minimal parameters."""
        model = MicroLM(vocab_size=2, embedding_dim=1, hidden_dim=2, num_layers=1)
        inspector = ModelInspector(model)
        
        # Should not crash with very small model
        summary = inspector.get_architecture_summary()
        assert 'basic_info' in summary
    
    def test_training_history_edge_cases(self):
        """Test training visualizer with edge case histories."""
        # Single epoch
        single_epoch = {'train_loss': [1.0]}
        visualizer = TrainingVisualizer(single_epoch)
        
        # Should handle single epoch gracefully
        try:
            analysis = visualizer.analyze_training_dynamics()
            # Some metrics might not be available, but shouldn't crash
        except ModelError:
            pass  # Expected for insufficient data
        
        # Empty history
        empty_visualizer = TrainingVisualizer({})
        with pytest.raises(ModelError):
            empty_visualizer.analyze_training_dynamics()
    
    def test_activation_inspection_edge_cases(self):
        """Test activation inspection edge cases."""
        model = MicroLM(vocab_size=10, embedding_dim=8, hidden_dim=16, num_layers=1)
        inspector = ModelInspector(model)
        
        # Very small input
        small_input = torch.randint(0, 10, (1, 1))
        activations = inspector.inspect_activations(small_input)
        assert len(activations) > 0
        
        # Large input (within reason for testing)
        large_input = torch.randint(0, 10, (8, 50))
        activations = inspector.inspect_activations(large_input)
        assert len(activations) > 0
    
    def test_performance_profiling_edge_cases(self):
        """Test performance profiling edge cases."""
        model = MicroLM(vocab_size=10, embedding_dim=8, hidden_dim=16, num_layers=1)
        inspector = ModelInspector(model)
        
        # Very small input
        small_input = torch.randint(0, 10, (1, 1))
        profile = inspector.profile_model_performance(small_input, num_runs=2)
        assert 'total_time' in profile
        
        # Single run
        profile = inspector.profile_model_performance(small_input, num_runs=1)
        assert 'total_time' in profile


if __name__ == "__main__":
    pytest.main([__file__])