#!/usr/bin/env python3
"""
Performance optimizations for the Micro Language Model.

This script applies various performance optimizations while maintaining
code clarity and educational value.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import warnings


def optimize_model_for_inference(model: nn.Module) -> nn.Module:
    """
    Apply inference-time optimizations to the model.
    
    Args:
        model (nn.Module): Model to optimize
        
    Returns:
        nn.Module: Optimized model
    """
    # Set model to evaluation mode
    model.eval()
    
    # Disable gradient computation for inference
    for param in model.parameters():
        param.requires_grad = False
    
    # Try to compile the model if PyTorch 2.0+ is available
    try:
        if hasattr(torch, 'compile'):
            print("Applying torch.compile optimization...")
            # Suppress dynamo errors for educational model
            torch._dynamo.config.suppress_errors = True
            model = torch.compile(model, mode='default')
        else:
            print("torch.compile not available (requires PyTorch 2.0+)")
    except Exception as e:
        print(f"Could not apply torch.compile (this is normal for educational models): {e}")
        print("Continuing without torch.compile optimization...")
    
    return model


def optimize_memory_usage():
    """Apply memory optimization settings."""
    # Enable memory efficient attention if available
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        print("Enabled Flash Attention for memory efficiency")
    except:
        pass
    
    # Set memory fraction for CUDA if available
    if torch.cuda.is_available():
        try:
            torch.cuda.set_per_process_memory_fraction(0.8)
            print("Set CUDA memory fraction to 80%")
        except:
            pass


def create_optimized_data_loader(dataset, batch_size: int, num_workers: Optional[int] = None):
    """
    Create an optimized DataLoader with performance settings.
    
    Args:
        dataset: PyTorch dataset
        batch_size (int): Batch size
        num_workers (int, optional): Number of worker processes
        
    Returns:
        DataLoader: Optimized data loader
    """
    from torch.utils.data import DataLoader
    
    # Auto-detect optimal number of workers
    if num_workers is None:
        import os
        num_workers = min(4, os.cpu_count() or 1)
    
    # Create optimized DataLoader
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),  # Pin memory for GPU
        persistent_workers=num_workers > 0,    # Keep workers alive
        prefetch_factor=2 if num_workers > 0 else 2,  # Prefetch batches
        drop_last=False
    )
    
    print(f"Created optimized DataLoader with {num_workers} workers")
    return data_loader


def apply_mixed_precision_training():
    """
    Configure mixed precision training for better performance.
    
    Returns:
        Tuple: (scaler, autocast_context)
    """
    if torch.cuda.is_available():
        try:
            from torch.cuda.amp import GradScaler, autocast
            scaler = GradScaler()
            print("Enabled mixed precision training (FP16)")
            return scaler, autocast
        except ImportError:
            print("Mixed precision not available")
            return None, None
    else:
        print("Mixed precision requires CUDA")
        return None, None


def optimize_lstm_performance(model):
    """
    Apply LSTM-specific optimizations.
    
    Args:
        model: Model containing LSTM layers
    """
    # Enable cuDNN benchmark mode for consistent input sizes
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print("Enabled cuDNN benchmark mode")
    
    # Flatten LSTM parameters for better memory layout
    for module in model.modules():
        if isinstance(module, nn.LSTM):
            module.flatten_parameters()
            print("Flattened LSTM parameters for better performance")


def profile_model_performance(model, sample_input, num_runs: int = 100):
    """
    Profile model performance and provide optimization suggestions.
    
    Args:
        model: Model to profile
        sample_input: Sample input tensor
        num_runs (int): Number of profiling runs
    """
    import time
    
    model.eval()
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(10):
            _ = model(sample_input)
    
    # Timing runs
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            output = model(sample_input)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    throughput = sample_input.size(0) / avg_time  # samples per second
    
    print(f"\nPerformance Profile:")
    print(f"Average inference time: {avg_time*1000:.2f} ms")
    print(f"Throughput: {throughput:.1f} samples/second")
    print(f"Output shape: {output.shape}")
    
    # Memory usage
    if torch.cuda.is_available():
        memory_used = torch.cuda.max_memory_allocated() / 1024**2
        print(f"Peak GPU memory: {memory_used:.1f} MB")
    
    return avg_time, throughput


def suggest_optimizations(model, training_time: float, generation_time: float):
    """
    Provide optimization suggestions based on profiling results.
    
    Args:
        model: The model to analyze
        training_time (float): Training time per epoch
        generation_time (float): Generation time per character
    """
    print("\n" + "="*60)
    print("PERFORMANCE OPTIMIZATION SUGGESTIONS")
    print("="*60)
    
    # Model size analysis
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = (total_params * 4) / (1024**2)
    
    print(f"Model Analysis:")
    print(f"  Parameters: {total_params:,}")
    print(f"  Model size: {model_size_mb:.1f} MB")
    
    # Training optimizations
    print(f"\nTraining Optimizations:")
    if training_time > 10:
        print("  • Consider reducing batch size or sequence length")
        print("  • Enable mixed precision training (FP16)")
        print("  • Use gradient accumulation for large effective batch sizes")
    
    if not torch.cuda.is_available():
        print("  • Consider using GPU for significant speedup")
    else:
        print("  • GPU available - ensure model and data are on GPU")
    
    # Generation optimizations
    print(f"\nGeneration Optimizations:")
    if generation_time > 0.1:
        print("  • Use torch.compile() for faster inference (PyTorch 2.0+)")
        print("  • Consider model quantization for deployment")
        print("  • Batch multiple generation requests together")
    
    # Memory optimizations
    print(f"\nMemory Optimizations:")
    if model_size_mb > 100:
        print("  • Consider model pruning or distillation")
        print("  • Use gradient checkpointing during training")
    
    print("  • Enable memory-efficient attention mechanisms")
    print("  • Use pin_memory=True in DataLoader for GPU training")
    
    # Architecture suggestions
    print(f"\nArchitecture Suggestions:")
    if total_params < 1000:
        print("  • Model might be too small - consider increasing capacity")
    elif total_params > 10_000_000:
        print("  • Large model - ensure sufficient training data")
    
    print("="*60)


def main():
    """Demonstrate performance optimizations."""
    print("Micro Language Model - Performance Optimizations")
    print("="*60)
    
    # Apply global optimizations
    optimize_memory_usage()
    
    # Example model for demonstration
    from micro_lm import CharacterTokenizer, MicroLM
    
    text = "Hello world! This is a performance test."
    tokenizer = CharacterTokenizer(text)
    
    model = MicroLM(
        vocab_size=tokenizer.vocab_size(),
        embedding_dim=64,
        hidden_dim=128,
        num_layers=2
    )
    
    print(f"\nOriginal model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Apply optimizations
    optimize_lstm_performance(model)
    optimized_model = optimize_model_for_inference(model)
    
    # Profile performance
    sample_input = torch.randint(0, tokenizer.vocab_size(), (4, 50))
    avg_time, throughput = profile_model_performance(optimized_model, sample_input)
    
    # Provide suggestions
    suggest_optimizations(model, training_time=5.0, generation_time=avg_time)
    
    print("\nOptimization complete!")


if __name__ == "__main__":
    main()