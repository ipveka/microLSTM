"""
Custom exception classes for MicroLSTM.

This module defines custom exception classes that provide clear, specific error
messages for different types of failures that can occur during model training,
text generation, and data processing. These exceptions help users understand
what went wrong and how to fix issues.

The exception hierarchy follows Python best practices:
- Base exception class for all model-related errors
- Specific exception classes for different components
- Clear error messages with helpful context
"""


class ModelError(Exception):
    """
    Base exception class for all MicroLSTM related errors.
    
    This is the parent class for all custom exceptions in the micro_lstm package.
    It provides a consistent interface and allows users to catch all model-related
    errors with a single exception type if needed.
    
    Attributes:
        message (str): Human-readable error message
        error_code (str): Optional error code for programmatic handling
        context (dict): Optional context information about the error
    
    Example:
        >>> try:
        ...     # Some model operation
        ...     pass
        ... except ModelError as e:
        ...     print(f"Model error occurred: {e}")
        ...     if hasattr(e, 'context'):
        ...         print(f"Context: {e.context}")
    """
    
    def __init__(self, message: str, error_code: str = None, context: dict = None):
        """
        Initialize the ModelError with message and optional context.
        
        Args:
            message (str): Human-readable error description
            error_code (str, optional): Machine-readable error code
            context (dict, optional): Additional context information
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
    
    def __str__(self) -> str:
        """Return a formatted error message."""
        base_msg = self.message
        if self.error_code:
            base_msg = f"[{self.error_code}] {base_msg}"
        return base_msg
    
    def __repr__(self) -> str:
        """Return a detailed representation of the error."""
        return f"{self.__class__.__name__}(message='{self.message}', error_code='{self.error_code}')"


class TrainingError(ModelError):
    """
    Exception raised when training-related operations fail.
    
    This exception is raised during model training when issues occur such as:
    - Invalid training parameters
    - Data loading failures
    - Optimization problems
    - Memory issues during training
    - Model saving/loading failures during training
    
    The exception provides specific context about what went wrong during
    the training process to help users diagnose and fix issues.
    
    Example:
        >>> try:
        ...     trainer.train(data_loader, epochs=-1)  # Invalid epochs
        ... except TrainingError as e:
        ...     print(f"Training failed: {e}")
        ...     print(f"Error occurred at epoch: {e.context.get('epoch', 'unknown')}")
    """
    
    def __init__(self, message: str, epoch: int = None, batch: int = None, **kwargs):
        """
        Initialize TrainingError with training-specific context.
        
        Args:
            message (str): Error description
            epoch (int, optional): Epoch number where error occurred
            batch (int, optional): Batch number where error occurred
            **kwargs: Additional context information
        """
        context = kwargs.copy()
        if epoch is not None:
            context['epoch'] = epoch
        if batch is not None:
            context['batch'] = batch
        
        super().__init__(message, error_code="TRAINING_ERROR", context=context)


class GenerationError(ModelError):
    """
    Exception raised when text generation operations fail.
    
    This exception is raised during text generation when issues occur such as:
    - Invalid generation parameters
    - Model not properly loaded
    - Prompt processing failures
    - Sampling strategy errors
    - Memory issues during generation
    
    The exception provides context about the generation parameters and state
    to help users understand what caused the failure.
    
    Example:
        >>> try:
        ...     generator.generate("hello", length=-10)  # Invalid length
        ... except GenerationError as e:
        ...     print(f"Generation failed: {e}")
        ...     print(f"Prompt was: {e.context.get('prompt', 'unknown')}")
    """
    
    def __init__(self, message: str, prompt: str = None, length: int = None, **kwargs):
        """
        Initialize GenerationError with generation-specific context.
        
        Args:
            message (str): Error description
            prompt (str, optional): Prompt that caused the error
            length (int, optional): Requested generation length
            **kwargs: Additional context information
        """
        context = kwargs.copy()
        if prompt is not None:
            context['prompt'] = prompt
        if length is not None:
            context['length'] = length
        
        super().__init__(message, error_code="GENERATION_ERROR", context=context)


class TokenizationError(ModelError):
    """
    Exception raised when tokenization operations fail.
    
    This exception is raised during text tokenization when issues occur such as:
    - Unknown characters in input text
    - Invalid token indices during decoding
    - Vocabulary loading/saving failures
    - Incompatible tokenizer configurations
    
    The exception provides context about the tokenization operation to help
    users understand what text or tokens caused the issue.
    
    Example:
        >>> try:
        ...     tokenizer.encode("hello 🚀")  # Emoji not in vocab
        ... except TokenizationError as e:
        ...     print(f"Tokenization failed: {e}")
        ...     print(f"Problematic text: {e.context.get('text', 'unknown')}")
    """
    
    def __init__(self, message: str, text: str = None, tokens: list = None, **kwargs):
        """
        Initialize TokenizationError with tokenization-specific context.
        
        Args:
            message (str): Error description
            text (str, optional): Text that caused the error
            tokens (list, optional): Token sequence that caused the error
            **kwargs: Additional context information
        """
        context = kwargs.copy()
        if text is not None:
            context['text'] = text
        if tokens is not None:
            context['tokens'] = tokens
        
        super().__init__(message, error_code="TOKENIZATION_ERROR", context=context)


class DataError(ModelError):
    """
    Exception raised when data processing operations fail.
    
    This exception is raised during data preparation and processing when issues occur such as:
    - Invalid data formats
    - Insufficient data for training
    - Data loading failures
    - Sequence preparation errors
    - Batch creation problems
    
    The exception provides context about the data operation to help users
    understand what data caused the issue.
    
    Example:
        >>> try:
        ...     prepare_sequences("", sequence_length=10)  # Empty text
        ... except DataError as e:
        ...     print(f"Data processing failed: {e}")
        ...     print(f"Data length: {e.context.get('data_length', 'unknown')}")
    """
    
    def __init__(self, message: str, data_length: int = None, sequence_length: int = None, **kwargs):
        """
        Initialize DataError with data-specific context.
        
        Args:
            message (str): Error description
            data_length (int, optional): Length of problematic data
            sequence_length (int, optional): Requested sequence length
            **kwargs: Additional context information
        """
        context = kwargs.copy()
        if data_length is not None:
            context['data_length'] = data_length
        if sequence_length is not None:
            context['sequence_length'] = sequence_length
        
        super().__init__(message, error_code="DATA_ERROR", context=context)


class ModelConfigurationError(ModelError):
    """
    Exception raised when model configuration is invalid.
    
    This exception is raised when model parameters or configuration are invalid such as:
    - Invalid architecture parameters (negative dimensions, etc.)
    - Incompatible model and tokenizer configurations
    - Invalid device specifications
    - Unsupported model configurations
    
    The exception provides context about the configuration issue to help users
    understand what parameters need to be corrected.
    
    Example:
        >>> try:
        ...     MicroLSTM(vocab_size=-10, embedding_dim=128)  # Invalid vocab_size
        ... except ModelConfigurationError as e:
        ...     print(f"Model configuration error: {e}")
        ...     print(f"Invalid parameter: {e.context.get('parameter', 'unknown')}")
    """
    
    def __init__(self, message: str, parameter: str = None, value: any = None, **kwargs):
        """
        Initialize ModelConfigurationError with configuration-specific context.
        
        Args:
            message (str): Error description
            parameter (str, optional): Name of the invalid parameter
            value (any, optional): Invalid parameter value
            **kwargs: Additional context information
        """
        context = kwargs.copy()
        if parameter is not None:
            context['parameter'] = parameter
        if value is not None:
            context['value'] = value
        
        super().__init__(message, error_code="CONFIG_ERROR", context=context)


class FileOperationError(ModelError):
    """
    Exception raised when file operations fail.
    
    This exception is raised during file I/O operations when issues occur such as:
    - Model saving/loading failures
    - Vocabulary file operations
    - Permission errors
    - Disk space issues
    - Corrupted files
    
    The exception provides context about the file operation to help users
    understand what file operation failed and why.
    
    Example:
        >>> try:
        ...     model.save("/invalid/path/model.pt")
        ... except FileOperationError as e:
        ...     print(f"File operation failed: {e}")
        ...     print(f"File path: {e.context.get('filepath', 'unknown')}")
    """
    
    def __init__(self, message: str, filepath: str = None, operation: str = None, **kwargs):
        """
        Initialize FileOperationError with file operation context.
        
        Args:
            message (str): Error description
            filepath (str, optional): Path of the file that caused the error
            operation (str, optional): Type of operation that failed (save, load, etc.)
            **kwargs: Additional context information
        """
        context = kwargs.copy()
        if filepath is not None:
            context['filepath'] = filepath
        if operation is not None:
            context['operation'] = operation
        
        super().__init__(message, error_code="FILE_ERROR", context=context)


class CudaError(ModelError):
    """
    Exception raised when CUDA/GPU operations fail.
    
    This exception is raised when GPU-related operations fail such as:
    - CUDA out of memory errors
    - Device compatibility issues
    - GPU driver problems
    - Tensor device mismatches
    
    The exception provides context about the CUDA operation to help users
    understand GPU-related issues and potential solutions.
    
    Example:
        >>> try:
        ...     model.to('cuda')  # GPU not available
        ... except CudaError as e:
        ...     print(f"CUDA error: {e}")
        ...     print(f"Suggested solution: {e.context.get('suggestion', 'Use CPU instead')}")
    """
    
    def __init__(self, message: str, device: str = None, suggestion: str = None, **kwargs):
        """
        Initialize CudaError with CUDA-specific context.
        
        Args:
            message (str): Error description
            device (str, optional): Device that caused the error
            suggestion (str, optional): Suggested solution
            **kwargs: Additional context information
        """
        context = kwargs.copy()
        if device is not None:
            context['device'] = device
        if suggestion is not None:
            context['suggestion'] = suggestion
        
        super().__init__(message, error_code="CUDA_ERROR", context=context)


# Convenience function to create appropriate exception based on error type
def create_model_error(error_type: str, message: str, **kwargs) -> ModelError:
    """
    Factory function to create appropriate exception based on error type.
    
    This function provides a convenient way to create the right type of exception
    based on the context, which can be useful for error handling utilities.
    
    Args:
        error_type (str): Type of error ('training', 'generation', 'tokenization', etc.)
        message (str): Error message
        **kwargs: Additional context for the specific error type
    
    Returns:
        ModelError: Appropriate exception instance
    
    Example:
        >>> error = create_model_error('training', 'Invalid epoch count', epoch=5)
        >>> raise error
    """
    error_type = error_type.lower()
    
    if error_type == 'training':
        return TrainingError(message, **kwargs)
    elif error_type == 'generation':
        return GenerationError(message, **kwargs)
    elif error_type == 'tokenization':
        return TokenizationError(message, **kwargs)
    elif error_type == 'data':
        return DataError(message, **kwargs)
    elif error_type == 'config':
        return ModelConfigurationError(message, **kwargs)
    elif error_type == 'file':
        return FileOperationError(message, **kwargs)
    elif error_type == 'cuda':
        return CudaError(message, **kwargs)
    else:
        return ModelError(message, **kwargs)