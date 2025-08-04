"""
Text generation module for MicroLSTM.

This module provides the TextGenerator class that handles text generation using
trained language models. It supports multiple sampling strategies including
greedy decoding and temperature-based sampling for varied output generation.

The generator is designed to be educational and includes extensive comments
explaining different text generation techniques and their trade-offs.
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Callable, Dict, Any
import warnings

from .model import MicroLM
from .tokenizer import CharacterTokenizer
from .exceptions import GenerationError, ModelConfigurationError, CudaError


class TextGenerator:
    """
    Text generator for MicroLSTM with multiple sampling strategies.
    
    This class provides text generation capabilities using trained language models.
    It supports different sampling strategies to control the creativity and
    determinism of generated text:
    
    1. Greedy Decoding: Always selects the most probable next character
    2. Temperature Sampling: Uses temperature to control randomness
    3. Top-k Sampling: Samples from the k most probable characters
    4. Top-p (Nucleus) Sampling: Samples from characters with cumulative probability p
    
    The generator handles prompt processing, continuation generation, length control,
    and various stopping criteria to produce coherent and controllable text output.
    
    Args:
        model (MicroLM): Trained language model for text generation
        tokenizer (CharacterTokenizer): Tokenizer for text processing
        device (torch.device, optional): Device to run generation on
        
    Attributes:
        model (MicroLM): The language model used for generation
        tokenizer (CharacterTokenizer): Text tokenizer
        device (torch.device): Generation device
        
    Example:
        >>> generator = TextGenerator(trained_model, tokenizer)
        >>> text = generator.generate("Hello", length=50, temperature=0.8)
        >>> print(text)  # "Hello world, this is generated text..."
    """
    
    def __init__(
        self,
        model: MicroLM,
        tokenizer: CharacterTokenizer,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the text generator with model and tokenizer.
        
        Sets up the generation environment and validates compatibility between
        the model and tokenizer. The generator automatically detects device
        and ensures the model is in evaluation mode for inference.
        
        Args:
            model (MicroLM): Trained language model for generation
            tokenizer (CharacterTokenizer): Tokenizer for text processing
            device (torch.device, optional): Device for generation. If None,
                                           uses the same device as the model
        
        Raises:
            TypeError: If model or tokenizer have incorrect types
            ValueError: If model and tokenizer have incompatible vocabulary sizes
            GenerationError: If model is not properly initialized
        """
        # Validate input types
        if not isinstance(model, MicroLM):
            raise ModelConfigurationError(
                f"model must be MicroLM instance, got {type(model)}",
                parameter="model",
                value=type(model)
            )
        
        if not isinstance(tokenizer, CharacterTokenizer):
            raise ModelConfigurationError(
                f"tokenizer must be CharacterTokenizer instance, got {type(tokenizer)}",
                parameter="tokenizer",
                value=type(tokenizer)
            )
        
        # Validate compatibility between model and tokenizer
        if model.vocab_size != tokenizer.vocab_size():
            raise ModelConfigurationError(
                f"Model vocab_size ({model.vocab_size}) must match tokenizer vocab_size "
                f"({tokenizer.vocab_size()})",
                parameter="vocab_size_mismatch",
                model_vocab_size=model.vocab_size,
                tokenizer_vocab_size=tokenizer.vocab_size()
            )
        
        self.model = model
        self.tokenizer = tokenizer
        
        # Set up device
        if device is None:
            try:
                # Use the same device as the model
                self.device = next(model.parameters()).device
            except StopIteration:
                raise ModelConfigurationError(
                    "Model has no parameters to determine device from",
                    parameter="device"
                )
        else:
            if not isinstance(device, torch.device):
                try:
                    self.device = torch.device(device)
                except Exception as e:
                    raise CudaError(
                        f"Invalid device specification: {device}",
                        device=str(device),
                        original_error=str(e)
                    )
            else:
                self.device = device
            
            # Move model to specified device if needed
            try:
                if next(model.parameters()).device != self.device:
                    self.model.to(self.device)
            except RuntimeError as e:
                if "CUDA" in str(e) or "out of memory" in str(e).lower():
                    raise CudaError(
                        f"Failed to move model to device {self.device}: {e}",
                        device=str(self.device),
                        suggestion="Try using CPU or reducing model size",
                        original_error=str(e)
                    )
                else:
                    raise ModelConfigurationError(
                        f"Failed to move model to device: {e}",
                        parameter="device",
                        value=str(self.device),
                        original_error=str(e)
                    )
        
        # Set model to evaluation mode for inference
        # This disables dropout and batch normalization training behavior
        self.model.eval()
        
        # Validate that model has been trained (has non-random weights)
        self._validate_model_state()
        
        print(f"TextGenerator initialized on device: {self.device}")
        print(f"Vocabulary size: {self.tokenizer.vocab_size()}")
    
    def _validate_model_state(self):
        """
        Validate that the model appears to be trained (not just randomly initialized).
        
        This is a basic check to warn users if they're trying to generate text
        with an untrained model, which would produce nonsensical output.
        """
        # Check if model parameters have reasonable variance
        # Randomly initialized models typically have very uniform distributions
        param_vars = []
        for param in self.model.parameters():
            if param.requires_grad and param.numel() > 1:
                param_vars.append(param.var().item())
        
        if param_vars:
            avg_var = sum(param_vars) / len(param_vars)
            # If variance is very low, model might be untrained
            if avg_var < 1e-6:
                warnings.warn(
                    "Model parameters have very low variance. "
                    "Make sure the model has been trained before generating text.",
                    UserWarning
                )
    
    def generate(
        self,
        prompt: str,
        length: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        stop_tokens: Optional[List[str]] = None,
        seed: Optional[int] = None
    ) -> str:
        """
        Generate text continuation from a given prompt.
        
        This is the main text generation method that supports multiple sampling
        strategies and generation controls. The method processes the prompt,
        generates tokens one by one using the specified sampling strategy,
        and returns the complete generated text.
        
        Sampling Strategies:
        - temperature=0.0: Greedy decoding (deterministic)
        - temperature>0.0: Temperature sampling (higher = more random)
        - top_k: Limit sampling to k most probable tokens
        - top_p: Nucleus sampling with cumulative probability threshold
        
        Args:
            prompt (str): Starting text for generation
            length (int): Number of characters to generate
            temperature (float): Sampling temperature (0.0 = greedy, higher = more random)
            top_k (int, optional): Limit sampling to top k tokens
            top_p (float, optional): Nucleus sampling threshold (0.0-1.0)
            repetition_penalty (float): Penalty for repeating tokens (>1.0 = less repetition)
            stop_tokens (List[str], optional): Tokens that stop generation
            seed (int, optional): Random seed for reproducible generation
        
        Returns:
            str: Generated text including the original prompt
        
        Raises:
            ValueError: If parameters are invalid
            GenerationError: If generation fails
        
        Example:
            >>> # Greedy generation (deterministic)
            >>> text = generator.generate("Hello", length=20, temperature=0.0)
            >>> 
            >>> # Creative generation with temperature
            >>> text = generator.generate("Once upon a time", length=100, temperature=0.8)
            >>> 
            >>> # Controlled generation with top-k sampling
            >>> text = generator.generate("The weather", length=50, temperature=0.7, top_k=10)
        """
        try:
            # Validate input parameters
            self._validate_generation_params(prompt, length, temperature, top_k, top_p, repetition_penalty)
            
            # Set random seed for reproducible generation
            if seed is not None:
                torch.manual_seed(seed)
            
            # Process prompt and prepare for generation
            prompt_tokens = self._process_prompt(prompt)
            
            # Initialize generation state
            generated_tokens = prompt_tokens.copy()
            current_sequence = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
            
            # Initialize LSTM hidden state
            hidden = self.model.init_hidden(batch_size=1, device=self.device)
            
            print(f"Generating {length} characters from prompt: '{prompt}'")
            print(f"Sampling strategy: temperature={temperature}, top_k={top_k}, top_p={top_p}")
            
            # Generation loop
            with torch.no_grad():  # Disable gradient computation for efficiency
                for step in range(length):
                    # Get next token probabilities
                    next_token, hidden = self._generate_next_token(
                        current_sequence, hidden, temperature, top_k, top_p, 
                        repetition_penalty, generated_tokens
                    )
                    
                    # Add generated token to sequence
                    generated_tokens.append(next_token)
                    
                    # Update current sequence (keep only recent context for efficiency)
                    # For very long generation, we can limit context window
                    max_context = min(len(generated_tokens), 200)  # Limit context to last 200 chars
                    current_sequence = torch.tensor([generated_tokens[-max_context:]], 
                                                  dtype=torch.long, device=self.device)
                    
                    # Check for stop tokens
                    if stop_tokens and self._should_stop(generated_tokens, stop_tokens):
                        print(f"Generation stopped at step {step + 1} due to stop token")
                        break
                    
                    # Optional: Print progress for long generations
                    if length > 100 and (step + 1) % 50 == 0:
                        progress = (step + 1) / length * 100
                        print(f"Generation progress: {progress:.1f}%")
            
            # Decode generated tokens back to text
            generated_text = self.tokenizer.decode(generated_tokens)
            
            print(f"Generation complete: {len(generated_tokens)} characters generated")
            return generated_text
            
        except GenerationError:
            # Re-raise GenerationError as-is to preserve context
            raise
        except Exception as e:
            raise GenerationError(f"Text generation failed: {e}")
    
    def _validate_generation_params(
        self, 
        prompt: str, 
        length: int, 
        temperature: float, 
        top_k: Optional[int], 
        top_p: Optional[float],
        repetition_penalty: float
    ):
        """Validate generation parameters."""
        if not isinstance(prompt, str):
            raise GenerationError(
                "Prompt must be a string",
                prompt=prompt
            )
        
        if len(prompt) == 0:
            raise GenerationError(
                "Prompt cannot be empty",
                prompt=prompt
            )
        
        if not isinstance(length, int) or length <= 0:
            raise GenerationError(
                f"Length must be a positive integer, got {length}",
                length=length
            )
        
        if not isinstance(temperature, (int, float)) or temperature < 0.0:
            raise GenerationError(
                f"Temperature must be a non-negative number, got {temperature}",
                temperature=temperature
            )
        
        if top_k is not None and (not isinstance(top_k, int) or top_k <= 0):
            raise GenerationError(
                f"top_k must be a positive integer, got {top_k}",
                top_k=top_k
            )
        
        if top_p is not None and (not isinstance(top_p, (int, float)) or not 0.0 <= top_p <= 1.0):
            raise GenerationError(
                f"top_p must be a number between 0.0 and 1.0, got {top_p}",
                top_p=top_p
            )
        
        if not isinstance(repetition_penalty, (int, float)) or repetition_penalty <= 0.0:
            raise GenerationError(
                f"repetition_penalty must be a positive number, got {repetition_penalty}",
                repetition_penalty=repetition_penalty
            )
    
    def _process_prompt(self, prompt: str) -> List[int]:
        """
        Process the input prompt and convert it to token indices.
        
        Args:
            prompt (str): Input prompt text
        
        Returns:
            List[int]: Token indices for the prompt
        
        Raises:
            GenerationError: If prompt contains unknown characters
        """
        try:
            # Encode prompt to token indices
            prompt_tokens = self.tokenizer.encode(prompt)
            
            if len(prompt_tokens) == 0:
                raise GenerationError("Prompt resulted in empty token sequence")
            
            return prompt_tokens
            
        except ValueError as e:
            # Handle unknown characters in prompt
            raise GenerationError(f"Prompt contains unknown characters: {e}")
    
    def _generate_next_token(
        self,
        current_sequence: torch.Tensor,
        hidden: tuple,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        repetition_penalty: float,
        generated_tokens: List[int]
    ) -> tuple:
        """
        Generate the next token using the specified sampling strategy.
        
        This method implements the core token generation logic:
        1. Forward pass through the model to get logits
        2. Apply repetition penalty if specified
        3. Apply temperature scaling
        4. Apply top-k or top-p filtering if specified
        5. Sample from the resulting probability distribution
        
        Args:
            current_sequence (torch.Tensor): Current token sequence
            hidden (tuple): LSTM hidden state
            temperature (float): Sampling temperature
            top_k (int, optional): Top-k filtering
            top_p (float, optional): Top-p (nucleus) filtering
            repetition_penalty (float): Repetition penalty factor
            generated_tokens (List[int]): Previously generated tokens
        
        Returns:
            tuple: (next_token_id, updated_hidden_state)
        """
        # Forward pass through the model
        # We need to get both logits and updated hidden state
        # The model's forward method returns only logits, but the LSTM internally updates hidden state
        
        # Step 1: Get logits and capture the updated hidden state
        # We'll use the model's internal LSTM to get both outputs
        batch_size, seq_length = current_sequence.shape
        
        try:
            # Convert to embeddings
            embedded = self.model.embedding(current_sequence)
            
            # Process through LSTM to get both output and updated hidden state
            lstm_out, updated_hidden = self.model.lstm(embedded, hidden)
            
            # Project to vocabulary logits
            logits = self.model.output_projection(lstm_out)
            
        except RuntimeError as e:
            if "CUDA" in str(e) or "out of memory" in str(e).lower():
                raise CudaError(
                    f"CUDA error during text generation: {e}",
                    device=str(self.device),
                    suggestion="Try reducing sequence length or using CPU",
                    original_error=str(e)
                )
            else:
                raise GenerationError(
                    f"Error during model forward pass: {e}",
                    original_error=str(e)
                )
        
        # Get logits for the last position (next token prediction)
        next_token_logits = logits[0, -1, :]  # Shape: (vocab_size,)
        
        # Apply repetition penalty to reduce repetitive generation
        if repetition_penalty != 1.0:
            next_token_logits = self._apply_repetition_penalty(
                next_token_logits, generated_tokens, repetition_penalty
            )
        
        # Apply temperature scaling
        if temperature == 0.0:
            # Greedy decoding: always select the most probable token
            next_token = torch.argmax(next_token_logits).item()
        else:
            # Temperature sampling: scale logits and sample from distribution
            scaled_logits = next_token_logits / temperature
            
            # Apply top-k filtering if specified
            if top_k is not None:
                scaled_logits = self._apply_top_k_filtering(scaled_logits, top_k)
            
            # Apply top-p (nucleus) filtering if specified
            if top_p is not None:
                scaled_logits = self._apply_top_p_filtering(scaled_logits, top_p)
            
            # Convert logits to probabilities and sample
            probabilities = F.softmax(scaled_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1).item()
        
        return next_token, updated_hidden
    
    def _apply_repetition_penalty(
        self, 
        logits: torch.Tensor, 
        generated_tokens: List[int], 
        penalty: float
    ) -> torch.Tensor:
        """
        Apply repetition penalty to reduce repetitive text generation.
        
        Repetition penalty works by reducing the probability of tokens that
        have already been generated. This helps create more diverse and
        interesting text by discouraging the model from repeating itself.
        
        Args:
            logits (torch.Tensor): Original logits from the model
            generated_tokens (List[int]): Previously generated tokens
            penalty (float): Penalty factor (>1.0 = less repetition)
        
        Returns:
            torch.Tensor: Modified logits with repetition penalty applied
        """
        if penalty == 1.0 or len(generated_tokens) == 0:
            return logits
        
        # Create a copy of logits to modify
        penalized_logits = logits.clone()
        
        # Apply penalty to tokens that have already been generated
        for token_id in set(generated_tokens):  # Use set to avoid duplicate penalties
            if penalized_logits[token_id] > 0:
                # If logit is positive, divide by penalty (reduce probability)
                penalized_logits[token_id] = penalized_logits[token_id] / penalty
            else:
                # If logit is negative, multiply by penalty (reduce probability further)
                penalized_logits[token_id] = penalized_logits[token_id] * penalty
        
        return penalized_logits
    
    def _apply_top_k_filtering(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        """
        Apply top-k filtering to limit sampling to the k most probable tokens.
        
        Top-k sampling helps balance creativity and coherence by only considering
        the k most likely next tokens. This prevents the model from selecting
        very unlikely tokens that might lead to incoherent text.
        
        Args:
            logits (torch.Tensor): Input logits
            k (int): Number of top tokens to keep
        
        Returns:
            torch.Tensor: Filtered logits with only top-k tokens
        """
        if k >= logits.size(-1):
            # If k is larger than vocabulary, no filtering needed
            return logits
        
        # Get the k-th largest value
        top_k_values, _ = torch.topk(logits, k)
        threshold = top_k_values[-1]
        
        # Set all values below threshold to negative infinity
        # This effectively removes them from the probability distribution
        filtered_logits = logits.clone()
        filtered_logits[logits < threshold] = float('-inf')
        
        return filtered_logits
    
    def _apply_top_p_filtering(self, logits: torch.Tensor, p: float) -> torch.Tensor:
        """
        Apply top-p (nucleus) filtering to sample from tokens with cumulative probability p.
        
        Top-p sampling dynamically adjusts the number of tokens considered based
        on their cumulative probability. This allows for more tokens when the
        distribution is flat and fewer tokens when there's a clear best choice.
        
        Args:
            logits (torch.Tensor): Input logits
            p (float): Cumulative probability threshold (0.0-1.0)
        
        Returns:
            torch.Tensor: Filtered logits with nucleus sampling applied
        """
        if p >= 1.0:
            # If p is 1.0 or higher, no filtering needed
            return logits
        
        # Convert logits to probabilities and sort in descending order
        probabilities = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
        
        # Calculate cumulative probabilities
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Find tokens to remove (those beyond the cumulative threshold)
        # Keep at least one token to avoid empty distribution
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[0] = False  # Always keep the most probable token
        
        # Create mask for original indices
        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
        indices_to_remove[sorted_indices[sorted_indices_to_remove]] = True
        
        # Set filtered tokens to negative infinity
        filtered_logits = logits.clone()
        filtered_logits[indices_to_remove] = float('-inf')
        
        return filtered_logits
    
    def _should_stop(self, generated_tokens: List[int], stop_tokens: List[str]) -> bool:
        """
        Check if generation should stop based on stop tokens.
        
        Args:
            generated_tokens (List[int]): Currently generated token sequence
            stop_tokens (List[str]): List of stop token strings
        
        Returns:
            bool: True if generation should stop, False otherwise
        """
        if not stop_tokens:
            return False
        
        # Convert recent tokens back to text for checking
        # Only check the last few tokens to avoid expensive string operations
        check_length = min(len(generated_tokens), max(len(token) for token in stop_tokens) + 10)
        recent_text = self.tokenizer.decode(generated_tokens[-check_length:])
        
        # Check if any stop token appears at the end of generated text
        for stop_token in stop_tokens:
            if recent_text.endswith(stop_token):
                return True
        
        return False
    
    def generate_greedy(self, prompt: str, length: int) -> str:
        """
        Generate text using greedy decoding (deterministic).
        
        Greedy decoding always selects the most probable next character,
        resulting in deterministic and often coherent text. This is useful
        when you want consistent, predictable output.
        
        Args:
            prompt (str): Starting text for generation
            length (int): Number of characters to generate
        
        Returns:
            str: Generated text using greedy decoding
        
        Example:
            >>> text = generator.generate_greedy("The cat", length=30)
            >>> print(text)  # Deterministic output
        """
        return self.generate(prompt, length, temperature=0.0)
    
    def generate_with_temperature(self, prompt: str, length: int, temperature: float) -> str:
        """
        Generate text using temperature-based sampling.
        
        Temperature sampling introduces controlled randomness:
        - Low temperature (0.1-0.5): More focused, coherent text
        - Medium temperature (0.6-0.9): Balanced creativity and coherence
        - High temperature (1.0+): More creative but potentially less coherent
        
        Args:
            prompt (str): Starting text for generation
            length (int): Number of characters to generate
            temperature (float): Sampling temperature (higher = more random)
        
        Returns:
            str: Generated text with temperature sampling
        
        Example:
            >>> # Conservative generation
            >>> text = generator.generate_with_temperature("Hello", 50, temperature=0.3)
            >>> 
            >>> # Creative generation
            >>> text = generator.generate_with_temperature("Hello", 50, temperature=1.2)
        """
        return self.generate(prompt, length, temperature=temperature)
    
    def generate_interactive(
        self, 
        initial_prompt: str = "", 
        max_length: int = 1000,
        temperature: float = 0.8
    ) -> str:
        """
        Interactive text generation that allows user input during generation.
        
        This method provides an interactive generation experience where users
        can guide the generation process by providing additional input or
        stopping generation at any point.
        
        Args:
            initial_prompt (str): Starting prompt for generation
            max_length (int): Maximum length of generated text
            temperature (float): Sampling temperature
        
        Returns:
            str: Final generated text
        
        Note:
            This method is designed for interactive use and may not be suitable
            for automated generation pipelines.
        """
        print("Interactive Text Generation")
        print("Commands: 'stop' to end, 'continue' to generate more, or provide new prompt")
        print("-" * 50)
        
        current_text = initial_prompt
        print(f"Starting with: '{current_text}'")
        
        while len(current_text) < max_length:
            user_input = input("\nEnter command or new text (or press Enter to continue): ").strip()
            
            if user_input.lower() == 'stop':
                break
            elif user_input.lower() == 'continue' or user_input == '':
                # Generate more text
                chunk_length = min(50, max_length - len(current_text))
                if chunk_length <= 0:
                    print("Maximum length reached.")
                    break
                
                new_text = self.generate(current_text, chunk_length, temperature=temperature)
                current_text = new_text
                print(f"Generated: '{current_text[-chunk_length:]}'")
            else:
                # User provided new prompt/continuation
                current_text += user_input
                print(f"Added: '{user_input}'")
        
        print(f"\nFinal text ({len(current_text)} characters):")
        print("-" * 50)
        print(current_text)
        return current_text
    
    def get_generation_info(self) -> Dict[str, Any]:
        """
        Get information about the generator and its capabilities.
        
        Returns:
            Dict[str, Any]: Dictionary containing generator information
        """
        return {
            'model_info': self.model.get_model_info(),
            'vocab_size': self.tokenizer.vocab_size(),
            'device': str(self.device),
            'supported_strategies': [
                'greedy_decoding',
                'temperature_sampling',
                'top_k_sampling',
                'top_p_sampling',
                'repetition_penalty'
            ],
            'features': [
                'prompt_processing',
                'length_control',
                'stop_tokens',
                'interactive_generation',
                'reproducible_generation'
            ]
        }
    
    def __repr__(self) -> str:
        """String representation of the generator."""
        return (f"TextGenerator(model={self.model.__class__.__name__}, "
                f"vocab_size={self.tokenizer.vocab_size()}, "
                f"device={self.device})")