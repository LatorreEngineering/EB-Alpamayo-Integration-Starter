"""
Alpamayo Model Wrapper

Clean interface for loading and running inference with NVIDIA's Alpamayo 1
Vision-Language-Action (VLA) model for autonomous driving research.

This module provides:
- Automatic device and dtype management
- Quantization support for smaller GPUs
- Structured output formatting
- Error handling and logging

Example:
    >>> from src.alpamayo_wrapper import AlpamayoInference
    >>> model = AlpamayoInference("nvidia/Alpamayo-R1-10B", device="cuda")
    >>> result = model.predict(images=[camera_frame], prompt="Navigate safely")
    >>> print(result.trajectory)
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """
    Structured output from Alpamayo model inference.

    Attributes:
        trajectory: Predicted waypoints as (N, 2) array of (x, y) coordinates in meters
        reasoning_trace: Chain-of-thought reasoning steps as text
        confidence: Model confidence score (0-1), if available
        raw_output: Raw model output for advanced analysis
        metadata: Additional information (timestamp, model version, etc.)
    """

    trajectory: np.ndarray
    reasoning_trace: str
    confidence: Optional[float] = None
    raw_output: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self) -> None:
        """Validate trajectory shape."""
        if self.trajectory.ndim != 2 or self.trajectory.shape[1] != 2:
            raise ValueError(
                f"Trajectory must be (N, 2) array, got shape {self.trajectory.shape}"
            )


class AlpamayoInference:
    """
    High-level interface for NVIDIA Alpamayo 1 VLA model inference.

    Handles model loading, device management, and structured output formatting.
    Supports quantization for GPUs with limited VRAM.

    Attributes:
        model_name: HuggingFace model identifier
        device: Compute device ("cuda", "cpu", or specific GPU like "cuda:0")
        dtype: Model precision (torch.float32, torch.float16, torch.bfloat16)
        model: Loaded transformers model
        processor: Associated processor/tokenizer
    """

    def __init__(
        self,
        model_name: str = "nvidia/Alpamayo-R1-10B",
        device: Optional[str] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        trust_remote_code: bool = True,
        cache_dir: Optional[Path] = None,
    ) -> None:
        """
        Initialize Alpamayo model wrapper.

        Args:
            model_name: HuggingFace model identifier (default: nvidia/Alpamayo-R1-10B)
            device: Target device. If None, auto-selects CUDA if available
            dtype: Model precision. If None, uses bfloat16 on CUDA, float32 on CPU
            load_in_8bit: Enable 8-bit quantization (requires ~14GB VRAM)
            load_in_4bit: Enable 4-bit quantization (requires ~8GB VRAM, experimental)
            trust_remote_code: Allow execution of model's custom code
            cache_dir: Directory for model cache (default: HF_HOME)

        Raises:
            ValueError: If both load_in_8bit and load_in_4bit are True
            RuntimeError: If CUDA is not available when quantization is requested

        Example:
            >>> # Standard loading for GPUs with >=24GB VRAM
            >>> model = AlpamayoInference("nvidia/Alpamayo-R1-10B")
            >>>
            >>> # 8-bit quantization for smaller GPUs
            >>> model = AlpamayoInference(
            ...     "nvidia/Alpamayo-R1-10B",
            ...     load_in_8bit=True
            ... )
        """
        if load_in_8bit and load_in_4bit:
            raise ValueError("Cannot enable both 8-bit and 4-bit quantization")

        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Auto-detected device: {device}")

        # Validate CUDA for quantization
        if (load_in_8bit or load_in_4bit) and not torch.cuda.is_available():
            raise RuntimeError("Quantization requires CUDA-enabled GPU")

        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir

        # Set dtype
        if dtype is None:
            if device.startswith("cuda"):
                dtype = torch.bfloat16  # Optimal for modern GPUs
            else:
                dtype = torch.float32
        elif isinstance(dtype, str):
            dtype = getattr(torch, dtype)

        self.dtype = dtype

        # Configure quantization
        quantization_config = None
        if load_in_8bit or load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
                bnb_4bit_compute_dtype=torch.bfloat16 if load_in_4bit else None,
            )
            logger.info(
                f"Using {'8-bit' if load_in_8bit else '4-bit'} quantization"
            )

        # Load model and processor
        logger.info(f"Loading model: {model_name}")
        try:
            self.model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=self.dtype,
                device_map="auto" if quantization_config else None,
                quantization_config=quantization_config,
                trust_remote_code=trust_remote_code,
                cache_dir=cache_dir,
            )

            if not quantization_config:
                self.model.to(self.device)

            self.processor = AutoProcessor.from_pretrained(
                model_name, trust_remote_code=trust_remote_code, cache_dir=cache_dir
            )

            logger.info(f"Model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        # Set to evaluation mode
        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        images: Union[Image.Image, List[Image.Image]],
        prompt: str = "Navigate this scenario safely and efficiently.",
        max_reasoning_steps: int = 10,
        temperature: float = 0.7,
        max_new_tokens: int = 1024,
        **generation_kwargs: Any,
    ) -> InferenceResult:
        """
        Run inference on driving scenario.

        Args:
            images: Single image or list of images (multi-camera setup)
            prompt: Text instruction/query for the model
            max_reasoning_steps: Maximum chain-of-thought steps to generate
            temperature: Sampling temperature (0 = greedy, higher = more random)
            max_new_tokens: Maximum tokens to generate
            **generation_kwargs: Additional arguments for model.generate()

        Returns:
            InferenceResult with trajectory, reasoning trace, and metadata

        Raises:
            ValueError: If images list is empty
            RuntimeError: If inference fails

        Example:
            >>> from PIL import Image
            >>> img = Image.open("dashcam_frame.jpg")
            >>> result = model.predict(
            ...     images=img,
            ...     prompt="Turn left at the intersection",
            ...     temperature=0.5
            ... )
            >>> print(f"Predicted {len(result.trajectory)} waypoints")
        """
        # Normalize input to list
        if isinstance(images, Image.Image):
            images = [images]

        if not images:
            raise ValueError("At least one image must be provided")

        logger.info(
            f"Running inference with {len(images)} image(s), prompt: '{prompt[:50]}...'"
        )

        try:
            # Process inputs
            inputs = self.processor(
                images=images, text=prompt, return_tensors="pt", padding=True
            ).to(self.device)

            # Generate output
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                **generation_kwargs,
            )

            # Decode output
            generated_text = self.processor.batch_decode(
                outputs, skip_special_tokens=True
            )[0]

            # Parse output (model-specific parsing logic)
            trajectory, reasoning_trace, confidence = self._parse_model_output(
                generated_text, outputs
            )

            result = InferenceResult(
                trajectory=trajectory,
                reasoning_trace=reasoning_trace,
                confidence=confidence,
                raw_output={"generated_text": generated_text, "outputs": outputs},
                metadata={
                    "model_name": self.model_name,
                    "num_images": len(images),
                    "prompt": prompt,
                    "temperature": temperature,
                },
            )

            logger.info(f"Inference successful: {len(trajectory)} waypoints generated")
            return result

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise RuntimeError(f"Inference failed: {e}") from e

    def _parse_model_output(
        self, generated_text: str, raw_outputs: torch.Tensor
    ) -> Tuple[np.ndarray, str, Optional[float]]:
        """
        Parse model output into structured format.

        This is a placeholder implementation. Actual parsing depends on
        Alpamayo 1's specific output format.

        Args:
            generated_text: Decoded text from model
            raw_outputs: Raw model output tensors

        Returns:
            Tuple of (trajectory, reasoning_trace, confidence)
        """
        # TODO: Implement actual Alpamayo 1 output parsing
        # This is a placeholder that demonstrates the expected interface

        # Example parsing logic (replace with actual Alpamayo format)
        reasoning_trace = generated_text

        # Extract trajectory (placeholder: generate random waypoints)
        # In real implementation, this would parse from model output
        num_waypoints = 10
        trajectory = np.array(
            [[i * 2.0, np.sin(i * 0.5) * 1.0] for i in range(num_waypoints)]
        )

        # Confidence (placeholder)
        confidence = 0.85

        logger.warning(
            "Using placeholder trajectory parsing. "
            "Implement actual Alpamayo 1 format parsing."
        )

        return trajectory, reasoning_trace, confidence

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model configuration and system information.

        Returns:
            Dictionary with model metadata
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "dtype": str(self.dtype),
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "cuda_memory_allocated": (
                torch.cuda.memory_allocated(0) if torch.cuda.is_available() else None
            ),
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"AlpamayoInference(model='{self.model_name}', "
            f"device='{self.device}', dtype={self.dtype})"
        )


# Example usage
if __name__ == "__main__":
    # This example requires a GPU and HuggingFace authentication
    print("Alpamayo Inference Wrapper")
    print("-" * 50)

    # Create inference wrapper
    try:
        model = AlpamayoInference(
            model_name="nvidia/Alpamayo-R1-10B",
            device="cuda",
            dtype=torch.bfloat16,
        )

        print("\nModel Info:")
        info = model.get_model_info()
        for key, value in info.items():
            print(f"  {key}: {value}")

        # Example inference (requires actual image)
        # result = model.predict(
        #     images=Image.open("example.jpg"),
        #     prompt="Navigate to the left lane safely"
        # )
        # print(f"\nGenerated {len(result.trajectory)} waypoints")

    except Exception as e:
        print(f"\nError: {e}")
        print("Note: This example requires GPU and HuggingFace authentication")
