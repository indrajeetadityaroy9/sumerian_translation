"""
vLLM-based Inference for High-Throughput Evaluation.

Provides 10-100x throughput improvement over sequential HuggingFace generation
by leveraging vLLM's continuous batching and tensor parallelism across dual H100s.

Usage:
    from evaluation.vllm_inference import VLLMInference

    # Initialize with tensor parallelism for dual H100
    engine = VLLMInference("models_llm/sumerian_llama3_sft", tensor_parallel_size=2)

    # Generate translations in batch
    translations = engine.translate_batch(sumerian_texts)

Requirements:
    pip install vllm>=0.4.0
"""

from pathlib import Path
from typing import List, Optional, Union

from sumerian_nmt.config import LLMConfig


class VLLMInference:
    """
    High-throughput inference engine using vLLM.

    Optimized for:
    - Dual H100 80GB (tensor_parallel_size=2)
    - Continuous batching for maximum throughput
    - BFloat16 precision for H100 optimization
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        tensor_parallel_size: int = 2,
        max_model_len: int = 512,
        gpu_memory_utilization: float = 0.85,
        dtype: str = "bfloat16",
        trust_remote_code: bool = True,
        seed: int = 42,
    ):
        """
        Initialize vLLM inference engine.

        Args:
            model_path: Path to fine-tuned model or HuggingFace model ID
            tensor_parallel_size: Number of GPUs for tensor parallelism (2 for dual H100)
            max_model_len: Maximum sequence length
            gpu_memory_utilization: Fraction of GPU memory to use (0.85 recommended)
            dtype: Model dtype ("bfloat16" for H100, "float16" for older GPUs)
            trust_remote_code: Whether to trust remote code in model
            seed: Random seed for reproducibility
        """
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError(
                "vLLM not installed. Install with: pip install vllm>=0.4.0\n"
                "Or install the optional dependency: pip install -e '.[vllm]'"
            )

        self._SamplingParams = SamplingParams

        print(f"Initializing vLLM with tensor_parallel_size={tensor_parallel_size}...")
        print(f"  Model: {model_path}")
        print(f"  Max sequence length: {max_model_len}")
        print(f"  GPU memory utilization: {gpu_memory_utilization:.0%}")
        print(f"  Dtype: {dtype}")

        self.model = LLM(
            model=str(model_path),
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            seed=seed,
        )

        # Default sampling parameters matching LLMConfig
        self.default_sampling_params = SamplingParams(
            max_tokens=LLMConfig.MAX_NEW_TOKENS,
            temperature=LLMConfig.TEMPERATURE,
            top_p=LLMConfig.TOP_P,
            stop=["<|eot_id|>", "<|end_of_text|>"],
        )

        print("vLLM engine initialized successfully.")

    def generate_batch(
        self,
        prompts: List[str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> List[str]:
        """
        Generate completions for a batch of prompts.

        Uses vLLM's continuous batching for maximum throughput.

        Args:
            prompts: List of formatted prompts
            max_tokens: Maximum tokens to generate (default: LLMConfig.MAX_NEW_TOKENS)
            temperature: Sampling temperature (default: LLMConfig.TEMPERATURE)
            top_p: Top-p sampling (default: LLMConfig.TOP_P)

        Returns:
            List of generated text completions
        """
        # Create sampling params with overrides
        params = self._SamplingParams(
            max_tokens=max_tokens or LLMConfig.MAX_NEW_TOKENS,
            temperature=temperature if temperature is not None else LLMConfig.TEMPERATURE,
            top_p=top_p or LLMConfig.TOP_P,
            stop=["<|eot_id|>", "<|end_of_text|>"],
        )

        # Generate with vLLM's continuous batching
        outputs = self.model.generate(prompts, params)

        # Extract generated text from outputs
        results = []
        for output in outputs:
            # vLLM returns RequestOutput with outputs list
            if output.outputs:
                text = output.outputs[0].text.strip()
                results.append(text)
            else:
                results.append("")

        return results

    def translate_batch(
        self,
        inputs: List[str],
        instruction: str = "Translate this Sumerian text into English:",
    ) -> List[str]:
        """
        Translate a batch of Sumerian texts to English.

        Args:
            inputs: List of Sumerian text inputs
            instruction: Translation instruction (default matches SFT training)

        Returns:
            List of English translations
        """
        # Format prompts using LLMConfig for consistency with training
        prompts = [LLMConfig.format_prompt(instruction, input_text) for input_text in inputs]

        return self.generate_batch(prompts)

    def translate_single(
        self,
        input_text: str,
        instruction: str = "Translate this Sumerian text into English:",
    ) -> str:
        """
        Translate a single Sumerian text.

        Args:
            input_text: Sumerian text to translate
            instruction: Translation instruction

        Returns:
            English translation
        """
        translations = self.translate_batch([input_text], instruction)
        return translations[0] if translations else ""


def create_vllm_engine(
    model_path: Optional[Union[str, Path]] = None,
    tensor_parallel_size: int = 2,
) -> VLLMInference:
    """
    Factory function to create vLLM inference engine.

    Args:
        model_path: Path to model (default: auto-detect best checkpoint)
        tensor_parallel_size: Number of GPUs (default: 2 for dual H100)

    Returns:
        Configured VLLMInference instance
    """
    if model_path is None:
        from config import get_llm_checkpoint

        model_path = get_llm_checkpoint()

    return VLLMInference(
        model_path=model_path,
        tensor_parallel_size=tensor_parallel_size,
    )


def main():
    """Test vLLM inference."""
    import argparse

    parser = argparse.ArgumentParser(description="Test vLLM inference")
    parser.add_argument(
        "--model",
        type=str,
        default="models_llm/sumerian_llama3_sft",
        help="Path to model",
    )
    parser.add_argument(
        "--tensor-parallel",
        type=int,
        default=2,
        help="Tensor parallel size (GPUs)",
    )
    args = parser.parse_args()

    # Initialize engine
    engine = VLLMInference(
        model_path=args.model,
        tensor_parallel_size=args.tensor_parallel,
    )

    # Test translations
    test_inputs = [
        "lugal-e e2-gal-la-na ba-gen",
        "dingir gal-gal-e-ne",
        "en-lil2 nibru{ki}-a",
    ]

    print("\nTest translations:")
    print("=" * 60)

    translations = engine.translate_batch(test_inputs)

    for inp, trans in zip(test_inputs, translations):
        print(f"Input:  {inp}")
        print(f"Output: {trans}")
        print("-" * 40)


if __name__ == "__main__":
    main()
