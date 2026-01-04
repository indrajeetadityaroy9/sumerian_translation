"""
Hardware detection and setup utilities.

Provides unified GPU detection, precision configuration, and device setup
across all training scripts.
"""

import os
from typing import Dict, Any, Optional

import torch


def get_hardware_info() -> Dict[str, Any]:
    """
    Detect and report available hardware capabilities.

    Returns:
        Dictionary with:
        - cuda_available: bool
        - gpu_count: int
        - bf16_supported: bool (compute capability >= 8.0)
        - flash_attn_available: bool
        - gpu_names: List[str]
        - gpu_memory: List[float] (in GB)
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "bf16_supported": False,
        "flash_attn_available": False,
        "gpu_names": [],
        "gpu_memory": [],
    }

    if info["cuda_available"]:
        # Check compute capability for BF16 support (SM >= 8.0)
        capability = torch.cuda.get_device_capability()
        info["bf16_supported"] = capability[0] >= 8

        # Check Flash Attention 2 availability
        try:
            from transformers.utils import is_flash_attn_2_available
            info["flash_attn_available"] = is_flash_attn_2_available()
        except ImportError:
            info["flash_attn_available"] = False

        # Collect GPU info
        info["gpu_names"] = [
            torch.cuda.get_device_name(i)
            for i in range(info["gpu_count"])
        ]
        info["gpu_memory"] = [
            torch.cuda.get_device_properties(i).total_memory / 1e9
            for i in range(info["gpu_count"])
        ]

    return info


def setup_device(
    local_rank: Optional[int] = None,
    verbose: bool = True
) -> torch.device:
    """
    Set up the appropriate device for training.

    Args:
        local_rank: Local rank for DDP (from env or torchrun)
        verbose: Print device info

    Returns:
        torch.device for training
    """
    # Check for DDP environment
    if local_rank is None:
        local_rank = int(os.environ.get("LOCAL_RANK", -1))

    if local_rank >= 0:
        # DDP mode
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        if verbose and local_rank == 0:
            print(f"  DDP mode: using GPU {local_rank}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            print(f"  Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        if verbose:
            print("  Using CPU")

    return device


def print_hardware_summary(hw_info: Dict[str, Any]) -> None:
    """
    Print a formatted hardware summary.

    Args:
        hw_info: Output from get_hardware_info()
    """
    print("\n[Hardware Detection]")
    print(f"  CUDA available: {hw_info['cuda_available']}")

    if hw_info['cuda_available']:
        print(f"  GPU count: {hw_info['gpu_count']}")
        for i, (name, mem) in enumerate(zip(
            hw_info['gpu_names'],
            hw_info['gpu_memory']
        )):
            print(f"  GPU {i}: {name} ({mem:.1f} GB)")
        print(f"  BF16 supported: {hw_info['bf16_supported']}")
        print(f"  Flash Attention 2: {hw_info['flash_attn_available']}")


def is_main_process(local_rank: Optional[int] = None) -> bool:
    """
    Check if this is the main process in DDP training.

    Args:
        local_rank: Local rank (auto-detected if None)

    Returns:
        True if main process (rank 0 or non-DDP)
    """
    if local_rank is None:
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
    return local_rank <= 0
