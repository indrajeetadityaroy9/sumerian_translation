"""
Unwrap DDP Model Checkpoint

Converts a DDP checkpoint (with module. prefix) to standard HuggingFace format.

Note: train_mt5_ddp.py already saves unwrapped models via model.module.save_pretrained(),
so this script is only needed if you manually saved the raw state_dict.

Usage:
    python unwrap_model.py --input ./models/sumerian_mt5_ddp --output ./models/sumerian_mt5_production
"""
import argparse
from pathlib import Path
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Unwrap DDP model checkpoint")
    parser.add_argument("--input", type=str, default="./models/sumerian_mt5_ddp",
                        help="Path to DDP checkpoint directory")
    parser.add_argument("--output", type=str, default="./models/sumerian_mt5_production",
                        help="Output path for unwrapped model")
    parser.add_argument("--base-model", type=str, default="google/mt5-small",
                        help="Base model architecture")
    args = parser.parse_args()

    input_path = Path(args.input)

    # Check if this is already a standard HF checkpoint
    if (input_path / "config.json").exists():
        print(f"Found config.json - this appears to be a standard HuggingFace checkpoint.")
        print("Loading and re-saving to ensure compatibility...")

        model = AutoModelForSeq2SeqLM.from_pretrained(args.input)
        tokenizer = AutoTokenizer.from_pretrained(args.input)

        print(f"Saving to {args.output}...")
        model.save_pretrained(args.output)
        tokenizer.save_pretrained(args.output)
        print("Done!")
        return

    # Handle raw pytorch_model.bin with module. prefix
    checkpoint_path = input_path / "pytorch_model.bin"
    if not checkpoint_path.exists():
        print(f"ERROR: No checkpoint found at {checkpoint_path}")
        return

    print(f"Loading DDP checkpoint from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    # Check if unwrapping is needed
    needs_unwrap = any(key.startswith("module.") for key in state_dict.keys())

    if needs_unwrap:
        print("Removing 'module.' prefix from state dict keys...")
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("module.", "")
            new_state_dict[new_key] = value
        state_dict = new_state_dict
    else:
        print("Checkpoint already unwrapped (no 'module.' prefix found)")

    # Load base model and apply weights
    print(f"Loading base model architecture: {args.base_model}")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)
    model.load_state_dict(state_dict)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # Save
    print(f"Saving standard HuggingFace model to {args.output}...")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    print("\nDone! You can now use this model with translate.py:")
    print(f"  python translate.py --model {args.output}")


if __name__ == "__main__":
    main()
