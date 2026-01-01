"""
Sumerian-English Translation Inference

OPTIMIZED FOR: 2x NVIDIA H100 80GB + Intel Xeon Platinum 8480+

Test the trained translation model on Sumerian text.

Optimizations enabled:
- BF16 inference (native H100 support)
- Flash Attention 2 (2-4x faster attention)
- torch.compile() (kernel fusion)
- Batched inference for file translation
- Multi-GPU support

Usage:
    python translate.py "lugal-e e2-gal-la-na ba-gen"
    python translate.py --interactive
    python translate.py --file input.txt
    python translate.py --file input.txt --batch-size 32
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import (
    EncoderDecoderModel,
    AutoModelForSeq2SeqLM,
    PreTrainedTokenizerFast,
    BertTokenizerFast,
    AutoTokenizer,
)

# Default model path
MODEL_DIR = Path("./models/sumerian_nmt_final")


def get_hardware_info():
    """Detect available hardware."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "bf16_supported": False,
        "flash_attn_available": False,
    }

    if info["cuda_available"]:
        capability = torch.cuda.get_device_capability()
        info["bf16_supported"] = capability[0] >= 8

        try:
            from transformers.utils import is_flash_attn_2_available
            info["flash_attn_available"] = is_flash_attn_2_available()
        except ImportError:
            pass

        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory / 1e9

    return info


def load_model(model_dir: Path, use_compile: bool = True, verbose: bool = True):
    """Load the trained translation model and tokenizers with optimizations."""
    hw = get_hardware_info()

    if verbose:
        print(f"Loading model from {model_dir}...")
        if hw["cuda_available"]:
            print(f"  GPU: {hw['gpu_name']} ({hw['gpu_memory']:.1f} GB)")
            print(f"  BF16: {hw['bf16_supported']}")
            print(f"  Flash Attention 2: {hw['flash_attn_available']}")

    # Determine model type
    config_path = model_dir / "config.json"
    is_mt5 = False
    if config_path.exists():
        import json
        with open(config_path) as f:
            config = json.load(f)
            is_mt5 = "mt5" in config.get("_name_or_path", "").lower() or \
                     config.get("model_type", "") == "mt5"

    # Load with Flash Attention if available
    attn_impl = "flash_attention_2" if hw["flash_attn_available"] else None

    if is_mt5:
        # mT5 model
        if attn_impl:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_dir,
                attn_implementation=attn_impl,
                torch_dtype=torch.bfloat16 if hw["bf16_supported"] else torch.float32
            )
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_dir,
                torch_dtype=torch.bfloat16 if hw["bf16_supported"] else torch.float32
            )
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        tokenizer_src = tokenizer
        tokenizer_tgt = tokenizer
        model_type = "mt5"
    else:
        # Encoder-Decoder model
        if attn_impl:
            model = EncoderDecoderModel.from_pretrained(
                model_dir,
                attn_implementation=attn_impl,
                torch_dtype=torch.bfloat16 if hw["bf16_supported"] else torch.float32
            )
        else:
            model = EncoderDecoderModel.from_pretrained(
                model_dir,
                torch_dtype=torch.bfloat16 if hw["bf16_supported"] else torch.float32
            )
        tokenizer_src = PreTrainedTokenizerFast.from_pretrained(model_dir / "tokenizer_src")
        tokenizer_tgt_path = model_dir / "tokenizer_tgt"
        if tokenizer_tgt_path.exists():
            tokenizer_tgt = AutoTokenizer.from_pretrained(tokenizer_tgt_path)
        else:
            tokenizer_tgt = BertTokenizerFast.from_pretrained("bert-base-uncased")
        model_type = "encoder_decoder"

    # Move to GPU
    device = torch.device("cuda" if hw["cuda_available"] else "cpu")
    model = model.to(device)
    model.eval()

    # Apply torch.compile() for faster inference
    if use_compile and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            if verbose:
                print("  torch.compile(): enabled")
        except Exception as e:
            if verbose:
                print(f"  torch.compile() skipped: {e}")

    if verbose:
        print(f"  Model type: {model_type}")
        print(f"  Device: {device}")
        if hw["bf16_supported"]:
            print("  Precision: BF16")

    return model, tokenizer_src, tokenizer_tgt, device, model_type


def translate(
    text: str,
    model,
    tokenizer_src,
    tokenizer_tgt,
    device,
    model_type: str = "encoder_decoder",
    max_length: int = 128,
    num_beams: int = 4,
    task_prefix: str = "translate Sumerian to English: ",
) -> str:
    """Translate Sumerian text to English."""
    # Prepare input
    if model_type == "mt5":
        input_text = task_prefix + text
        inputs = tokenizer_src(
            input_text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True,
        )
    else:
        inputs = tokenizer_src(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True,
        )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate translation
    with torch.no_grad():
        if model_type == "mt5":
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
            )
        else:
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                decoder_start_token_id=tokenizer_tgt.cls_token_id if hasattr(tokenizer_tgt, 'cls_token_id') else None,
                eos_token_id=tokenizer_tgt.sep_token_id if hasattr(tokenizer_tgt, 'sep_token_id') else tokenizer_tgt.eos_token_id,
                pad_token_id=tokenizer_tgt.pad_token_id,
            )

    # Decode output
    translation = tokenizer_tgt.decode(outputs[0], skip_special_tokens=True)
    return translation


def translate_batch(
    texts: list,
    model,
    tokenizer_src,
    tokenizer_tgt,
    device,
    model_type: str = "encoder_decoder",
    max_length: int = 128,
    num_beams: int = 4,
    task_prefix: str = "translate Sumerian to English: ",
) -> list:
    """Translate a batch of Sumerian texts to English."""
    # Prepare inputs
    if model_type == "mt5":
        input_texts = [task_prefix + text for text in texts]
        inputs = tokenizer_src(
            input_texts,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True,
        )
    else:
        inputs = tokenizer_src(
            texts,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True,
        )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate translations
    with torch.no_grad():
        if model_type == "mt5":
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
            )
        else:
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                decoder_start_token_id=tokenizer_tgt.cls_token_id if hasattr(tokenizer_tgt, 'cls_token_id') else None,
                eos_token_id=tokenizer_tgt.sep_token_id if hasattr(tokenizer_tgt, 'sep_token_id') else tokenizer_tgt.eos_token_id,
                pad_token_id=tokenizer_tgt.pad_token_id,
            )

    # Decode outputs
    translations = tokenizer_tgt.batch_decode(outputs, skip_special_tokens=True)
    return translations


def interactive_mode(model, tokenizer_src, tokenizer_tgt, device, model_type):
    """Interactive translation mode."""
    print("\n" + "=" * 60)
    print("Sumerian-English Interactive Translator (H100 Optimized)")
    print("=" * 60)
    print("Enter Sumerian text to translate (or 'quit' to exit)")
    print("-" * 60)

    while True:
        try:
            text = input("\nSumerian> ").strip()
            if not text:
                continue
            if text.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            translation = translate(text, model, tokenizer_src, tokenizer_tgt, device, model_type)
            print(f"English > {translation}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def translate_file(
    input_path: Path,
    output_path: Path,
    model,
    tokenizer_src,
    tokenizer_tgt,
    device,
    model_type: str,
    batch_size: int = 32,
):
    """Translate a file of Sumerian text with batched inference."""
    print(f"Translating {input_path}...")
    print(f"  Batch size: {batch_size}")

    # Read all lines
    with open(input_path, encoding="utf-8") as f:
        lines = [line.strip() for line in f]

    print(f"  Total lines: {len(lines)}")

    # Process in batches
    translations = []
    for i in range(0, len(lines), batch_size):
        batch = lines[i:i + batch_size]

        # Handle empty lines
        non_empty_indices = [j for j, line in enumerate(batch) if line]
        non_empty_texts = [batch[j] for j in non_empty_indices]

        if non_empty_texts:
            batch_translations = translate_batch(
                non_empty_texts, model, tokenizer_src, tokenizer_tgt, device, model_type
            )

            # Reconstruct with empty lines
            result = [""] * len(batch)
            for j, trans in zip(non_empty_indices, batch_translations):
                result[j] = trans
            translations.extend(result)
        else:
            translations.extend([""] * len(batch))

        if (i + batch_size) % 100 == 0 or i + batch_size >= len(lines):
            print(f"  Translated {min(i + batch_size, len(lines))}/{len(lines)} lines...")

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        for translation in translations:
            f.write(f"{translation}\n")

    print(f"Output saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Translate Sumerian text to English (H100 Optimized)"
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="Sumerian text to translate"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=MODEL_DIR,
        help=f"Model directory (default: {MODEL_DIR})"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Interactive translation mode"
    )
    parser.add_argument(
        "--file", "-f",
        type=Path,
        help="Input file to translate"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output file for translations"
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=4,
        help="Number of beams for beam search (default: 4)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for file translation (default: 32)"
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile()"
    )

    args = parser.parse_args()

    # Check model exists
    if not args.model_dir.exists():
        print(f"ERROR: Model not found at {args.model_dir}")
        print("Run 'python train_mlm.py' and 'python train_nmt.py' first.")
        sys.exit(1)

    # Load model
    model, tokenizer_src, tokenizer_tgt, device, model_type = load_model(
        args.model_dir,
        use_compile=not args.no_compile
    )

    # Determine mode
    if args.interactive:
        interactive_mode(model, tokenizer_src, tokenizer_tgt, device, model_type)
    elif args.file:
        output_path = args.output or args.file.with_suffix(".en.txt")
        translate_file(
            args.file, output_path, model, tokenizer_src, tokenizer_tgt,
            device, model_type, batch_size=args.batch_size
        )
    elif args.text:
        translation = translate(
            args.text, model, tokenizer_src, tokenizer_tgt, device, model_type,
            num_beams=args.num_beams
        )
        print(f"Sumerian: {args.text}")
        print(f"English:  {translation}")
    else:
        # Default to interactive if no arguments
        interactive_mode(model, tokenizer_src, tokenizer_tgt, device, model_type)


# Example translations to test
EXAMPLES = [
    "lugal-e e2-gal-la-na ba-gen",
    "dingir-re-e-ne an-ki-a",
    "nam-lu2-ulu3",
]

if __name__ == "__main__":
    main()
