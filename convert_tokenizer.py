"""
Convert SentencePiece tokenizer to HuggingFace format.

This is required before running the training scripts, as HuggingFace
Transformers expects a specific directory structure.
"""

import json
import shutil
from pathlib import Path

# Paths
SP_MODEL = Path("output_training_v2_clean/tokenizer/sumerian.model")
SP_VOCAB = Path("output_training_v2_clean/tokenizer/sumerian.vocab")
OUTPUT_DIR = Path("output_training_v2_clean/tokenizer_hf")


def convert_sentencepiece_to_hf():
    """Convert SentencePiece model to HuggingFace tokenizer."""

    print("Converting SentencePiece tokenizer to HuggingFace format...")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Read vocab to build the Unigram vocabulary
    vocab_list = []
    with open(SP_VOCAB, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            token = parts[0]
            score = float(parts[1]) if len(parts) > 1 else 0.0
            vocab_list.append([token, score])

    print(f"Loaded {len(vocab_list)} tokens from vocabulary")

    # Build tokenizer.json in HuggingFace format
    # This is a Unigram tokenizer (SentencePiece default)

    # Get token to ID mapping
    vocab_dict = {token: idx for idx, (token, score) in enumerate(vocab_list)}

    # Ensure special tokens exist
    special_tokens = {
        "<s>": 0,
        "<pad>": 1,
        "</s>": 2,
        "<unk>": 3,
        "<mask>": 4,
    }

    # Add special tokens if not in vocab
    for token, default_id in special_tokens.items():
        if token not in vocab_dict:
            # Add to end of vocab
            new_id = len(vocab_list)
            vocab_list.append([token, 0.0])
            vocab_dict[token] = new_id
            print(f"  Added missing special token: {token} -> {new_id}")

    # Create the tokenizer.json structure
    tokenizer_json = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": vocab_dict.get("<s>", 0),
                "content": "<s>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True
            },
            {
                "id": vocab_dict.get("<pad>", 1),
                "content": "<pad>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True
            },
            {
                "id": vocab_dict.get("</s>", 2),
                "content": "</s>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True
            },
            {
                "id": vocab_dict.get("<unk>", 3),
                "content": "<unk>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True
            },
            {
                "id": vocab_dict.get("<mask>", 4),
                "content": "<mask>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True
            },
        ],
        "normalizer": {
            "type": "Sequence",
            "normalizers": [
                {"type": "Nmt"},
                {"type": "NFKC"},
                {"type": "Replace", "pattern": {"Regex": " {2,}"}, "content": " "}
            ]
        },
        "pre_tokenizer": {
            "type": "Metaspace",
            "replacement": "▁",
            "add_prefix_space": True
        },
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [
                {"SpecialToken": {"id": "<s>", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "</s>", "type_id": 0}}
            ],
            "pair": [
                {"SpecialToken": {"id": "<s>", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "</s>", "type_id": 0}},
                {"Sequence": {"id": "B", "type_id": 1}},
                {"SpecialToken": {"id": "</s>", "type_id": 1}}
            ],
            "special_tokens": {
                "<s>": {"id": "<s>", "ids": [vocab_dict.get("<s>", 0)], "tokens": ["<s>"]},
                "</s>": {"id": "</s>", "ids": [vocab_dict.get("</s>", 2)], "tokens": ["</s>"]}
            }
        },
        "decoder": {
            "type": "Metaspace",
            "replacement": "▁",
            "add_prefix_space": True
        },
        "model": {
            "type": "Unigram",
            "unk_id": vocab_dict.get("<unk>", 3),
            "vocab": vocab_list
        }
    }

    # Save tokenizer.json
    with open(OUTPUT_DIR / "tokenizer.json", "w", encoding="utf-8") as f:
        json.dump(tokenizer_json, f, indent=2, ensure_ascii=False)
    print(f"Saved tokenizer.json")

    # Save tokenizer_config.json
    tokenizer_config = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "mask_token": "<mask>",
        "model_max_length": 512,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "clean_up_tokenization_spaces": True
    }
    with open(OUTPUT_DIR / "tokenizer_config.json", "w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, indent=2)
    print(f"Saved tokenizer_config.json")

    # Save special_tokens_map.json
    special_tokens_map = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "mask_token": "<mask>",
    }
    with open(OUTPUT_DIR / "special_tokens_map.json", "w", encoding="utf-8") as f:
        json.dump(special_tokens_map, f, indent=2)
    print(f"Saved special_tokens_map.json")

    # Copy original SP model for reference
    shutil.copy(SP_MODEL, OUTPUT_DIR / "sentencepiece.bpe.model")
    print(f"Copied original SentencePiece model")

    print(f"\nTokenizer saved to {OUTPUT_DIR}")

    # Test the tokenizer
    print("\nTesting tokenizer...")
    try:
        from transformers import PreTrainedTokenizerFast

        tokenizer = PreTrainedTokenizerFast.from_pretrained(OUTPUT_DIR)
        print(f"  Vocab size: {tokenizer.vocab_size}")
        print(f"  Pad token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")
        print(f"  Mask token: {tokenizer.mask_token} (id={tokenizer.mask_token_id})")

        # Test tokenization
        test_texts = [
            "lugal-e e2-gal-la-na ba-gen",
            "dingir-re-e-ne an-ki-a",
            "{{DET_DIVINE}}en-ki lugal abzu-ka",
        ]
        for text in test_texts:
            tokens = tokenizer.tokenize(text)
            ids = tokenizer.encode(text)
            print(f"\n  Input: '{text}'")
            print(f"  Tokens: {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
            print(f"  IDs: {ids[:10]}{'...' if len(ids) > 10 else ''}")

        return tokenizer

    except Exception as e:
        print(f"  Warning: Could not load tokenizer for testing: {e}")
        print("  The tokenizer files have been created, training may still work.")
        return None


if __name__ == "__main__":
    convert_sentencepiece_to_hf()
