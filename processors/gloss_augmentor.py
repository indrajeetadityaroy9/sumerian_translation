"""
Gloss Substitution & Dictionary Augmentation

Expands ~5,000 parallel translation pairs to ~20,000+ via lexical substitution
using the ORACC glossary. Critical for neural machine translation with limited data.

Key Safety Checks:
1. English Anchor Verification - ABORT if gw not found in target
2. Transitivity Matching - V/t <-> V/t, V/i <-> V/i only
3. Dynamic Grouping - group by exact (POS, gw), no hardcoded keywords
4. Lemma Normalization - ETCSL uses c/j, ORACC uses š/g
"""

import argparse
import copy
import json
import random
import re
from collections import defaultdict
from pathlib import Path


def normalize_lemma(lemma: str) -> str:
    """
    Normalize ETCSL lemma to match ORACC glossary conventions.

    ETCSL uses:
    - ASCII subscripts: dug4, lu2, e2
    - c for shin: cu -> šu
    - j for eng: jar -> gar

    ORACC glossary uses:
    - No subscripts in cf: dug, lu, e
    - š for shin: šu
    - Plain g for eng: gar (NOT ĝar)

    Examples:
        dug4 -> dug
        cu   -> šu
        jal2 -> gal
        saj  -> sag
    """
    # Remove trailing digits (e.g., dug4 -> dug)
    result = re.sub(r'\d+$', '', lemma)
    # Convert c -> š (shin)
    result = result.replace('c', 'š').replace('C', 'Š')
    # Convert j -> g (eng - glossary uses plain g)
    result = result.replace('j', 'g').replace('J', 'G')
    return result


def load_glossary(glossary_paths: list[Path]) -> dict:
    """
    Load and merge ORACC glossaries.

    Args:
        glossary_paths: List of paths to gloss-sux.json files

    Returns:
        {
            "entries": [...],  # All entries
            "by_lemma": {lemma: {cf, gw, pos, icount}},  # Lookup by lemma
            "by_normalized": {normalized_lemma: {cf, gw, pos, icount}},  # For ETCSL matching
        }
    """
    all_entries = []
    by_lemma = {}
    by_normalized = {}

    for path in glossary_paths:
        if not path.exists():
            print(f"Warning: Glossary not found: {path}")
            continue

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        entries = data.get("entries", [])
        all_entries.extend(entries)

        for entry in entries:
            cf = entry.get("cf", "")
            if not cf:
                continue

            entry_data = {
                "cf": cf,
                "gw": entry.get("gw", ""),
                "pos": entry.get("pos", ""),
                "icount": int(entry.get("icount", 0)),
            }

            if cf not in by_lemma:
                by_lemma[cf] = entry_data

            # Also index by normalized form for ETCSL matching
            # (This allows ETCSL lemma "dug4" to match glossary "dug")
            normalized = normalize_lemma(cf)
            if normalized not in by_normalized:
                by_normalized[normalized] = entry_data

    return {
        "entries": all_entries,
        "by_lemma": by_lemma,
        "by_normalized": by_normalized,
    }


def build_synonym_groups(glossary: dict, min_freq: int = 10) -> dict:
    """
    Dynamically discover synonym groups from glossary.
    Groups all lemmas by their exact (POS, gw) pair.
    Only keeps groups with 2+ members.

    POS Tag Handling: Treat POS strings as strict identifiers.
    V/t stays separate from V and V/i. No merging.

    Args:
        glossary: Loaded ORACC glossary
        min_freq: Minimum icount for reliable entries

    Returns:
        {("N", "water"): [{"cf": "a", "gw": "water", "icount": 907}, ...], ...}
    """
    groups = defaultdict(list)

    for entry in glossary.get("entries", []):
        pos = entry.get("pos", "").strip()
        gw = entry.get("gw", "").lower().strip()
        icount = int(entry.get("icount", 0))
        cf = entry.get("cf", "")

        # Skip entries without required fields or below frequency threshold
        if not (pos and gw and cf and icount >= min_freq):
            continue

        groups[(pos, gw)].append({
            "cf": cf,
            "gw": entry.get("gw", ""),  # Keep original case
            "icount": icount,
        })

    # Filter to groups with 2+ members (valid for substitution)
    valid_groups = {k: v for k, v in groups.items() if len(v) >= 2}

    # Sort each group by frequency (highest first) for better substitution
    for key in valid_groups:
        valid_groups[key] = sorted(
            valid_groups[key], key=lambda x: -x["icount"]
        )

    return valid_groups


def build_lemma_to_group(synonym_groups: dict) -> dict:
    """
    Build reverse lookup: lemma -> (pos, gw) group key.
    Includes both original and normalized forms for matching.

    Args:
        synonym_groups: Output of build_synonym_groups

    Returns:
        {
            "a": ("N", "water"),
            "abzu": ("N", "water"),
            "dug": ("V/t", "speak"),  # Normalized form
            ...
        }
    """
    lemma_to_group = {}
    for (pos, gw), members in synonym_groups.items():
        for member in members:
            cf = member["cf"]
            # Index by original form
            lemma_to_group[cf] = (pos, gw)
            # Also index by normalized form for ETCSL matching
            normalized = normalize_lemma(cf)
            if normalized not in lemma_to_group:
                lemma_to_group[normalized] = (pos, gw)
    return lemma_to_group


def validate_english_anchor(target_text: str, gw: str, fuzzy: bool = False) -> bool:
    """
    CRITICAL SAFETY CHECK: Verify English gw exists in target.

    If not found, augmentation must be ABORTED for this token.
    Prevents source/target mismatch.

    Args:
        target_text: English translation
        gw: Guide word to find
        fuzzy: If True, allow substring matches (e.g., "build" matches "building")

    Returns:
        True if gw found in target, False otherwise
    """
    if not gw:
        return False

    if fuzzy:
        # Fuzzy matching: allow substring matches
        # "build" matches "build", "builds", "building", "rebuilt"
        pattern = rf'{re.escape(gw)}'
    else:
        # Strict matching: require word boundaries
        pattern = rf'\b{re.escape(gw)}\b'

    return bool(re.search(pattern, target_text, flags=re.IGNORECASE))


def substitute_token(token: dict, new_lemma: str, new_gw: str) -> dict:
    """
    Replace lemma in token while preserving morphology.

    Example:
        token = {"form": "lugal-la", "lemma": "lugal", "label": "king"}
        new_lemma = "ensi2", new_gw = "governor"

        Returns: {"form": "ensi2-la", "lemma": "ensi2", "label": "governor"}

    Args:
        token: Original token dict
        new_lemma: New lemma to substitute
        new_gw: New guide word (for label)

    Returns:
        New token dict with substituted values
    """
    old_lemma = token.get("lemma", "")
    old_form = token.get("form", "")

    # Replace lemma in form (preserves morpheme suffixes)
    new_form = old_form.replace(old_lemma, new_lemma)

    # Create new token with substituted values
    new_token = copy.deepcopy(token)
    new_token["form"] = new_form
    new_token["lemma"] = new_lemma
    new_token["label"] = new_gw

    # Update form_normalized if present
    if "form_normalized" in new_token:
        new_token["form_normalized"] = new_form

    return new_token


def regenerate_text(tokens: list[dict]) -> str:
    """Rebuild text_normalized from token forms."""
    return " ".join(t.get("form", "") for t in tokens if t.get("form"))


def augment_pair(
    pair: dict,
    glossary: dict,
    synonym_groups: dict,
    lemma_to_group: dict,
    max_augments: int = 3,
    fuzzy_anchor: bool = False,
) -> list[dict]:
    """
    Generate augmented pairs via token substitution.

    Args:
        pair: Original training pair
        glossary: Loaded glossary
        synonym_groups: Synonym groups by (pos, gw)
        lemma_to_group: Lemma -> group lookup
        max_augments: Maximum augmented pairs per original
        fuzzy_anchor: Allow substring matches for English anchor check

    Returns:
        List of augmented pairs (excluding original)
    """
    augmented = []
    tokens = pair.get("source", {}).get("tokens", [])
    target_text = pair.get("target", {}).get("text", "")

    if not tokens or not target_text:
        return []

    # Find substitutable tokens
    substitutable = []
    for i, token in enumerate(tokens):
        lemma = token.get("lemma", "")
        if not lemma:
            continue

        # Normalize ETCSL lemma to match glossary (dug4 -> dug, cu -> šu)
        normalized_lemma = normalize_lemma(lemma)

        # Check if normalized lemma has a synonym group
        if normalized_lemma in lemma_to_group:
            group_key = lemma_to_group[normalized_lemma]
            gw = group_key[1]  # Get gw from (pos, gw) key

            # CRITICAL: English anchor check (with optional fuzzy matching)
            if validate_english_anchor(target_text, gw, fuzzy=fuzzy_anchor):
                substitutable.append((i, lemma, normalized_lemma, group_key))

    if not substitutable:
        return []

    # Randomly select tokens to substitute (up to max_augments)
    random.shuffle(substitutable)

    for idx, original_lemma, normalized_lemma, group_key in substitutable[:max_augments]:
        # Get synonyms from the group (excluding the current lemma)
        group_members = synonym_groups.get(group_key, [])

        # Exclude current lemma (check both original and normalized forms)
        synonyms = [
            m for m in group_members
            if m["cf"] != original_lemma and normalize_lemma(m["cf"]) != normalized_lemma
        ]

        if not synonyms:
            continue

        # Pick a random synonym (weighted by frequency could be an option)
        new_entry = random.choice(synonyms)
        new_lemma = new_entry["cf"]
        new_gw = new_entry["gw"]

        # Create augmented pair
        new_pair = copy.deepcopy(pair)

        # Substitute the token
        new_tokens = copy.deepcopy(tokens)
        new_tokens[idx] = substitute_token(tokens[idx], new_lemma, new_gw)

        # Update source
        new_pair["source"]["tokens"] = new_tokens
        new_pair["source"]["text_normalized"] = regenerate_text(new_tokens)

        # Mark as augmented
        new_pair["augmented"] = True
        new_pair["augmentation_info"] = {
            "original_lemma": original_lemma,
            "normalized_lemma": normalized_lemma,
            "new_lemma": new_lemma,
            "token_index": idx,
        }

        augmented.append(new_pair)

    return augmented


def augment_dataset(
    input_path: Path,
    output_path: Path,
    glossary: dict,
    synonym_groups: dict,
    lemma_to_group: dict,
    max_augments_per_pair: int = 3,
    fuzzy_anchor: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Augment training data via gloss substitution.

    Args:
        input_path: Path to train.jsonl
        output_path: Path to augmented output
        glossary: Loaded ORACC glossary
        synonym_groups: Synonym groups
        lemma_to_group: Lemma -> group lookup
        max_augments_per_pair: Cap on synthetic pairs per original
        fuzzy_anchor: Allow substring matches for English anchor check
        verbose: Print progress

    Returns:
        Statistics dict
    """
    stats = {
        "original_pairs": 0,
        "augmented_pairs": 0,
        "total_pairs": 0,
        "pairs_with_augmentation": 0,
        "tokens_substituted": 0,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:

        for line in f_in:
            pair = json.loads(line)
            stats["original_pairs"] += 1

            # Write original pair
            f_out.write(json.dumps(pair, ensure_ascii=False) + "\n")
            stats["total_pairs"] += 1

            # Generate augmented pairs
            augmented = augment_pair(
                pair, glossary, synonym_groups, lemma_to_group,
                max_augments=max_augments_per_pair,
                fuzzy_anchor=fuzzy_anchor,
            )

            if augmented:
                stats["pairs_with_augmentation"] += 1
                stats["augmented_pairs"] += len(augmented)
                stats["tokens_substituted"] += len(augmented)

                for aug_pair in augmented:
                    f_out.write(json.dumps(aug_pair, ensure_ascii=False) + "\n")
                    stats["total_pairs"] += 1

            if verbose and stats["original_pairs"] % 1000 == 0:
                print(f"  Processed {stats['original_pairs']} pairs, "
                      f"generated {stats['augmented_pairs']} augmented...")

    if verbose:
        expansion = stats["total_pairs"] / stats["original_pairs"]
        print(f"\nAugmentation complete:")
        print(f"  Original pairs: {stats['original_pairs']:,}")
        print(f"  Augmented pairs: {stats['augmented_pairs']:,}")
        print(f"  Total pairs: {stats['total_pairs']:,}")
        print(f"  Expansion factor: {expansion:.2f}x")
        print(f"  Pairs with augmentation: {stats['pairs_with_augmentation']:,}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Augment training data via gloss substitution"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("output_training_v2_clean/finetune/train.jsonl"),
        help="Input training data path"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output_training_v2_clean/finetune/train_augmented.jsonl"),
        help="Output augmented data path"
    )
    parser.add_argument(
        "--glossary",
        type=Path,
        nargs="+",
        default=[
            Path("oracc_data/epsd2-literary/epsd2/literary/gloss-sux.json"),
            Path("oracc_data/epsd2-royal/epsd2/royal/gloss-sux.json"),
        ],
        help="Glossary file paths"
    )
    parser.add_argument(
        "--max-per-pair",
        type=int,
        default=3,
        help="Maximum augmented pairs per original (default: 3)"
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=10,
        help="Minimum lemma frequency for substitution (default: 10)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--fuzzy",
        action="store_true",
        help="Use fuzzy English anchor matching (substring instead of word boundary)"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()
    random.seed(args.seed)
    verbose = not args.quiet

    # Load glossary
    if verbose:
        print("Loading glossary...")
    glossary = load_glossary(args.glossary)
    if verbose:
        print(f"  Loaded {len(glossary['entries']):,} entries")
        print(f"  Unique lemmas: {len(glossary['by_lemma']):,}")

    # Build synonym groups
    if verbose:
        print("\nBuilding synonym groups...")
    synonym_groups = build_synonym_groups(glossary, min_freq=args.min_frequency)
    lemma_to_group = build_lemma_to_group(synonym_groups)

    if verbose:
        print(f"  Found {len(synonym_groups):,} valid synonym groups")
        print(f"  Lemmas with synonyms: {len(lemma_to_group):,}")

        # Show some example groups
        print("\n  Example synonym groups:")
        for i, ((pos, gw), members) in enumerate(
            sorted(synonym_groups.items(), key=lambda x: -len(x[1]))[:5]
        ):
            lemmas = [m["cf"] for m in members[:4]]
            more = f"... (+{len(members)-4})" if len(members) > 4 else ""
            print(f"    ({pos}, {gw}): {lemmas}{more}")

    # Run augmentation
    if verbose:
        print(f"\nAugmenting dataset...")
        print(f"  Input: {args.input}")
        print(f"  Output: {args.output}")
        print(f"  Max augments per pair: {args.max_per_pair}")
        print(f"  Fuzzy anchor matching: {args.fuzzy}")

    stats = augment_dataset(
        args.input,
        args.output,
        glossary,
        synonym_groups,
        lemma_to_group,
        max_augments_per_pair=args.max_per_pair,
        fuzzy_anchor=args.fuzzy,
        verbose=verbose,
    )

    # Save stats
    stats_path = args.output.parent / "augmentation_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    if verbose:
        print(f"\nStats saved to: {stats_path}")


if __name__ == "__main__":
    main()
