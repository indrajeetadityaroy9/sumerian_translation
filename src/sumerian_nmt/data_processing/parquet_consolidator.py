"""
Consolidate ORACC/ETCSL datasets to efficient Parquet format.

Zero-Drift Approach:
- ETCSL is loaded from validated JSONL artifact (NOT re-parsed from XML)
- ORACC is parsed from raw JSON with catalogue metadata
- Explicit PyArrow schemas for type safety

Usage:
    # Consolidate ORACC from raw JSON
    python consolidate_to_parquet.py --oracc-dir data/oracc --output-dir data/consolidated

    # Consolidate ETCSL from validated JSONL
    python consolidate_to_parquet.py --etcsl-jsonl output/parallel_corpus.jsonl --output-dir data/consolidated

    # Create archive of raw sources
    python consolidate_to_parquet.py --archive-only --archive-path archives/raw_sources.tar.gz

    # Full pipeline
    python consolidate_to_parquet.py --oracc-dir data/oracc --etcsl-jsonl output/parallel_corpus.jsonl \
        --output-dir data/consolidated --include-glossary
"""

import argparse
import json
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import pyarrow as pa
import pyarrow.parquet as pq

from sumerian_nmt.data_ingestion.oracc_core import ORACCParser
from sumerian_nmt.utils.quality import is_mostly_broken


# ============================================================================
# PyArrow Schema Definitions (Explicit for type safety)
# ============================================================================

ORACC_SCHEMA = pa.schema([
    ("text_id", pa.string()),
    ("line_num", pa.string()),  # String for "o ii 4" edge cases
    ("text_normalized", pa.string()),
    ("text_raw", pa.string()),
    ("gloss_synthetic", pa.string()),
    ("word_count", pa.int16()),
    ("corpus", pa.string()),
    ("period", pa.string()),
    ("provenance", pa.string()),
    ("genre", pa.string()),
])

ETCSL_SCHEMA = pa.schema([
    ("composition_id", pa.string()),
    ("alignment_id", pa.string()),
    ("line_ids", pa.list_(pa.string())),
    ("line_numbers", pa.list_(pa.string())),
    ("text_raw", pa.string()),
    ("text_display", pa.string()),
    ("text_normalized", pa.string()),
    ("target_text", pa.string()),
    ("target_paragraph_id", pa.string()),
    ("tokens", pa.string()),  # JSON string for simplicity
    ("quality_has_damage", pa.bool_()),
    ("quality_has_supplied", pa.bool_()),
    ("quality_has_unclear", pa.bool_()),
    ("quality_has_gaps", pa.bool_()),
    ("token_count", pa.int16()),
])

GLOSSARY_SCHEMA = pa.schema([
    ("citation_form", pa.string()),
    ("guide_word", pa.string()),
    ("pos", pa.string()),
    ("instances", pa.int32()),
])


# ============================================================================
# Catalogue Loading (Left Join Logic)
# ============================================================================

def load_catalogue(catalogue_path: Path) -> Dict[str, Dict[str, str]]:
    """
    Load ORACC catalogue into lookup dictionary.

    Returns:
        Dict mapping text_id (P-number) to metadata dict
    """
    if not catalogue_path.exists():
        print(f"Warning: Catalogue not found at {catalogue_path}")
        return {}

    with open(catalogue_path, encoding="utf-8") as f:
        data = json.load(f)

    members = data.get("members", {})
    catalogue = {}

    for text_id, meta in members.items():
        catalogue[text_id] = {
            "period": meta.get("period", "Unknown"),
            "provenance": meta.get("provenience", "Unknown"),  # Note: ORACC uses "provenience"
            "genre": meta.get("genre", meta.get("supergenre", "Unknown")),
        }

    return catalogue


# ============================================================================
# ORACC Consolidation
# ============================================================================

def consolidate_oracc_corpus(
    corpus_dir: Path,
    output_path: Path,
    corpus_name: str,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Consolidate ORACC corpus to Parquet with catalogue metadata.

    Uses Left Join logic: texts not in catalogue get "Unknown" metadata.

    Args:
        corpus_dir: Path to corpus directory (e.g., data/oracc/epsd2-literary)
        output_path: Output parquet file path
        corpus_name: Corpus identifier ("literary" or "royal")
        verbose: Print progress

    Returns:
        Statistics dict
    """
    stats = {
        "corpus": corpus_name,
        "texts_processed": 0,
        "texts_skipped": 0,
        "lines_extracted": 0,
        "catalogue_matches": 0,
        "catalogue_misses": 0,
    }

    # Find the actual corpus subdirectory (e.g., epsd2/literary/)
    corpus_subdirs = list(corpus_dir.rglob("corpusjson"))
    if not corpus_subdirs:
        print(f"Error: No corpusjson directory found in {corpus_dir}")
        return stats

    corpusjson_dir = corpus_subdirs[0]
    catalogue_path = corpusjson_dir.parent / "catalogue.json"

    # Load catalogue (Left Join: missing entries get "Unknown")
    catalogue = load_catalogue(catalogue_path)
    if verbose:
        print(f"Loaded catalogue with {len(catalogue)} entries")

    # Initialize parser and data collection
    parser = ORACCParser(normalize=True)
    rows = []

    # Find all JSON files
    json_files = list(corpusjson_dir.glob("*.json"))
    if verbose:
        print(f"Found {len(json_files)} JSON files in {corpusjson_dir}")

    for i, json_file in enumerate(json_files):
        try:
            text_id = json_file.stem  # P-number

            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)

            # Left Join: get metadata or default to "Unknown"
            meta = catalogue.get(text_id, {
                "period": "Unknown",
                "provenance": "Unknown",
                "genre": "Unknown",
            })

            if text_id in catalogue:
                stats["catalogue_matches"] += 1
            else:
                stats["catalogue_misses"] += 1

            # Extract lines with glosses
            line_num = 0
            for sumerian, gloss in parser.extract_parallel_pairs(data):
                if is_mostly_broken(sumerian):
                    continue

                line_num += 1
                word_count = len(sumerian.split())

                rows.append({
                    "text_id": text_id,
                    "line_num": str(line_num),
                    "text_normalized": sumerian,
                    "text_raw": sumerian,  # TODO: preserve raw form if needed
                    "gloss_synthetic": gloss,
                    "word_count": word_count,
                    "corpus": corpus_name,
                    "period": meta["period"],
                    "provenance": meta["provenance"],
                    "genre": meta["genre"],
                })
                stats["lines_extracted"] += 1

            if line_num > 0:
                stats["texts_processed"] += 1
            else:
                stats["texts_skipped"] += 1

        except (json.JSONDecodeError, Exception) as e:
            stats["texts_skipped"] += 1
            if verbose and i < 5:
                print(f"  Warning: Error processing {json_file.name}: {e}")

        if verbose and (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(json_files)} files...")

    # Create Parquet table with explicit schema
    if rows:
        table = pa.Table.from_pylist(rows, schema=ORACC_SCHEMA)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, output_path, compression="snappy")

        if verbose:
            print(f"\nWrote {len(rows)} rows to {output_path}")
            print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    return stats


# ============================================================================
# ETCSL Consolidation (Zero-Drift: from JSONL)
# ============================================================================

def consolidate_etcsl_jsonl(
    jsonl_path: Path,
    output_path: Path,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Consolidate ETCSL from validated JSONL artifact to Parquet.

    Zero-Drift Approach: Uses existing extraction, does NOT re-parse XML.

    Args:
        jsonl_path: Path to parallel_corpus.jsonl
        output_path: Output parquet file path
        verbose: Print progress

    Returns:
        Statistics dict
    """
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"ETCSL JSONL not found: {jsonl_path}\n"
            "Run ETCSL extraction first:\n"
            "  python -m etcsl_extractor.main --output-dir output"
        )

    stats = {
        "alignments_processed": 0,
        "total_tokens": 0,
    }

    rows = []

    with open(jsonl_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line)

                source = record.get("source", {})
                target = record.get("target", {})
                quality = record.get("quality_flags", {})

                # Serialize tokens as JSON string
                tokens_json = json.dumps(source.get("tokens", []), ensure_ascii=False)

                rows.append({
                    "composition_id": record.get("composition_id", ""),
                    "alignment_id": record.get("alignment_id", ""),
                    "line_ids": source.get("line_ids", []),
                    "line_numbers": source.get("line_numbers", []),
                    "text_raw": source.get("text_raw", ""),
                    "text_display": source.get("text_display", ""),
                    "text_normalized": source.get("text_normalized", ""),
                    "target_text": target.get("text", ""),
                    "target_paragraph_id": target.get("paragraph_id", ""),
                    "tokens": tokens_json,
                    "quality_has_damage": quality.get("has_damage", False),
                    "quality_has_supplied": quality.get("has_supplied", False),
                    "quality_has_unclear": quality.get("has_unclear", False),
                    "quality_has_gaps": quality.get("has_gaps", False),
                    "token_count": source.get("token_count", 0),
                })

                stats["alignments_processed"] += 1
                stats["total_tokens"] += source.get("token_count", 0)

            except json.JSONDecodeError as e:
                if verbose:
                    print(f"  Warning: Invalid JSON at line {line_num}: {e}")

            if verbose and line_num % 1000 == 0:
                print(f"  Processed {line_num} alignments...")

    # Create Parquet table with explicit schema
    if rows:
        table = pa.Table.from_pylist(rows, schema=ETCSL_SCHEMA)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, output_path, compression="snappy")

        if verbose:
            print(f"\nWrote {len(rows)} alignments to {output_path}")
            print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    return stats


# ============================================================================
# Glossary Consolidation
# ============================================================================

def consolidate_glossary(
    corpus_dir: Path,
    output_path: Path,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Consolidate ORACC glossary to Parquet.

    Extracts vocabulary from index-sux.json (main Sumerian vocabulary)
    and gloss-qpn.json (proper nouns). Keeps ALL entries for future
    flexibility - filtering should happen at runtime, not storage.

    Args:
        corpus_dir: Path to corpus directory (e.g., data/oracc/epsd2-literary)
        output_path: Output parquet file path
        verbose: Print progress

    Returns:
        Statistics dict
    """
    import re

    stats = {
        "sux_entries": 0,
        "qpn_entries": 0,
        "total_entries": 0,
    }

    rows = []

    # Find the actual corpus subdirectory
    corpus_subdirs = list(corpus_dir.rglob("index-sux.json"))
    if not corpus_subdirs:
        print(f"Warning: No index-sux.json found in {corpus_dir}")
        return stats

    base_dir = corpus_subdirs[0].parent

    # 1. Extract from index-sux.json (main Sumerian vocabulary)
    # Format: entries like "lugal[king]" with count
    index_sux_path = base_dir / "index-sux.json"
    if index_sux_path.exists():
        with open(index_sux_path, encoding="utf-8") as f:
            data = json.load(f)

        # Pattern: word[meaning] or word[meaning]POS
        pattern = re.compile(r'^([^\[]+)\[([^\]]+)\](\w*)$')

        for entry in data.get("keys", []):
            key = entry.get("key", "")
            count = int(entry.get("count", 0))

            match = pattern.match(key)
            if match:
                cf = match.group(1).strip()  # citation form
                gw = match.group(2).strip()  # guide word (meaning)
                pos = match.group(3).strip() if match.group(3) else ""  # POS tag

                rows.append({
                    "citation_form": cf,
                    "guide_word": gw,
                    "pos": pos,
                    "instances": count,
                })
                stats["sux_entries"] += 1

        if verbose:
            print(f"  Extracted {stats['sux_entries']} entries from index-sux.json")

    # 2. Extract from gloss-qpn.json (proper nouns: divine names, places, etc.)
    gloss_qpn_path = base_dir / "gloss-qpn.json"
    if gloss_qpn_path.exists():
        with open(gloss_qpn_path, encoding="utf-8") as f:
            data = json.load(f)

        for entry in data.get("entries", []):
            cf = entry.get("cf", "")
            gw = entry.get("gw", "")
            pos = entry.get("pos", "")
            icount = int(entry.get("icount", 0))

            if cf:
                rows.append({
                    "citation_form": cf,
                    "guide_word": gw if gw else cf,  # Use cf if no gw
                    "pos": pos,
                    "instances": icount,
                })
                stats["qpn_entries"] += 1

        if verbose:
            print(f"  Extracted {stats['qpn_entries']} entries from gloss-qpn.json")

    # 3. Also check Emesal dialect glossary
    gloss_emesal_path = base_dir / "gloss-sux-x-emesal.json"
    if gloss_emesal_path.exists():
        with open(gloss_emesal_path, encoding="utf-8") as f:
            data = json.load(f)

        emesal_count = 0
        for entry in data.get("entries", []):
            cf = entry.get("cf", "")
            gw = entry.get("gw", "")
            pos = entry.get("pos", "")
            icount = int(entry.get("icount", 0))

            if cf and gw:
                rows.append({
                    "citation_form": cf,
                    "guide_word": gw,
                    "pos": f"{pos}/Emesal" if pos else "Emesal",
                    "instances": icount,
                })
                emesal_count += 1

        if verbose and emesal_count > 0:
            print(f"  Extracted {emesal_count} entries from gloss-sux-x-emesal.json")

    stats["total_entries"] = len(rows)

    if rows:
        table = pa.Table.from_pylist(rows, schema=GLOSSARY_SCHEMA)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, output_path, compression="snappy")

        if verbose:
            print(f"Wrote {len(rows)} total glossary entries to {output_path}")
            print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")

    return stats


# ============================================================================
# Archive Creation
# ============================================================================

def create_archive(
    source_dirs: List[Path],
    archive_path: Path,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Create tar.gz archive of source directories.

    Args:
        source_dirs: List of directories to archive
        archive_path: Output archive path
        verbose: Print progress

    Returns:
        Statistics dict
    """
    stats = {
        "directories": len(source_dirs),
        "files_archived": 0,
        "total_size_mb": 0,
    }

    archive_path.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, "w:gz") as tar:
        for src_dir in source_dirs:
            if not src_dir.exists():
                if verbose:
                    print(f"Skipping {src_dir} (not found)")
                continue

            if verbose:
                print(f"Archiving {src_dir}...")

            for file_path in src_dir.rglob("*"):
                if file_path.is_file():
                    # Use relative path in archive
                    arcname = file_path.relative_to(src_dir.parent)
                    tar.add(file_path, arcname=arcname)
                    stats["files_archived"] += 1

    if archive_path.exists():
        stats["total_size_mb"] = archive_path.stat().st_size / 1024 / 1024
        if verbose:
            print(f"\nArchive created: {archive_path}")
            print(f"  Files: {stats['files_archived']}")
            print(f"  Size: {stats['total_size_mb']:.2f} MB")

    return stats


# ============================================================================
# Metadata Generation
# ============================================================================

def save_metadata(
    output_dir: Path,
    stats: Dict[str, Any],
    verbose: bool = True
):
    """Save consolidation metadata to JSON."""
    metadata = {
        "consolidation_timestamp": datetime.utcnow().isoformat(),
        "schema_version": "1.0",
        "stats": stats,
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if verbose:
        print(f"Metadata saved to {metadata_path}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Consolidate ORACC/ETCSL datasets to Parquet format"
    )

    parser.add_argument(
        "--oracc-dir",
        type=Path,
        help="Path to ORACC data directory (contains epsd2-literary/, epsd2-royal/)"
    )
    parser.add_argument(
        "--etcsl-jsonl",
        type=Path,
        help="Path to ETCSL parallel_corpus.jsonl (Zero-Drift: from extraction)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/consolidated"),
        help="Output directory for Parquet files"
    )
    parser.add_argument(
        "--include-glossary",
        action="store_true",
        help="Also consolidate glossary files"
    )
    parser.add_argument(
        "--archive-only",
        action="store_true",
        help="Only create archive, don't extract"
    )
    parser.add_argument(
        "--archive-path",
        type=Path,
        help="Path for archive (default: archives/raw_sources_YYYY.tar.gz)"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()
    verbose = not args.quiet

    all_stats = {}

    # Archive only mode
    if args.archive_only:
        if not args.archive_path:
            year = datetime.now().year
            args.archive_path = Path(f"archives/raw_sources_{year}.tar.gz")

        source_dirs = [
            Path("data/oracc"),
            Path("ota_20"),
        ]

        all_stats["archive"] = create_archive(
            source_dirs=source_dirs,
            archive_path=args.archive_path,
            verbose=verbose
        )
        return

    # ORACC consolidation
    if args.oracc_dir:
        if verbose:
            print("=" * 60)
            print("ORACC Consolidation")
            print("=" * 60)

        # Process literary corpus
        literary_dir = args.oracc_dir / "epsd2-literary"
        if literary_dir.exists():
            if verbose:
                print(f"\nProcessing literary corpus: {literary_dir}")
            all_stats["oracc_literary"] = consolidate_oracc_corpus(
                corpus_dir=literary_dir,
                output_path=args.output_dir / "oracc_literary.parquet",
                corpus_name="literary",
                verbose=verbose
            )

        # Process royal corpus
        royal_dir = args.oracc_dir / "epsd2-royal"
        if royal_dir.exists():
            if verbose:
                print(f"\nProcessing royal corpus: {royal_dir}")
            all_stats["oracc_royal"] = consolidate_oracc_corpus(
                corpus_dir=royal_dir,
                output_path=args.output_dir / "oracc_royal.parquet",
                corpus_name="royal",
                verbose=verbose
            )

        # Glossary (from literary corpus - includes index-sux.json + gloss-qpn.json)
        if args.include_glossary:
            if verbose:
                print(f"\nProcessing glossary from: {literary_dir}")
            all_stats["glossary"] = consolidate_glossary(
                corpus_dir=literary_dir,
                output_path=args.output_dir / "glossary_sux.parquet",
                verbose=verbose
            )

    # ETCSL consolidation (Zero-Drift: from JSONL)
    if args.etcsl_jsonl:
        if verbose:
            print("\n" + "=" * 60)
            print("ETCSL Consolidation (Zero-Drift from JSONL)")
            print("=" * 60)

        all_stats["etcsl"] = consolidate_etcsl_jsonl(
            jsonl_path=args.etcsl_jsonl,
            output_path=args.output_dir / "etcsl_gold.parquet",
            verbose=verbose
        )

    # Save metadata
    if all_stats:
        save_metadata(args.output_dir, all_stats, verbose=verbose)

    # Summary
    if verbose:
        print("\n" + "=" * 60)
        print("CONSOLIDATION COMPLETE")
        print("=" * 60)
        print(f"\nOutput directory: {args.output_dir}")
        for name, stats in all_stats.items():
            print(f"\n{name}:")
            for key, value in stats.items():
                print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
