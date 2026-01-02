"""
Data Loading Abstraction Layer.

Provides format-agnostic loading for ETCSL and ORACC data,
supporting both Parquet (preferred) and JSONL (fallback) formats.

Usage:
    from common.data_loader import load_etcsl_data, load_oracc_data

    # Load ETCSL (auto-detects format)
    df = load_etcsl_data()

    # Load ORACC with specific corpus
    df = load_oracc_data(corpus="literary")

    # Force JSONL format for debugging
    records = load_etcsl_data(use_parquet=False)
"""

import json
from pathlib import Path
from typing import Union, List, Dict, Any, Optional

try:
    import pandas as pd
    import pyarrow.parquet as pq
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False

# Add parent for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Paths


def load_etcsl_data(
    use_parquet: bool = True,
    parquet_path: Optional[Path] = None,
    jsonl_path: Optional[Path] = None,
) -> Union["pd.DataFrame", List[Dict[str, Any]]]:
    """
    Load ETCSL parallel corpus data.

    Prefers Parquet format for efficiency, falls back to JSONL.

    Args:
        use_parquet: Prefer Parquet if available (default True)
        parquet_path: Override path to Parquet file
        jsonl_path: Override path to JSONL file

    Returns:
        DataFrame (if Parquet) or List of dicts (if JSONL)

    Raises:
        FileNotFoundError: If no data source found
    """
    # Resolve paths
    if parquet_path is None:
        parquet_path = getattr(Paths, "ETCSL_PARQUET", None)
        if parquet_path is None:
            parquet_path = Paths.ROOT / "data" / "consolidated" / "etcsl_gold.parquet"

    if jsonl_path is None:
        jsonl_path = Paths.ETCSL_CORPUS

    # Try Parquet first
    if use_parquet and HAS_PARQUET and parquet_path.exists():
        return pd.read_parquet(parquet_path)

    # Fall back to JSONL
    if jsonl_path.exists():
        from common.io import load_jsonl
        return load_jsonl(jsonl_path)

    raise FileNotFoundError(
        f"No ETCSL data found. Checked:\n"
        f"  Parquet: {parquet_path}\n"
        f"  JSONL: {jsonl_path}\n"
        "Run ETCSL extraction first:\n"
        "  python -m etcsl_extractor.main --output-dir output"
    )


def load_oracc_data(
    corpus: str = "literary",
    use_parquet: bool = True,
    parquet_path: Optional[Path] = None,
) -> Union["pd.DataFrame", List[Dict[str, Any]]]:
    """
    Load ORACC corpus data.

    Args:
        corpus: Corpus name ("literary" or "royal")
        use_parquet: Prefer Parquet if available (default True)
        parquet_path: Override path to Parquet file

    Returns:
        DataFrame (if Parquet) or empty list (if not found)

    Raises:
        FileNotFoundError: If no data source found
    """
    # Resolve path
    if parquet_path is None:
        parquet_dir = getattr(Paths, "CONSOLIDATED_DIR", None)
        if parquet_dir is None:
            parquet_dir = Paths.ROOT / "data" / "consolidated"
        parquet_path = parquet_dir / f"oracc_{corpus}.parquet"

    # Try Parquet
    if use_parquet and HAS_PARQUET and parquet_path.exists():
        return pd.read_parquet(parquet_path)

    raise FileNotFoundError(
        f"ORACC {corpus} data not found at {parquet_path}\n"
        "Run ORACC consolidation first:\n"
        f"  python processors/consolidate_to_parquet.py --oracc-dir data/oracc"
    )


def load_glossary(
    use_parquet: bool = True,
    parquet_path: Optional[Path] = None,
) -> Union["pd.DataFrame", List[Dict[str, Any]]]:
    """
    Load Sumerian glossary data.

    Args:
        use_parquet: Prefer Parquet if available (default True)
        parquet_path: Override path to Parquet file

    Returns:
        DataFrame (if Parquet) or list of dicts

    Raises:
        FileNotFoundError: If no data source found
    """
    # Resolve path
    if parquet_path is None:
        parquet_dir = getattr(Paths, "CONSOLIDATED_DIR", None)
        if parquet_dir is None:
            parquet_dir = Paths.ROOT / "data" / "consolidated"
        parquet_path = parquet_dir / "glossary_sux.parquet"

    # Try Parquet
    if use_parquet and HAS_PARQUET and parquet_path.exists():
        return pd.read_parquet(parquet_path)

    raise FileNotFoundError(
        f"Glossary not found at {parquet_path}\n"
        "Run consolidation with --include-glossary:\n"
        "  python processors/consolidate_to_parquet.py --oracc-dir data/oracc --include-glossary"
    )


def get_training_data(
    split: str = "train",
    use_parquet: bool = True,
) -> Union["pd.DataFrame", List[Dict[str, Any]]]:
    """
    Load training or validation data.

    This is a convenience wrapper that loads the prepared training data,
    preferring the consolidated Parquet format.

    Args:
        split: "train" or "valid"
        use_parquet: Prefer Parquet if available

    Returns:
        DataFrame or list of training examples
    """
    from config import get_train_file

    if split == "train":
        jsonl_path = get_train_file()
    elif split == "valid":
        jsonl_path = Paths.VALID_FILE
    else:
        raise ValueError(f"Unknown split: {split}. Use 'train' or 'valid'")

    # For now, training data is always JSONL
    # TODO: Add Parquet versions of training data
    if jsonl_path.exists():
        from common.io import load_jsonl
        return load_jsonl(jsonl_path)

    raise FileNotFoundError(
        f"Training data not found at {jsonl_path}\n"
        "Run data preparation first:\n"
        "  python processors/prepare_training_data.py"
    )


def check_data_availability() -> Dict[str, bool]:
    """
    Check which data sources are available.

    Returns:
        Dict mapping data source names to availability status
    """
    consolidated_dir = Paths.ROOT / "data" / "consolidated"

    return {
        "etcsl_parquet": (consolidated_dir / "etcsl_gold.parquet").exists(),
        "etcsl_jsonl": Paths.ETCSL_CORPUS.exists(),
        "oracc_literary_parquet": (consolidated_dir / "oracc_literary.parquet").exists(),
        "oracc_royal_parquet": (consolidated_dir / "oracc_royal.parquet").exists(),
        "glossary_parquet": (consolidated_dir / "glossary_sux.parquet").exists(),
        "train_jsonl": Paths.TRAIN_FILE.exists() or Paths.TRAIN_FILE_V2.exists(),
        "valid_jsonl": Paths.VALID_FILE.exists(),
        "pyarrow_available": HAS_PARQUET,
    }


if __name__ == "__main__":
    # Quick check of data availability
    print("Data Availability Check")
    print("=" * 40)
    for name, available in check_data_availability().items():
        status = "OK" if available else "MISSING"
        print(f"  {name}: {status}")
