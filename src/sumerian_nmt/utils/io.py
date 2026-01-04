"""
Data I/O utilities for training data generation.

Provides consistent interface for:
- JSONL reading/writing (legacy)
- Chunked Parquet output (GitHub-friendly, reproducible)

Parquet chunking enables:
- Files under 50MB for GitHub LFS-free storage
- Efficient columnar compression
- Schema enforcement
- Parallel loading
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Any, Optional, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def load_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load all records from a JSONL file into memory.

    Args:
        path: Path to JSONL file

    Returns:
        List of parsed JSON objects
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def iter_jsonl(path: Union[str, Path]) -> Iterator[Dict[str, Any]]:
    """
    Iterate over JSONL file without loading all into memory.

    Args:
        path: Path to JSONL file

    Yields:
        Parsed JSON objects one at a time
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def save_jsonl(
    path: Union[str, Path],
    records: List[Dict[str, Any]],
    ensure_ascii: bool = False
) -> int:
    """
    Save records to a JSONL file.

    Args:
        path: Output file path
        records: List of JSON-serializable objects
        ensure_ascii: If True, escape non-ASCII characters

    Returns:
        Number of records written
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=ensure_ascii) + "\n")

    return len(records)


def load_text_corpus(path: Union[str, Path]) -> List[str]:
    """
    Load a plain text corpus (one line per document).

    Args:
        path: Path to text file

    Returns:
        List of non-empty lines
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# =============================================================================
# CHUNKED PARQUET I/O (GitHub-friendly)
# =============================================================================

# Default chunk size: ~10k records typically yields 5-30MB parquet files
DEFAULT_CHUNK_SIZE = 10000

# GitHub recommends files under 50MB, warn at 100MB
GITHUB_FILE_SIZE_WARN = 50 * 1024 * 1024  # 50MB
GITHUB_FILE_SIZE_MAX = 100 * 1024 * 1024  # 100MB


@dataclass
class ChunkMetadata:
    """Metadata for a parquet chunk."""
    chunk_index: int
    record_count: int
    file_size_bytes: int
    file_path: str


@dataclass
class DatasetMetadata:
    """Metadata for a chunked parquet dataset."""
    name: str
    total_records: int
    chunk_count: int
    chunks: List[ChunkMetadata]
    created_at: str
    schema: Dict[str, str]
    compression: str
    generator: str  # Script that generated this data


class ChunkedParquetWriter:
    """
    Write data to chunked parquet files for GitHub-friendly storage.

    Creates files like:
        output_dir/
            {prefix}_chunk_0000.parquet
            {prefix}_chunk_0001.parquet
            ...
            {prefix}_metadata.json

    Usage:
        writer = ChunkedParquetWriter("data/dpo", "dpo_pairs", chunk_size=10000)
        for record in records:
            writer.add_record(record)
        metadata = writer.finalize()
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        prefix: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        compression: str = "snappy",
        generator: str = "unknown",
    ):
        """
        Initialize chunked parquet writer.

        Args:
            output_dir: Directory to write chunks
            prefix: Filename prefix for chunks
            chunk_size: Records per chunk (default: 10000)
            compression: Parquet compression (snappy, gzip, zstd)
            generator: Name of script generating this data
        """
        self.output_dir = Path(output_dir)
        self.prefix = prefix
        self.chunk_size = chunk_size
        self.compression = compression
        self.generator = generator

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._buffer: List[Dict[str, Any]] = []
        self._chunk_index = 0
        self._total_records = 0
        self._chunks: List[ChunkMetadata] = []
        self._schema: Optional[Dict[str, str]] = None

    def add_record(self, record: Dict[str, Any]) -> None:
        """Add a single record to the buffer."""
        self._buffer.append(record)

        if len(self._buffer) >= self.chunk_size:
            self._flush_chunk()

    def add_records(self, records: List[Dict[str, Any]]) -> None:
        """Add multiple records."""
        for record in records:
            self.add_record(record)

    def _flush_chunk(self) -> None:
        """Write current buffer to a parquet chunk."""
        if not self._buffer:
            return

        # Convert to DataFrame
        df = pd.DataFrame(self._buffer)

        # Capture schema from first chunk
        if self._schema is None:
            self._schema = {col: str(df[col].dtype) for col in df.columns}

        # Write parquet
        chunk_path = self.output_dir / f"{self.prefix}_chunk_{self._chunk_index:04d}.parquet"
        df.to_parquet(
            chunk_path,
            compression=self.compression,
            index=False,
        )

        # Check file size
        file_size = chunk_path.stat().st_size
        if file_size > GITHUB_FILE_SIZE_MAX:
            print(f"WARNING: Chunk {chunk_path.name} is {file_size / 1024**2:.1f}MB (>100MB)")
            print("  Consider reducing chunk_size for GitHub compatibility")
        elif file_size > GITHUB_FILE_SIZE_WARN:
            print(f"Note: Chunk {chunk_path.name} is {file_size / 1024**2:.1f}MB")

        # Track metadata
        self._chunks.append(ChunkMetadata(
            chunk_index=self._chunk_index,
            record_count=len(self._buffer),
            file_size_bytes=file_size,
            file_path=chunk_path.name,
        ))

        self._total_records += len(self._buffer)
        self._chunk_index += 1
        self._buffer = []

    def finalize(self) -> DatasetMetadata:
        """
        Finalize writing and return metadata.

        Flushes remaining buffer and writes metadata JSON.
        """
        # Flush remaining records
        self._flush_chunk()

        # Create metadata
        metadata = DatasetMetadata(
            name=self.prefix,
            total_records=self._total_records,
            chunk_count=len(self._chunks),
            chunks=[
                {
                    "chunk_index": c.chunk_index,
                    "record_count": c.record_count,
                    "file_size_bytes": c.file_size_bytes,
                    "file_path": c.file_path,
                }
                for c in self._chunks
            ],
            created_at=datetime.utcnow().isoformat(),
            schema=self._schema or {},
            compression=self.compression,
            generator=self.generator,
        )

        # Write metadata
        metadata_path = self.output_dir / f"{self.prefix}_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump({
                "name": metadata.name,
                "total_records": metadata.total_records,
                "chunk_count": metadata.chunk_count,
                "chunks": metadata.chunks,
                "created_at": metadata.created_at,
                "schema": metadata.schema,
                "compression": metadata.compression,
                "generator": metadata.generator,
            }, f, indent=2)

        # Print summary
        total_size = sum(c.file_size_bytes for c in self._chunks)
        print(f"Written {self._total_records:,} records to {len(self._chunks)} chunks")
        print(f"  Total size: {total_size / 1024**2:.1f}MB")
        print(f"  Metadata: {metadata_path}")

        return metadata


def load_chunked_parquet(
    input_dir: Union[str, Path],
    prefix: str,
) -> pd.DataFrame:
    """
    Load all chunks from a chunked parquet dataset.

    Args:
        input_dir: Directory containing chunks
        prefix: Filename prefix used when writing

    Returns:
        Combined DataFrame with all records
    """
    input_dir = Path(input_dir)

    # Find all chunks
    pattern = f"{prefix}_chunk_*.parquet"
    chunk_files = sorted(input_dir.glob(pattern))

    if not chunk_files:
        raise FileNotFoundError(f"No chunks found matching {input_dir / pattern}")

    # Load and concatenate
    dfs = [pd.read_parquet(f) for f in chunk_files]
    combined = pd.concat(dfs, ignore_index=True)

    print(f"Loaded {len(combined):,} records from {len(chunk_files)} chunks")

    return combined


def load_chunked_parquet_lazy(
    input_dir: Union[str, Path],
    prefix: str,
) -> Iterator[pd.DataFrame]:
    """
    Lazily iterate over chunks without loading all into memory.

    Args:
        input_dir: Directory containing chunks
        prefix: Filename prefix

    Yields:
        DataFrame for each chunk
    """
    input_dir = Path(input_dir)
    pattern = f"{prefix}_chunk_*.parquet"
    chunk_files = sorted(input_dir.glob(pattern))

    for chunk_file in chunk_files:
        yield pd.read_parquet(chunk_file)


def get_dataset_metadata(
    input_dir: Union[str, Path],
    prefix: str,
) -> Dict[str, Any]:
    """
    Load metadata for a chunked parquet dataset.

    Args:
        input_dir: Directory containing chunks
        prefix: Filename prefix

    Returns:
        Metadata dictionary
    """
    input_dir = Path(input_dir)
    metadata_path = input_dir / f"{prefix}_metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    with open(metadata_path, encoding="utf-8") as f:
        return json.load(f)


def parquet_to_json(
    input_dir: Union[str, Path],
    prefix: str,
    output_path: Union[str, Path],
) -> int:
    """
    Convert chunked parquet to single JSON file (for Axolotl compatibility).

    Args:
        input_dir: Directory containing chunks
        prefix: Filename prefix
        output_path: Output JSON file path

    Returns:
        Number of records written
    """
    df = load_chunked_parquet(input_dir, prefix)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = df.to_dict(orient="records")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Converted {len(records):,} records to {output_path}")
    return len(records)


def zip_parquet_dataset(
    input_dir: Union[str, Path],
    prefix: str,
    output_zip: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Create a zip archive of a chunked parquet dataset for GitHub storage.

    Args:
        input_dir: Directory containing parquet chunks
        prefix: Filename prefix used when writing
        output_zip: Output zip path (default: {input_dir}/{prefix}.zip)

    Returns:
        Path to created zip file
    """
    import zipfile

    input_dir = Path(input_dir)

    if output_zip is None:
        output_zip = input_dir.parent / f"{prefix}.zip"
    else:
        output_zip = Path(output_zip)

    # Find all chunks and metadata
    pattern = f"{prefix}_chunk_*.parquet"
    chunk_files = sorted(input_dir.glob(pattern))
    metadata_file = input_dir / f"{prefix}_metadata.json"

    if not chunk_files:
        raise FileNotFoundError(f"No chunks found matching {input_dir / pattern}")

    # Create zip
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        for chunk_file in chunk_files:
            zf.write(chunk_file, chunk_file.name)

        if metadata_file.exists():
            zf.write(metadata_file, metadata_file.name)

    zip_size = output_zip.stat().st_size
    print(f"Created {output_zip} ({zip_size / 1024**2:.1f}MB)")
    print(f"  Contains {len(chunk_files)} chunks + metadata")

    return output_zip


def unzip_parquet_dataset(
    zip_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Extract a zipped parquet dataset.

    Args:
        zip_path: Path to zip file
        output_dir: Output directory (default: same directory as zip)

    Returns:
        Path to extracted directory
    """
    import zipfile

    zip_path = Path(zip_path)

    if output_dir is None:
        output_dir = zip_path.parent / zip_path.stem
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(output_dir)

    print(f"Extracted to {output_dir}")
    return output_dir
