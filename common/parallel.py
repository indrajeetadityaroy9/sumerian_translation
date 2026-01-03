"""
Parallel Processing Utilities for Multi-Core CPU Optimization.

Provides ProcessPoolExecutor-based parallelization for data pipeline operations.
Optimized for 52 vCPU systems (H100 instance typical configuration).

Usage:
    from common.parallel import parallel_map, get_optimal_workers

    # Process items in parallel
    results = parallel_map(process_func, items)

    # With custom worker count
    results = parallel_map(process_func, items, num_workers=32)
"""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, List, Optional, TypeVar

from tqdm import tqdm

T = TypeVar("T")
R = TypeVar("R")


def get_optimal_workers(reserve_cores: int = 4) -> int:
    """
    Get optimal worker count for CPU parallelization.

    Reserves some cores for the main process, system tasks, and potential
    GPU coordination overhead.

    Args:
        reserve_cores: Number of cores to reserve (default: 4)

    Returns:
        Optimal number of worker processes

    Example:
        >>> get_optimal_workers()  # On 52 vCPU system
        48
    """
    cpu_count = mp.cpu_count()
    # Reserve cores for main process and system
    optimal = max(1, cpu_count - reserve_cores)
    # Cap at 48 to avoid excessive context switching
    return min(48, optimal)


def parallel_map(
    func: Callable[[T], R],
    items: Iterable[T],
    num_workers: Optional[int] = None,
    chunk_size: int = 100,
    desc: Optional[str] = None,
    show_progress: bool = True,
) -> List[R]:
    """
    Process items in parallel using ProcessPoolExecutor.

    This is the primary function for CPU-bound parallelization. Uses
    multiprocessing to bypass Python's GIL for true parallelism.

    Args:
        func: Function to apply to each item (must be picklable)
        items: Iterable of items to process
        num_workers: Number of worker processes (default: auto-detect)
        chunk_size: Items per worker task (larger = less overhead)
        desc: Description for progress bar
        show_progress: Whether to show tqdm progress bar

    Returns:
        List of results in same order as input

    Example:
        def process_record(record: dict) -> dict:
            # CPU-intensive processing
            return transformed_record

        results = parallel_map(process_record, records, desc="Processing")
    """
    if num_workers is None:
        num_workers = get_optimal_workers()

    items_list = list(items)

    if len(items_list) == 0:
        return []

    # For small workloads, sequential is faster due to pickling overhead
    if len(items_list) < num_workers * 2:
        if show_progress and desc:
            items_list = tqdm(items_list, desc=desc)
        return [func(item) for item in items_list]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        if show_progress:
            results = list(
                tqdm(
                    executor.map(func, items_list, chunksize=chunk_size),
                    total=len(items_list),
                    desc=desc or "Processing",
                )
            )
        else:
            results = list(executor.map(func, items_list, chunksize=chunk_size))

    return results


def parallel_map_unordered(
    func: Callable[[T], R],
    items: Iterable[T],
    num_workers: Optional[int] = None,
    desc: Optional[str] = None,
    show_progress: bool = True,
) -> List[R]:
    """
    Process items in parallel, returning results as they complete.

    Unlike parallel_map, results may be returned in any order. Use this
    when order doesn't matter and you want to process results as soon
    as they're available.

    Args:
        func: Function to apply to each item
        items: Iterable of items to process
        num_workers: Number of worker processes
        desc: Description for progress bar
        show_progress: Whether to show tqdm progress bar

    Returns:
        List of results (order not guaranteed)
    """
    if num_workers is None:
        num_workers = get_optimal_workers()

    items_list = list(items)

    if len(items_list) == 0:
        return []

    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(func, item): item for item in items_list}

        iterator = as_completed(futures)
        if show_progress:
            iterator = tqdm(iterator, total=len(futures), desc=desc or "Processing")

        for future in iterator:
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                # Log but continue processing other items
                print(f"Warning: Task failed with error: {e}")

    return results


def thread_map(
    func: Callable[[T], R],
    items: Iterable[T],
    num_workers: Optional[int] = None,
    desc: Optional[str] = None,
    show_progress: bool = True,
) -> List[R]:
    """
    Process items using ThreadPoolExecutor.

    Use for I/O-bound tasks (file reading, network requests) where GIL
    is not a bottleneck. Lighter weight than ProcessPoolExecutor.

    Args:
        func: Function to apply to each item
        items: Iterable of items to process
        num_workers: Number of worker threads (default: 2x CPU cores for I/O)
        desc: Description for progress bar
        show_progress: Whether to show tqdm progress bar

    Returns:
        List of results in same order as input

    Example:
        def load_file(path: Path) -> dict:
            return json.load(open(path))

        data = thread_map(load_file, file_paths, desc="Loading files")
    """
    if num_workers is None:
        # For I/O bound, use more threads than cores
        num_workers = min(64, get_optimal_workers() * 2)

    items_list = list(items)

    if len(items_list) == 0:
        return []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        if show_progress:
            results = list(
                tqdm(
                    executor.map(func, items_list),
                    total=len(items_list),
                    desc=desc or "Processing",
                )
            )
        else:
            results = list(executor.map(func, items_list))

    return results


def chunked_parallel_map(
    func: Callable[[List[T]], List[R]],
    items: Iterable[T],
    batch_size: int = 1000,
    num_workers: Optional[int] = None,
    desc: Optional[str] = None,
) -> List[R]:
    """
    Process items in batches for reduced IPC overhead.

    When items are small (e.g., strings, small dicts), the overhead of
    pickling each item individually can dominate. This function batches
    items before sending to workers.

    Args:
        func: Function that takes a LIST of items and returns a LIST of results
        items: Iterable of items to process
        batch_size: Items per batch
        num_workers: Number of worker processes
        desc: Description for progress bar

    Returns:
        Flattened list of all results

    Example:
        def process_batch(records: List[dict]) -> List[dict]:
            return [transform(r) for r in records]

        results = chunked_parallel_map(process_batch, records, batch_size=500)
    """
    if num_workers is None:
        num_workers = get_optimal_workers()

    items_list = list(items)

    if len(items_list) == 0:
        return []

    # Create batches
    batches = [items_list[i : i + batch_size] for i in range(0, len(items_list), batch_size)]

    # Process batches in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        batch_results = list(
            tqdm(
                executor.map(func, batches),
                total=len(batches),
                desc=desc or "Processing batches",
            )
        )

    # Flatten results
    return [item for batch in batch_results for item in batch]


# Convenience exports
__all__ = [
    "get_optimal_workers",
    "parallel_map",
    "parallel_map_unordered",
    "thread_map",
    "chunked_parallel_map",
]
