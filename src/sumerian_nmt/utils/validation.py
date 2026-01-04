"""
Schema validation utilities for Sumerian translation pipeline.

Validates data at pipeline boundaries to catch interface mismatches early.
"""

from typing import Dict, List, Any, Optional


def validate_etcsl_record(record: Dict[str, Any]) -> List[str]:
    """
    Validate ETCSL extractor output schema.

    Expected fields:
        - composition_id: str
        - source.text_normalized: str
        - target.text: str

    Args:
        record: Record from ETCSL extraction

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    if 'composition_id' not in record:
        errors.append("Missing composition_id")

    if 'source' not in record:
        errors.append("Missing source")
    elif not isinstance(record.get('source'), dict):
        errors.append("source must be a dict")
    elif 'text_normalized' not in record.get('source', {}):
        errors.append("Missing source.text_normalized")

    if 'target' not in record:
        errors.append("Missing target")
    elif not isinstance(record.get('target'), dict):
        errors.append("target must be a dict")
    elif 'text' not in record.get('target', {}):
        errors.append("Missing target.text")

    return errors


def validate_augmented_record(record: Dict[str, Any]) -> List[str]:
    """
    Validate graph augmentor output schema.

    Expected fields:
        - source.text_normalized: str
        - target.text: str
        - quality.synthetic: bool
        - quality.method: str
        - metadata.template_line_id: str

    Args:
        record: Record from graph augmentor

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    if 'source' not in record:
        errors.append("Missing source")
    elif 'text_normalized' not in record.get('source', {}):
        errors.append("Missing source.text_normalized")

    if 'target' not in record:
        errors.append("Missing target")
    elif 'text' not in record.get('target', {}):
        errors.append("Missing target.text")

    if 'quality' not in record:
        errors.append("Missing quality (interface mismatch with consolidate_for_llm)")
    else:
        quality = record.get('quality', {})
        if 'synthetic' not in quality:
            errors.append("Missing quality.synthetic")
        if 'method' not in quality:
            errors.append("Missing quality.method")

    if 'metadata' not in record:
        errors.append("Missing metadata")
    elif 'template_line_id' not in record.get('metadata', {}):
        errors.append("Missing metadata.template_line_id (needed for composition tracking)")

    return errors


def validate_training_record(record: Dict[str, Any]) -> List[str]:
    """
    Validate Alpaca-format training record.

    Expected fields:
        - instruction: str
        - input: str
        - output: str

    Args:
        record: Alpaca-format training record

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    for field in ['instruction', 'input', 'output']:
        if field not in record:
            errors.append(f"Missing {field}")
        elif not isinstance(record.get(field), str):
            errors.append(f"{field} must be a string")
        elif not record.get(field, '').strip():
            errors.append(f"{field} is empty")

    return errors


def validate_dpo_record(record: Dict[str, Any]) -> List[str]:
    """
    Validate DPO preference pair record.

    Expected fields:
        - instruction: str
        - input: str
        - chosen: str
        - rejected: str

    Args:
        record: DPO preference pair

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    for field in ['instruction', 'input', 'chosen', 'rejected']:
        if field not in record:
            errors.append(f"Missing {field}")
        elif not isinstance(record.get(field), str):
            errors.append(f"{field} must be a string")
        elif not record.get(field, '').strip():
            errors.append(f"{field} is empty")

    return errors


def validate_batch(
    records: List[Dict[str, Any]],
    validator: callable,
    max_errors: int = 10,
) -> Dict[str, Any]:
    """
    Validate a batch of records and return summary statistics.

    Args:
        records: List of records to validate
        validator: Validation function to use
        max_errors: Maximum number of error examples to collect

    Returns:
        Dictionary with validation results:
            - valid_count: Number of valid records
            - invalid_count: Number of invalid records
            - error_examples: List of (index, errors) tuples
            - error_counts: Dict of error message -> count
    """
    valid_count = 0
    invalid_count = 0
    error_examples = []
    error_counts: Dict[str, int] = {}

    for i, record in enumerate(records):
        errors = validator(record)
        if errors:
            invalid_count += 1
            if len(error_examples) < max_errors:
                error_examples.append((i, errors))
            for err in errors:
                error_counts[err] = error_counts.get(err, 0) + 1
        else:
            valid_count += 1

    return {
        'valid_count': valid_count,
        'invalid_count': invalid_count,
        'error_examples': error_examples,
        'error_counts': error_counts,
    }
