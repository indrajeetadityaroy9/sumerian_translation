"""
XML Preprocessor for ETCSL files.

Replaces XML entities with placeholders/unicode BEFORE parsing,
avoiding DTD resolution failures.
"""

import re
from pathlib import Path
from typing import Union

from ..corpus_config import get_all_entity_replacements, DETERMINATIVES, SPECIAL_CHARS


class XMLPreprocessor:
    """
    Preprocesses ETCSL XML files by replacing entities with safe placeholders.

    Standard XML parsers fail on external entities like &d;, &ki; because they
    require DTD resolution. This preprocessor handles entity replacement
    using regex before the XML is parsed.
    """

    def __init__(self):
        self.replacements = get_all_entity_replacements()
        # Build regex pattern for all entities
        # Sort by length (longest first) to avoid partial matches
        entities = sorted(self.replacements.keys(), key=len, reverse=True)
        # Escape special regex characters in entity names
        escaped = [re.escape(e) for e in entities]
        self.pattern = re.compile("|".join(escaped))

    def preprocess(self, xml_text: str) -> str:
        """
        Replace all ETCSL entities in XML text with safe placeholders.

        Args:
            xml_text: Raw XML content as string

        Returns:
            XML text with entities replaced
        """
        def replacer(match):
            entity = match.group(0)
            return self.replacements.get(entity, entity)

        return self.pattern.sub(replacer, xml_text)

    def preprocess_file(self, file_path: Union[str, Path]) -> str:
        """
        Read and preprocess an XML file.

        Args:
            file_path: Path to XML file

        Returns:
            Preprocessed XML content
        """
        file_path = Path(file_path)
        # ETCSL files are ISO-8859-1 encoded
        with open(file_path, 'r', encoding='iso-8859-1') as f:
            raw_content = f.read()
        return self.preprocess(raw_content)


def get_determinative_type(placeholder: str) -> str:
    """
    Get the semantic type of a determinative from its placeholder.

    Args:
        placeholder: e.g., "{{DET_DIVINE}}"

    Returns:
        Type string, e.g., "divine"
    """
    for entity, (ph, det_type, _) in DETERMINATIVES.items():
        if ph == placeholder:
            return det_type
    return "unknown"


# Note: placeholder_to_display() and placeholder_to_normalized() are defined in
# etcsl_extractor/exporters/text_generator.py - import from there if needed.


if __name__ == "__main__":
    # Quick test
    from ..corpus_config import CORPUS_DIR

    preprocessor = XMLPreprocessor()
    test_file = CORPUS_DIR / "etcsl" / "transliterations" / "c.1.1.1.xml"

    if test_file.exists():
        result = preprocessor.preprocess_file(test_file)

        # Check for remaining unresolved entities
        remaining = re.findall(r'&[a-zA-Z0-9]+;', result)

        print(f"Preprocessed {test_file.name}")
        print(f"Output length: {len(result)} chars")

        if remaining:
            unique_remaining = set(remaining)
            print(f"Unresolved entities: {unique_remaining}")
        else:
            print("All entities resolved!")

        # Show sample with determinatives
        if "{{DET_DIVINE}}" in result:
            print("\nSample with divine determinative:")
            for line in result.split('\n'):
                if "{{DET_DIVINE}}" in line:
                    print(f"  {line[:100]}...")
                    break
    else:
        print(f"Test file not found: {test_file}")
