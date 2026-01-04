"""
XML parsing utilities for ETCSL extraction.

Handles mixed content, damage detection, and recursive text extraction.
"""

from lxml import etree
from typing import Optional, Dict, List, Any
from io import StringIO


def parse_xml_string(xml_content: str) -> etree._Element:
    """
    Parse preprocessed XML content into an lxml element tree.

    Args:
        xml_content: Preprocessed XML string (entities already replaced)

    Returns:
        Root element of the parsed XML
    """
    # Use a parser that recovers from minor issues
    parser = etree.XMLParser(recover=True, encoding='utf-8')
    # Encode to bytes for lxml
    xml_bytes = xml_content.encode('utf-8')
    return etree.fromstring(xml_bytes, parser=parser)


def extract_text_content(element: etree._Element) -> str:
    """
    Extract all text content from an element, handling mixed content.

    Args:
        element: XML element

    Returns:
        Concatenated text content
    """
    return "".join(element.itertext())


def check_damage_context(element: etree._Element) -> Dict[str, bool]:
    """
    Check if element is within damage/supplied/unclear context.

    Walks up the parent chain to detect enclosing editorial markup.

    Args:
        element: XML element to check

    Returns:
        Dict with damage/supplied/unclear flags
    """
    flags = {
        "damaged": False,
        "supplied": False,
        "unclear": False,
    }

    # Check element's own children for damage markers
    if element.find(".//damage") is not None:
        flags["damaged"] = True
    if element.find(".//supplied") is not None:
        flags["supplied"] = True
    if element.find(".//unclear") is not None:
        flags["unclear"] = True

    # Walk up parent chain
    parent = element.getparent()
    while parent is not None:
        tag = parent.tag
        if tag == "damage":
            flags["damaged"] = True
        elif tag == "supplied":
            flags["supplied"] = True
        elif tag == "unclear":
            flags["unclear"] = True
        # Stop at line level
        if tag == "l":
            break
        parent = parent.getparent()

    return flags


def extract_word_data(w_element: etree._Element) -> Dict[str, Any]:
    """
    Extract complete data from a <w> (word) element.

    Handles:
    - All word attributes (form, lemma, pos, type, label, det, etc.)
    - Nested damage/supplied/unclear markup
    - Mixed content text extraction

    Args:
        w_element: <w> XML element

    Returns:
        Dictionary with word data and flags
    """
    # Get all attributes
    form = w_element.get("form", "")
    lemma = w_element.get("lemma")
    pos = w_element.get("pos")
    word_type = w_element.get("type")
    label = w_element.get("label")
    det = w_element.get("det")
    form_type = w_element.get("form-type")
    emesal = w_element.get("emesal")
    emesal_prefix = w_element.get("emesal-prefix")
    bound = w_element.get("bound")
    npart = w_element.get("npart")

    # Extract text content (handles nested elements)
    text_content = extract_text_content(w_element)

    # Detect damage flags
    flags = check_damage_context(w_element)

    # Check for unclear with certainty attribute
    unclear_elem = w_element.find(".//unclear")
    unclear_certainty = None
    if unclear_elem is not None:
        unclear_certainty = unclear_elem.get("cert")

    return {
        "form": form,
        "lemma": lemma,
        "pos": pos,
        "type": word_type,
        "label": label,
        "det": det,
        "form_type": form_type,
        "emesal": emesal,
        "emesal_prefix": emesal_prefix,
        "bound": bound,
        "npart": npart,
        "text_content": text_content.strip(),
        "flags": {
            "damaged": flags["damaged"],
            "supplied": flags["supplied"],
            "unclear": flags["unclear"],
            "unclear_certainty": unclear_certainty,
        }
    }


def extract_line_data(l_element: etree._Element) -> Dict[str, Any]:
    """
    Extract complete data from an <l> (line) element.

    Args:
        l_element: <l> XML element

    Returns:
        Dictionary with line data including all words
    """
    line_n = l_element.get("n")
    line_id = l_element.get("id")
    corresp = l_element.get("corresp")

    # Extract all words
    words = []
    for idx, w_elem in enumerate(l_element.findall(".//w")):
        word_data = extract_word_data(w_elem)
        word_data["position"] = idx
        words.append(word_data)

    # Check if line has gaps
    gaps = l_element.findall(".//gap")
    gap_info = []
    for gap in gaps:
        gap_info.append({
            "extent": gap.get("extent", "unknown")
        })

    # Check line-level damage
    line_flags = check_damage_context(l_element)

    return {
        "n": line_n,
        "id": line_id,
        "corresp": corresp,
        "words": words,
        "gaps": gap_info,
        "flags": line_flags,
    }


def extract_paragraph_data(p_element: etree._Element) -> Dict[str, Any]:
    """
    Extract data from a <p> (paragraph) element in translations.

    Args:
        p_element: <p> XML element

    Returns:
        Dictionary with paragraph data
    """
    para_id = p_element.get("id")
    n = p_element.get("n")  # Line range, e.g., "1-4"
    corresp = p_element.get("corresp")

    # Extract text content
    text_content = extract_text_content(p_element)

    # Extract named entities from translation
    named_entities = []
    for w_elem in p_element.findall(".//w[@type]"):
        ne_type = w_elem.get("type")
        ne_text = extract_text_content(w_elem)
        named_entities.append({
            "text": ne_text.strip(),
            "type": ne_type,
        })

    # Extract quoted speech
    quotes = []
    for q_elem in p_element.findall(".//q"):
        quotes.append({
            "who": q_elem.get("who"),
            "toWhom": q_elem.get("toWhom"),
            "text": extract_text_content(q_elem).strip()[:100],  # Truncate for brevity
        })

    # Extract foreign words
    foreign_words = []
    for f_elem in p_element.findall(".//foreign"):
        foreign_words.append({
            "lang": f_elem.get("lang"),
            "text": extract_text_content(f_elem).strip(),
        })

    # Check for gaps
    gaps = []
    for gap in p_element.findall(".//gap"):
        gaps.append({"extent": gap.get("extent", "unknown")})

    return {
        "id": para_id,
        "n": n,
        "corresp": corresp,
        "text": text_content.strip(),
        "named_entities": named_entities,
        "quotes": quotes,
        "foreign_words": foreign_words,
        "gaps": gaps,
    }


def extract_header_metadata(tei_header: etree._Element) -> Dict[str, Any]:
    """
    Extract metadata from TEI header.

    Args:
        tei_header: <teiHeader> element

    Returns:
        Dictionary with metadata
    """
    metadata = {
        "title": None,
        "editors": [],
        "publication_date": None,
        "bibliography": [],
        "tablets": [],
        "revision_history": [],
    }

    # Title
    title_elem = tei_header.find(".//title")
    if title_elem is not None:
        metadata["title"] = extract_text_content(title_elem).strip()

    # Editors/Contributors
    for resp_stmt in tei_header.findall(".//respStmt"):
        name_elem = resp_stmt.find("name")
        resp_elem = resp_stmt.find("resp")
        if name_elem is not None:
            metadata["editors"].append({
                "name": extract_text_content(name_elem).strip(),
                "role": extract_text_content(resp_elem).strip() if resp_elem is not None else None,
            })

    # Publication date
    date_elem = tei_header.find(".//publicationStmt/date")
    if date_elem is not None:
        metadata["publication_date"] = extract_text_content(date_elem).strip()

    # Bibliography references
    for bibl in tei_header.findall(".//listBibl[@type='secondary']/bibl"):
        xref = bibl.find("xref")
        if xref is not None:
            metadata["bibliography"].append({
                "ref_id": xref.get("from"),
                "citation": extract_text_content(xref).strip(),
            })

    # Tablet sources
    for bibl in tei_header.findall(".//listBibl[@type='tablets']/bibl"):
        scope = bibl.find("biblScope[@type='museumNumber']")
        if scope is not None:
            metadata["tablets"].append({
                "museum_number": extract_text_content(scope).strip(),
            })

    # Revision history
    for change in tei_header.findall(".//revisionDesc/change"):
        date_elem = change.find("date")
        item_elem = change.find("item")
        resp_name = change.find("respStmt/name")
        metadata["revision_history"].append({
            "date": extract_text_content(date_elem).strip() if date_elem is not None else None,
            "editor": extract_text_content(resp_name).strip() if resp_name is not None else None,
            "action": extract_text_content(item_elem).strip() if item_elem is not None else None,
        })

    return metadata
