"""
Configuration constants and entity mappings for ETCSL extraction.
"""

from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
CORPUS_DIR = BASE_DIR / "ota_20"
TRANSLITERATIONS_DIR = CORPUS_DIR / "etcsl" / "transliterations"
TRANSLATIONS_DIR = CORPUS_DIR / "etcsl" / "translations"
BIBLIOGRAPHY_FILE = CORPUS_DIR / "etcsl" / "bibliography" / "bibliography.xml"
CATALOGUE_FILE = CORPUS_DIR / "etcslfullcat.html"
OUTPUT_DIR = BASE_DIR / "output"

# Determinative mappings: entity -> (placeholder, type, display)
DETERMINATIVES = {
    "&d;": ("{{DET_DIVINE}}", "divine", "d"),
    "&ki;": ("{{DET_PLACE}}", "place", "ki"),
    "&jic;": ("{{DET_WOOD}}", "wood", "gis"),
    "&na4;": ("{{DET_STONE}}", "stone", "na4"),
    "&gi;": ("{{DET_REED}}", "reed", "gi"),
    "&id2;": ("{{DET_WATER}}", "water", "id2"),
    "&e2;": ("{{DET_HOUSE}}", "house", "e2"),
    "&gud;": ("{{DET_OX}}", "ox", "gud"),
    "&iku;": ("{{DET_FIELD}}", "field", "iku"),
    "&itid;": ("{{DET_MONTH}}", "month", "itid"),
    "&im;": ("{{DET_CLAY}}", "clay", "im"),
    "&kac;": ("{{DET_BEER}}", "beer", "kas"),
    "&ku6;": ("{{DET_FISH}}", "fish", "ku6"),
    "&kur;": ("{{DET_MOUNTAIN}}", "mountain", "kur"),
    "&kuc;": ("{{DET_LEATHER}}", "leather", "kus"),
    "&lu2;": ("{{DET_PERSON}}", "person", "lu2"),
    "&m;": ("{{DET_MALE}}", "male", "m"),
    "&f;": ("{{DET_FEMALE}}", "female", "f"),
    "&mu;": ("{{DET_YEAR}}", "year", "mu"),
    "&mul;": ("{{DET_STAR}}", "star", "mul"),
    "&mucen;": ("{{DET_BIRD}}", "bird", "mucen"),
    "&ninda;": ("{{DET_BREAD}}", "bread", "ninda"),
    "&sa;": ("{{DET_BUNDLE}}", "bundle", "sa"),
    "&sar;": ("{{DET_GARDEN}}", "garden", "sar"),
    "&cah2;": ("{{DET_PIG}}", "pig", "sah2"),
    "&tug2;": ("{{DET_CLOTH}}", "cloth", "tug2"),
    "&tum9;": ("{{DET_WIND}}", "wind", "tum9"),
    "&u2;": ("{{DET_PLANT}}", "plant", "u2"),
    "&udu;": ("{{DET_SHEEP}}", "sheep", "udu"),
    "&urud;": ("{{DET_COPPER}}", "copper", "urud"),
    "&uzu;": ("{{DET_MEAT}}", "meat", "uzu"),
    "&zabar;": ("{{DET_BRONZE}}", "bronze", "zabar"),
    "&dug;": ("{{DET_POT}}", "pot", "dug"),
    "&ance;": ("{{DET_EQUID}}", "equid", "anse"),
}

# Special character mappings: entity -> (unicode, ascii)
SPECIAL_CHARS = {
    "&c;": ("š", "sz"),
    "&C;": ("Š", "SZ"),
    "&g;": ("ĝ", "j"),
    "&G;": ("Ĝ", "J"),
    "&h;": ("ḫ", "h"),
    "&H;": ("Ḫ", "H"),
    "&s;": ("ṣ", "s,"),
    "&S;": ("Ṣ", "S,"),
    "&t;": ("ṭ", "t,"),
    "&T;": ("Ṭ", "T,"),
    "&aleph;": ("ʾ", "'"),
    "&X;": ("…", "..."),
    "&hr;": ("―", "---"),
}

# Damage/editorial markers
EDITORIAL_MARKERS = {
    "&damb;": "{{DAMAGE_BEGIN}}",
    "&dame;": "{{DAMAGE_END}}",
    "&suppb;": "{{SUPPLIED_BEGIN}}",
    "&suppe;": "{{SUPPLIED_END}}",
    "&qryb;": "{{QUERY_BEGIN}}",
    "&qrye;": "{{QUERY_END}}",
    "&subb;": "{{SUBSCRIPT_BEGIN}}",
    "&sube;": "{{SUBSCRIPT_END}}",
}

# Subscript numerals
SUBSCRIPT_NUMERALS = {
    f"&s{i};": f"{{{{SUB_{i}}}}}" for i in range(10)
}

# Common HTML entities that may appear
HTML_ENTITIES = {
    "&aacute;": "á",
    "&eacute;": "é",
    "&iacute;": "í",
    "&oacute;": "ó",
    "&uacute;": "ú",
    "&auml;": "ä",
    "&euml;": "ë",
    "&iuml;": "ï",
    "&ouml;": "ö",
    "&uuml;": "ü",
    "&ntilde;": "ñ",
    "&commat;": "@",
    "&lt;": "<",
    "&gt;": ">",
    "&amp;": "&",
    "&quot;": '"',
}

# Part-of-speech mappings
POS_TAGS = {
    "N": "Noun",
    "V": "Verb",
    "AJ": "Adjective",
    "AV": "Adverb",
    "PD": "Pronoun/Demonstrative",
    "NU": "Numeral",
    "I": "Interjection",
    "C": "Conjunction",
    "NEG": "Negation",
    "X": "Unknown",
}

# Named entity types
NE_TYPES = {
    "DN": "Divine Name",
    "GN": "Geographic Name",
    "PN": "Personal Name",
    "SN": "Settlement Name",
    "TN": "Temple Name",
    "WN": "Water Name",
    "RN": "Royal Name",
    "MN": "Month Name",
    "ON": "Object Name",
    "EN": "Ethnicity Name",
}

# Composition category mappings
CATEGORY_GROUPS = {
    0: "Literary catalogues",
    1: "Narrative and mythological compositions",
    2: "Royal praise poetry and hymns",
    3: "Literary letters and letter-prayers",
    4: "Divine and temple hymns",
    5: "Other literary compositions",
    6: "Proverbs and proverb collections",
}

# Normalization for consistent tokenization
NORMALIZATION_MAP = {
    # Variant spellings to standard forms
    "sz": "š",
    "SZ": "Š",
    "j": "ĝ",
    "J": "Ĝ",
    # Remove subscript numbers for normalized form
    "₀": "", "₁": "", "₂": "", "₃": "", "₄": "",
    "₅": "", "₆": "", "₇": "", "₈": "", "₉": "",
}


def get_all_entity_replacements() -> dict:
    """
    Get combined dictionary of all entity replacements for preprocessing.
    Returns dict mapping entity -> replacement string.
    """
    replacements = {}

    # Determinatives -> placeholders
    for entity, (placeholder, _, _) in DETERMINATIVES.items():
        replacements[entity] = placeholder

    # Special chars -> unicode
    for entity, (unicode_char, _) in SPECIAL_CHARS.items():
        replacements[entity] = unicode_char

    # Editorial markers
    replacements.update(EDITORIAL_MARKERS)

    # Subscripts
    replacements.update(SUBSCRIPT_NUMERALS)

    # HTML entities
    replacements.update(HTML_ENTITIES)

    return replacements
