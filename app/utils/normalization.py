"""
Data normalization utilities for FRA OCR project
"""
from __future__ import annotations
import re
from datetime import datetime
from typing import Optional, Tuple, Dict, List
from loguru import logger
from unidecode import unidecode
from config.config import Config

# Optional transliteration
try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
except Exception:
    sanscript = None
    transliterate = None


def normalize_name(name: str) -> Tuple[str, List[str]]:
    """Normalize personal names by removing honorifics, fixing case and spacing.
    Returns normalized name and a list of notes.
    """
    notes: List[str] = []
    if not name:
        return name, notes

    raw = name
    # Remove content in parentheses that are translations e.g., (Ram Kumar)
    name = re.sub(r"[\(\)\[\]{}]", "", name)

    # Remove honorifics
    tokens = [t for t in re.split(r"\s+", name) if t]
    tokens = [t for t in tokens if t.strip('.').title() not in Config.HONORIFICS]

    # Fix common OCR glyph errors
    fixed = []
    for t in tokens:
        t = t.replace('@', 'a').replace('!', 'i').replace('$', 's').replace('0', 'o')
        fixed.append(t)
    name = " ".join(fixed)

    # Attempt transliteration if non-Latin detected
    if re.search(r"[^\x00-\x7F]", raw):
        try:
            if transliterate:
                name_latn = transliterate(raw, sanscript.DEVANAGARI, sanscript.ITRANS)
                notes.append("transliterated from Indic script")
                name = name_latn
            else:
                name = unidecode(raw)
                notes.append("romanized with unidecode")
        except Exception:
            pass

    # Title case
    name = " ".join(w.capitalize() for w in name.split())

    # Collapse multiple spaces
    name = re.sub(r"\s{2,}", " ", name).strip()

    return name, notes


def parse_date(value: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse date strings to ISO format YYYY-MM-DD. Returns (iso_date, note)."""
    if not value:
        return None, None

    txt = value.strip()

    # Replace Hindi months with English if any
    month_map = {
        'जनवरी': 'January', 'फ़रवरी': 'February', 'फरवरी': 'February', 'मार्च': 'March', 'अप्रैल': 'April',
        'मई': 'May', 'जून': 'June', 'जुलाई': 'July', 'अगस्त': 'August', 'सितंबर': 'September', 'अक्टूबर': 'October',
        'नवंबर': 'November', 'दिसंबर': 'December'
    }
    for hi, en in month_map.items():
        txt = txt.replace(hi, en)

    # Try multiple formats
    fmts = [
        "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%Y-%m-%d", "%d %B %Y", "%d %b %Y",
        "%m/%d/%Y", "%d-%b-%Y", "%d/%m/%y", "%d-%m-%y"
    ]
    for fmt in fmts:
        try:
            dt = datetime.strptime(txt, fmt)
            return dt.strftime("%Y-%m-%d"), None
        except Exception:
            continue

    # Only month/year
    m = re.search(r"(\d{1,2})/(\d{4})", txt)
    if m:
        month, year = m.group(1), m.group(2)
        try:
            dt = datetime.strptime(f"01/{month}/{year}", "%d/%m/%Y")
            return dt.strftime("%Y-%m-%d"), "day set to 01"
        except Exception:
            pass

    return None, "unparsable date"


def parse_area(value: str, state_hint: Optional[str] = None) -> Tuple[Optional[float], Optional[str]]:
    """Parse area value with units and convert to hectares.
    Returns (area_ha, note)."""
    if not value:
        return None, None

    txt = value.lower().replace(',', ' ').strip()

    # Patterns: number + unit
    patterns = [
        (r"(\d+(?:\.\d+)?)\s*(ha|hectare|hectares)", 'ha'),
        (r"(\d+(?:\.\d+)?)\s*(acre|acres)", 'acre'),
        (r"(\d+(?:\.\d+)?)\s*(sqm|sq\.?\s*m|square\s*meter|square\s*meters)", 'sqm'),
        (r"(\d+(?:\.\d+)?)\s*(sq\.?\s*ft|sqft|square\s*feet)", 'sqft'),
        (r"(\d+(?:\.\d+)?)\s*(bigha|biga|beegha)", 'bigha'),
        (r"(\d+(?:\.\d+)?)\s*(guntha|gunta)", 'guntha')
    ]

    for pattern, unit in patterns:
        m = re.search(pattern, txt)
        if m:
            num = float(m.group(1))
            if unit == 'ha':
                return num, None
            elif unit == 'acre':
                return round(num * Config.AREA_CONVERSIONS['acre'], 6), "converted from acres"
            elif unit == 'sqm':
                return round(num * Config.AREA_CONVERSIONS['sqm'], 6), "converted from sqm"
            elif unit == 'sqft':
                return round(num * Config.AREA_CONVERSIONS['sqft'], 6), "converted from sqft"
            elif unit == 'guntha':
                return round(num * Config.AREA_CONVERSIONS['guntha'], 6), "converted from guntha"
            elif unit == 'bigha':
                # Use state hint to pick correct bigha
                if state_hint and state_hint.lower() in ['madhya pradesh', 'rajasthan']:
                    cf = Config.AREA_CONVERSIONS['bigha_raj']
                    note = "converted from bigha (MP/RJ pucca)"
                elif state_hint and state_hint.lower() in ['west bengal', 'assam']:
                    cf = Config.AREA_CONVERSIONS['bigha_wb']
                    note = "converted from bigha (WB/Assam)"
                else:
                    # Ambiguous bigha
                    return None, "ambiguous bigha unit without state context"
                return round(num * cf, 6), note

    # If only number detected without unit
    if re.search(r"\b\d+(?:\.\d+)?\b", txt):
        return None, "numeric value without unit"

    return None, "no area found"


def normalize_state(value: str) -> Optional[str]:
    if not value:
        return None
    txt = value.strip()
    # Map abbreviations
    full = Config.STATE_ABBREVIATIONS.get(txt.upper())
    if full:
        return full
    # Hindi to English simple replacements
    replacements = {
        'मध्य प्रदेश': 'Madhya Pradesh', 'छत्तीसगढ़': 'Chhattisgarh', 'झारखंड': 'Jharkhand',
        'ओडिशा': 'Odisha', 'महाराष्ट्र': 'Maharashtra', 'राजस्थान': 'Rajasthan'
    }
    return replacements.get(txt, txt)


def normalize_claim_type(value: str) -> Optional[str]:
    if not value:
        return None
    txt = value.strip()
    for k, vals in Config.CLAIM_TYPE_MAPPINGS.items():
        if any(v.lower() in txt.lower() for v in vals):
            return k
    return 'unknown'


def clean_patta_id(value: str) -> Optional[str]:
    if not value:
        return None
    txt = value.strip()
    txt = txt.replace(' ', '')
    txt = txt.replace('/', '-').upper()
    # Fix common OCR digit/letter confusions
    txt = txt.replace('O', '0').replace('I', '1').replace('L', '1')
    # Keep alnum and hyphen
    txt = re.sub(r"[^A-Z0-9\-]", "", txt)
    return txt

