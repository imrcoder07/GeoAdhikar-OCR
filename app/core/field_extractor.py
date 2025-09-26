"""
Field extraction logic for FRA OCR project
Combines regex-based extraction with structured output expected by downstream normalization
"""
from __future__ import annotations
import re
from typing import Dict, Optional, Tuple, Any
from loguru import logger
from config.config import Config
from app.utils.normalization import (
    normalize_name,
    parse_date,
    parse_area,
    normalize_state,
    normalize_claim_type,
    clean_patta_id,
)

class FieldExtractor:
    """Extract fields from raw OCR text using regex heuristics"""

    def __init__(self):
        self.patterns = Config.FIELD_PATTERNS

    def extract(self, raw_text: str) -> Dict[str, Dict[str, Any]]:
        """Return a dict of extracted fields with raw snippets"""
        text = raw_text or ""
        out: Dict[str, Dict[str, Any]] = {}

        # Patta ID
        patta_raw = self._search_first(text, self.patterns['patta_id'])
        patta_val = clean_patta_id(patta_raw) if patta_raw else None
        confidence = 0.82 if patta_val else 0.4
        if confidence < 0.6:
            patta_val = None
        out['patta_id'] = {
            'raw': patta_raw,
            'value': patta_val,
            'confidence': confidence,
            'source': 'Regex'
        }

        # Holder Name
        holder_raw = self._search_first(text, self.patterns['holder_name'])
        holder_val, holder_notes = normalize_name(holder_raw) if holder_raw else (None, [])
        confidence = 0.7 if holder_val else 0.4
        if confidence < 0.6:
            holder_val = None
        out['holder_name'] = {
            'raw': holder_raw,
            'value': holder_val,
            'confidence': confidence,
            'source': 'Regex',
            'note': "; ".join(holder_notes) if holder_notes else None
        }

        # Relative Name
        relative_raw = self._search_first(text, [r"(?:Father|Father's|पिता|S/o|S\/o)\s*[:\-]?\s*([^\n]+)"])
        relative_val, _ = normalize_name(relative_raw) if relative_raw else (None, [])
        confidence = 0.6 if relative_val else 0.4
        if confidence < 0.6:
            relative_val = None
        out['relative_name'] = {
            'raw': relative_raw,
            'value': relative_val,
            'relation_type': 'father' if relative_raw else None,
            'confidence': confidence,
            'source': 'Regex'
        }

        # Village
        village_raw = self._search_first(text, self.patterns['village'])
        village_val = village_raw.strip() if village_raw else None
        confidence = 0.65 if village_val else 0.4
        if confidence < 0.6:
            village_val = None
        out['village'] = {
            'raw': village_raw,
            'value': village_val,
            'confidence': confidence,
            'source': 'Regex',
            'candidates': []
        }

        # District
        district_raw = self._search_first(text, self.patterns['district'])
        district_val = district_raw.strip() if district_raw else None
        confidence = 0.6 if district_val else 0.4
        if confidence < 0.6:
            district_val = None
        out['district'] = {
            'raw': district_raw,
            'value': district_val,
            'confidence': confidence,
            'source': 'Regex'
        }

        # State
        state_raw = self._search_first(text, [r"State\s*[:\-]?\s*([^\n,]+)", r"राज्य\s*[:\-]?\s*([^\n,]+)"])
        state_val = normalize_state(state_raw) if state_raw else None
        confidence = 0.7 if state_val else 0.4
        if confidence < 0.6:
            state_val = None
        out['state'] = {
            'raw': state_raw,
            'value': state_val,
            'confidence': confidence,
            'source': 'Regex'
        }

        # Area
        area_raw = self._search_first(text, [
            r"Area\s*[:\-]?\s*([^\n]+)", r"क्षेत्रफल\s*[:\-]?\s*([^\n]+)"
        ])
        area_val, area_note = parse_area(area_raw, state_val) if area_raw else (None, None)
        confidence = 0.8 if area_val else 0.4
        if confidence < 0.6:
            area_val = None
        out['area'] = {
            'raw': area_raw,
            'value': area_val,
            'unit': 'ha' if area_val else None,
            'confidence': confidence,
            'source': 'Regex',
            'note': area_note
        }

        # Issue date
        date_raw = self._search_first(text, [r"Issue\s*Date\s*[:\-]?\s*([^\n]+)", r"Date\s*[:\-]?\s*([^\n]+)", r"दिनांक\s*[:\-]?\s*([^\n]+)", r"इति\s*तिथि\s*[:\-]?\s*([^\n]+)"])
        date_val, date_note = parse_date(date_raw) if date_raw else (None, None)
        confidence = 0.8 if date_val else 0.4
        if confidence < 0.6:
            date_val = None
        out['issue_date'] = {
            'raw': date_raw,
            'value': date_val,
            'confidence': confidence,
            'source': 'Regex',
            'note': date_note
        }

        # Claim type
        claim_raw = self._search_first(text, [r"Claim\s*Type\s*[:\-]?\s*([^\n]+)", r"अधिकार\s*प्रकार\s*[:\-]?\s*([^\n]+)"])
        claim_val = normalize_claim_type(claim_raw) if claim_raw else None
        confidence = 0.75 if claim_val and claim_val != 'unknown' else 0.4
        if confidence < 0.6:
            claim_val = None
        out['claim_type'] = {
            'raw': claim_raw,
            'value': claim_val,
            'confidence': confidence,
            'source': 'Regex'
        }

        # Plot No
        plot_raw = self._search_first(text, self.patterns['plot_no'])
        plot_val = plot_raw.strip() if plot_raw else None
        confidence = 0.75 if plot_val else 0.4
        if confidence < 0.6:
            plot_val = None
        out['plot_no'] = {
            'raw': plot_raw,
            'value': plot_val,
            'confidence': confidence,
            'source': 'Regex'
        }

        # Status
        status_raw = self._search_first(text, [r"Status\s*[:\-]?\s*([^\n]+)", r"स्थिति\s*[:\-]?\s*([^\n]+)"])
        status_val = None
        if status_raw:
            sr = status_raw.lower()
            if 'grant' in sr or 'मंजूर' in sr or 'स्वीकृत' in sr:
                status_val = 'granted'
            elif 'reject' in sr or 'अस्वीक' in sr:
                status_val = 'rejected'
            elif 'pending' in sr or 'लंबित' in sr:
                status_val = 'pending'
            elif 'verify' in sr or 'प्रमाणित' in sr:
                status_val = 'verified'
        confidence = 0.7 if status_val else 0.4
        if confidence < 0.6:
            status_val = None
        out['status'] = {
            'raw': status_raw,
            'value': status_val,
            'confidence': confidence,
            'source': 'Regex'
        }

        return out

    def _search_first(self, text: str, patterns: list[str]) -> Optional[str]:
        for p in patterns:
            m = re.search(p, text, flags=re.IGNORECASE)
            if m:
                # return first capturing group if exists else full match
                return (m.group(1) if m.groups() else m.group(0)).strip()
        return None

