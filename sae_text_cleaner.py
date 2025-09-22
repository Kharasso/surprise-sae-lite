#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sae_text_cleaner.py  (minimal, generic)
--------------------------------------
Purpose: very light cleanup for long-form financial/technical text before encoding into an SAE.
- Keep semantics and formatting as intact as possible.
- No table parsing. No domain-specific keywords. No number/currency edits.

What it does:
1) Normalize Unicode to NFC.
2) Normalize newlines to '\n' and remove form feeds.
3) Fix common line-break hyphenation: "manu-\nfacturing" -> "manufacturing".
   (Only when both sides look like word characters.)
4) Trim trailing spaces; convert tabs to single spaces.
5) Collapse *excess* blank lines: allow at most one consecutive blank line.
6) Optionally drop obvious page artifacts (very conservative):
   - a line that is exactly "Table of Contents" (case-insensitive)
   - a line that is only digits (1-3 chars) and is surrounded by blank lines

Everything else is preserved.

CLI:
  python sae_text_cleaner.py input.txt -o cleaned.txt

Module:
  from sae_text_cleaner import clean_text
  cleaned = clean_text(raw_text)
"""
from __future__ import annotations
import re
import unicodedata
import argparse

__all__ = ["clean_text"]

def _normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text)

def _normalize_newlines(text: str) -> str:
    # Convert CRLF/CR to LF; remove form feeds
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x0c", "")  # form feed
    return text

def _fix_hyphenation(text: str) -> str:
    # Join hyphenated breaks when both sides are word chars (avoid touching bullets like "-\n")
    # Example: "multi-\nline" -> "multiline"
    pattern = re.compile(r"(?<=\w)-\n(?=\w)")
    return pattern.sub("", text)

def _trim_and_tabs(text: str) -> str:
    # Trim trailing spaces per line and replace tabs with a single space
    lines = text.split("\n")
    lines = [ln.rstrip().replace("\t", " ") for ln in lines]
    return "\n".join(lines)

def _collapse_blank_lines(text: str) -> str:
    # Allow at most one consecutive blank line
    return re.sub(r"\n{3,}", "\n\n", text)

def _drop_conservative_artifacts(text: str) -> str:
    lines = text.split("\n")
    out = []
    for i, ln in enumerate(lines):
        s = ln.strip()
        # Drop "Table of Contents" exactly
        if s.lower() == "table of contents":
            continue
        # Drop numeric-only page numbers if surrounded by blank lines (likely page footer/header)
        if s.isdigit() and 1 <= len(s) <= 3:
            prev_blank = (i == 0) or (lines[i-1].strip() == "")
            next_blank = (i == len(lines)-1) or (lines[i+1].strip() == "")
            if prev_blank and next_blank:
                continue
        out.append(ln)
    return "\n".join(out)

def clean_text(text: str) -> str:
    """
    Minimal generic cleaning; see module docstring.
    """
    text = _normalize_unicode(text)
    text = _normalize_newlines(text)
    text = _fix_hyphenation(text)
    text = _trim_and_tabs(text)
    text = _collapse_blank_lines(text)
    text = _drop_conservative_artifacts(text)
    return text.strip()

def main():
    ap = argparse.ArgumentParser(description="Minimal generic text cleaner for SAE ingestion.")
    ap.add_argument("input", help="Input UTF-8 text file")
    ap.add_argument("-o", "--output", default="cleaned.txt", help="Output path")
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    cleaned = clean_text(raw)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(cleaned)

if __name__ == "__main__":
    main()
