#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sae_table_linearizer.py (generic)
---------------------------------
Very light heuristics to linearize plain-text numeric tables commonly found in filings.
- No company-specific segment lists.
- Works best on blocks where each row has a label followed by one or more numeric columns.

Exports:
    linearize_tables(text: str, section_hint: str | None = None) -> str
      Returns a string to append to your document (or "" if nothing found).

Heuristics:
- Optionally bias search around a section hint (e.g., "Sales:", "Net sales", "Results of Operations").
- Identify blocks separated by blank lines with >= 3 lines having >= 1 numeric token.
- For each line in a block, split into (label, numbers...). Label = leading non-numeric text (trimmed).
- Emit stable key=value lines:
    TABLE=<index> | ROW=<row_index> | LABEL=<label> | COL1=<n1> | COL2=<n2> | ... | UNIT=<unit?>
- Detect a unit string like "(Dollars in thousands)" near the block (within ±5 lines).

Note: keeps numbers as they appear (removes commas, keeps optional leading $); casts to plain digits.
"""
from __future__ import annotations
import re
from typing import List, Tuple

def _find_candidate_blocks(lines: List[str]) -> List[Tuple[int,int]]:
    blocks = []
    i = 0
    N = len(lines)
    while i < N:
        # skip blank lines
        while i < N and lines[i].strip() == "":
            i += 1
        if i >= N: break
        j = i
        numeric_lines = 0
        while j < N and lines[j].strip() != "":
            if re.search(r"\d", lines[j]):
                numeric_lines += 1
            j += 1
        # A candidate "table-like" block has >=3 numeric lines
        if numeric_lines >= 3:
            blocks.append((i, j))  # [i, j)
        i = j
    return blocks

def _label_and_numbers(ln: str):
    # Extract a leading text label and the trailing numbers
    # Strategy: split by two+ spaces or tab first
    parts = re.split(r"[ \t]{2,}", ln.strip())
    if len(parts) >= 2:
        label = parts[0]
        tail = " ".join(parts[1:])
    else:
        # fallback: take non-numeric prefix as label
        m = re.match(r"([^\d$]+)(.*)", ln.strip())
        if m:
            label, tail = m.group(1).strip(), m.group(2).strip()
        else:
            label, tail = ln.strip(), ""
    # now parse numbers from tail (allow $, commas, decimals, parens)
    nums = []
    for m in re.finditer(r"[\$\(]?\s*[-+]?\d[\d,]*(?:\.\d+)?\)?", tail):
        tok = m.group(0).strip()
        if not re.search(r"\d", tok):
            continue
        # normalize ($1,234) -> -1234 ; $1,234.56 -> 1234.56 ; (123) -> -123
        neg = tok.startswith("(") and tok.endswith(")")
        tok = tok.replace("$", "").replace(",", "").replace("(", "").replace(")", "")
        try:
            val = float(tok) if "." in tok else int(tok)
        except Exception:
            continue
        if neg and isinstance(val, (int, float)):
            val = -val
        nums.append(val)
    return label, nums

def _nearby_unit(lines: List[str], start: int, end: int) -> str:
    lo = max(0, start-5); hi = min(len(lines), end+5)
    window = " ".join(lines[lo:hi])
    m = re.search(r"\(([^)]+)\)", window)
    if m:
        s = m.group(1)
        if any(u in s.lower() for u in ["dollar", "thousand", "million", "currency"]):
            return s.strip()
    return ""

def linearize_tables(text: str, section_hint: str | None = None) -> str:
    lines = text.splitlines()
    # If a hint is provided, rotate the text so blocks near the hint are prioritized
    if section_hint:
        hint_idx = None
        low = section_hint.lower()
        for i, ln in enumerate(lines):
            if low in ln.lower():
                hint_idx = i; break
        if hint_idx is not None:
            # Process lines starting near hint by moving that slice to the front
            lines = lines[hint_idx:] + [""] + lines[:hint_idx]

    blocks = _find_candidate_blocks(lines)
    out_lines = []
    table_idx = 0
    for (i, j) in blocks:
        unit = _nearby_unit(lines, i, j)
        rows = [ln for ln in lines[i:j] if ln.strip()]
        # Filter to lines that appear to have a label + numbers
        parsed = []
        for ln in rows:
            label, nums = _label_and_numbers(ln)
            if label and len(nums) >= 1:
                parsed.append((label, nums))
        # Require at least 3 parsed rows to reduce false positives
        if len(parsed) < 3:
            continue
        # Emit
        header = f"TABLE_LINEARIZED index={table_idx}" + (f" | UNIT={unit}" if unit else "")
        out_lines.append(header)
        for r_idx, (label, nums) in enumerate(parsed):
            cols = " ".join([f"| COL{k+1}={nums[k]}" for k in range(len(nums))])
            out_lines.append(f"ROW={r_idx} | LABEL={label}{(' ' + cols) if cols else ''}")
        out_lines.append("")  # blank separator
        table_idx += 1

    return "\n".join(out_lines).strip()
