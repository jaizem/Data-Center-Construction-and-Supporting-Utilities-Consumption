# this version follows ... and ...2.
"""
Option 4: FREE enrichment (DuckDuckGo only), no OpenAI / no paid APIs.

Per row:
  - One DuckDuckGo query via ddgs (formerly duckduckgo_search)
  - Parse snippets/titles to extract best-effort:
      CAMPUS_CITY
      operational_date (Month Day, Year | Month Year | Year)
      INITIAL_CAPACITY_MW (number near "MW")
      RENOVATED (heuristic)
      INFERRED_FROM_CAMPUS (heuristic)
      sources: URLs of the result that yielded each field
      DATE_EVIDENCE / MW_EVIDENCE: short tags describing extraction path

Install:
  pip install pandas ddgs
(or older: pip install duckduckgo_search)

Run:
  python enrich_dc_ddg_free.py
"""

from __future__ import annotations

import os
import re
import json
import time
import random
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# ✅ compatible with both new and old package/module names
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS


# -----------------------------
# Config
# -----------------------------
DDG_MAX_RESULTS = 10

# Throttling (increase if you get blocked)
SLEEP_BETWEEN_ROWS = 1.2
SLEEP_ON_RETRY_RANGE = (2.5, 7.0)
MAX_DDG_RETRIES = 4

# Cache
CACHE_JSONL = "dc_ddg_cache.jsonl"

# Output columns
OUT_COLS = [
    "CAMPUS_CITY",
    "city_source",
    "operational_date",
    "date_source",
    "DATE_EVIDENCE",
    "INITIAL_CAPACITY_MW",
    "capacity_source",
    "MW_EVIDENCE",
    "RENOVATED",
    "INFERRED_FROM_CAMPUS",
]

# -----------------------------
# Text helpers
# -----------------------------
def _norm(x: Any) -> str:
    return "" if pd.isna(x) else str(x).strip()

def _clean_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def _cache_key(parts: Tuple[str, ...]) -> str:
    s = "\n".join(parts)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def load_cache(path: str = CACHE_JSONL) -> Dict[str, dict]:
    cache: Dict[str, dict] = {}
    if not os.path.exists(path):
        return cache
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cache[obj["key"]] = obj["value"]
    return cache

def append_cache(key: str, value: dict, path: str = CACHE_JSONL) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")


# -----------------------------
# DuckDuckGo search
# -----------------------------
def ddg_search_once(query: str, max_results: int = DDG_MAX_RESULTS) -> List[Dict[str, str]]:
    last_err: Optional[Exception] = None

    for _ in range(MAX_DDG_RETRIES):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            out: List[Dict[str, str]] = []
            for r in results:
                out.append({
                    "name": r.get("title", "") or "",
                    "snippet": r.get("body", "") or "",
                    "url": r.get("href", "") or "",
                })
            if out:
                return out
        except Exception as e:
            last_err = e

        time.sleep(random.uniform(*SLEEP_ON_RETRY_RANGE))

    # optional logging:
    # if last_err:
    #     print(f"[DDG] failed: {query} :: {last_err}")
    return []


# -----------------------------
# Extraction heuristics
# -----------------------------

# MW
MW_RE = re.compile(r"(?<!\w)(\d{1,4}(?:\.\d{1,3})?)\s?(?:MW|megawatt(?:s)?)\b", re.IGNORECASE)

# Dates
MONTHS = ("Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|"
          "Sep|Sept|September|Oct|October|Nov|November|Dec|December")
DATE_FULL_RE = re.compile(rf"\b({MONTHS})\s+(\d{{1,2}}),\s*(\d{{4}})\b", re.IGNORECASE)
DATE_MONTH_YEAR_RE = re.compile(rf"\b({MONTHS})\s+(\d{{4}})\b", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")

OPENING_HINT_RE = re.compile(
    r"\b(opened|opening|launch(?:ed)?|commission(?:ed)?|operational|in service|"
    r"began operations|go[-\s]?live|went live|inaugurat(?:ed|ion))\b",
    re.IGNORECASE,
)

RENOVATION_HINT_RE = re.compile(
    r"\b(expand(?:ed|sion)?|upgrade(?:d)?|renovat(?:ed|ion)|retrofit|"
    r"additional capacity|phase\s*\d+|second phase|new hall|new data hall|extension)\b",
    re.IGNORECASE,
)

CAMPUS_HINT_RE = re.compile(r"\b(campus|metro|market|region|multiple facilities)\b", re.IGNORECASE)

CITY_STATE_RE = re.compile(r"\b([A-Z][a-z]+(?:[ -][A-Z][a-z]+){0,3}),\s*([A-Z]{2})\b")
IN_CITY_RE = re.compile(r"\b(?:in|near)\s+([A-Z][a-z]+(?:[ -][A-Z][a-z]+){0,3})(?:,\s*([A-Z]{2}))?\b")


@dataclass
class Extracted:
    CAMPUS_CITY: str = ""
    city_source: str = ""
    operational_date: str = ""
    date_source: str = ""
    INITIAL_CAPACITY_MW: str = ""
    capacity_source: str = ""
    RENOVATED: str = ""  # "1"/""
    INFERRED_FROM_CAMPUS: str = ""  # "1"/"0"/""
    DATE_EVIDENCE: str = ""
    MW_EVIDENCE: str = ""


def build_query(row: pd.Series) -> str:
    name = _norm(row.get("DATA_CENTER_BUILDING_NAME"))
    provider = _norm(row.get("PROVIDER_NAME"))
    state = _norm(row.get("STATE_CODE")) or _norm(row.get("STATE_NAME"))
    q = f'"{name}" "{provider}" {state} data center city opened operational initial capacity MW commissioning'
    return _clean_spaces(q)


def _text_blocks(results: List[Dict[str, str]]) -> List[Tuple[str, str]]:
    blocks: List[Tuple[str, str]] = []
    for r in results:
        url = r.get("url", "") or ""
        title = r.get("name", "") or ""
        snippet = r.get("snippet", "") or ""
        blocks.append((url, f"{title}\n{snippet}".strip()))
    return blocks


def pick_city(blocks: List[Tuple[str, str]], state_code: str) -> Tuple[str, str]:
    st = (state_code or "").strip().upper()
    candidates: List[Tuple[int, str, str]] = []  # (score, city, url)

    for url, text in blocks:
        for m in CITY_STATE_RE.finditer(text):
            city, st2 = m.group(1), m.group(2)
            score = 5
            if st and st2 == st:
                score += 3
            candidates.append((score, city, url))

        for m in IN_CITY_RE.finditer(text):
            city, st2 = m.group(1), (m.group(2) or "")
            score = 2
            if st and st2 and st2.upper() == st:
                score += 2
            candidates.append((score, city, url))

    if not candidates:
        return ("", "")
    candidates.sort(key=lambda x: x[0], reverse=True)
    _, city, url = candidates[0]
    return (city, url)


def pick_operational_date(blocks: List[Tuple[str, str]]) -> Tuple[str, str, bool, str]:
    """
    Returns (date_str, url, campusish, evidence_tag)
    """
    best: Tuple[int, str, str, bool, str] = (0, "", "", False, "no_date_found")

    for url, text in blocks:
        has_open = bool(OPENING_HINT_RE.search(text))
        campusish = bool(CAMPUS_HINT_RE.search(text))
        base = 5 if has_open else 0

        m = DATE_FULL_RE.search(text)
        if m:
            date = f"{m.group(1)} {m.group(2)}, {m.group(3)}"
            tag = "full_date+opening_hint" if has_open else "full_date_no_opening_hint"
            score = base + 6
            if score > best[0]:
                best = (score, date, url, campusish, tag)
            continue

        m = DATE_MONTH_YEAR_RE.search(text)
        if m:
            date = f"{m.group(1)} {m.group(2)}"
            tag = "month_year+opening_hint" if has_open else "month_year_no_opening_hint"
            score = base + 3
            if score > best[0]:
                best = (score, date, url, campusish, tag)
            continue

        m = YEAR_RE.search(text)
        if m and has_open:
            date = m.group(1)
            tag = "year_only+opening_hint"
            score = base + 1
            if score > best[0]:
                best = (score, date, url, campusish, tag)

    return (best[1], best[2], best[3], best[4])


def pick_capacity_mw(blocks: List[Tuple[str, str]]) -> Tuple[str, str, bool, str]:
    """
    Returns (mw_str, url, campusish, evidence_tag)
    """
    best: Tuple[int, str, str, bool, str] = (0, "", "", False, "no_mw_found")

    for url, text in blocks:
        campusish = bool(CAMPUS_HINT_RE.search(text))
        t = text.lower()
        strong_hint = ("initial" in t) or ("commission" in t) or ("phase 1" in t) or ("phase i" in t)

        m = MW_RE.search(text)
        if not m:
            continue

        mw = m.group(1)
        score = 2
        if strong_hint:
            score += 3
            tag = "mw+strong_hint(initial|commission|phase1)"
        else:
            tag = "mw_weak(no_initial_hint)"
        if score > best[0]:
            best = (score, mw, url, campusish, tag)

    return (best[1], best[2], best[3], best[4])


def infer_renovated(blocks: List[Tuple[str, str]]) -> str:
    for _, text in blocks:
        if RENOVATION_HINT_RE.search(text):
            return "1"
    return ""


def extract_from_results(row: pd.Series, results: List[Dict[str, str]]) -> Extracted:
    blocks = _text_blocks(results)
    state_code = (_norm(row.get("STATE_CODE")) or "").upper()

    city, city_url = pick_city(blocks, state_code)
    op_date, date_url, date_campusish, date_ev = pick_operational_date(blocks)
    mw, mw_url, mw_campusish, mw_ev = pick_capacity_mw(blocks)

    renovated = infer_renovated(blocks)

    inferred = ""
    if (op_date and date_campusish) or (mw and mw_campusish):
        inferred = "1"
    elif op_date or mw:
        inferred = "0"

    # ✅ FIX: return the evidence tags so they actually get written
    return Extracted(
        CAMPUS_CITY=city,
        city_source=city_url if city else "",
        operational_date=op_date,
        date_source=date_url if op_date else "",
        DATE_EVIDENCE=(date_ev if op_date else "no_date_found"),
        INITIAL_CAPACITY_MW=mw,
        capacity_source=mw_url if mw else "",
        MW_EVIDENCE=(mw_ev if mw else "no_mw_found"),
        RENOVATED=renovated,
        INFERRED_FROM_CAMPUS=inferred,
    )


# -----------------------------
# Main driver
# -----------------------------
def enrich_csv(in_csv: str, out_csv: str, max_rows: Optional[int] = None) -> None:
    df = pd.read_csv(in_csv)

    for col in OUT_COLS:
        if col not in df.columns:
            df[col] = ""

    cache = load_cache()
    n = len(df) if max_rows is None else min(len(df), max_rows)

    for i in range(n):
        row = df.iloc[i]

        # ✅ More robust resume-safe skip: skip if ANY enrichment field is already filled
        if any(str(df.at[i, col]).strip() for col in OUT_COLS):
            continue

        query = build_query(row)
        ck = _cache_key((query,))

        if ck in cache:
            payload = cache[ck]
            results = payload.get("results", [])
        else:
            results = ddg_search_once(query, max_results=DDG_MAX_RESULTS)
            payload = {"query": query, "results": results}
            cache[ck] = payload
            append_cache(ck, payload)
            time.sleep(SLEEP_BETWEEN_ROWS)

        extracted = extract_from_results(row, results)

        df.at[i, "CAMPUS_CITY"] = extracted.CAMPUS_CITY
        df.at[i, "city_source"] = extracted.city_source
        df.at[i, "operational_date"] = extracted.operational_date
        df.at[i, "date_source"] = extracted.date_source
        df.at[i, "DATE_EVIDENCE"] = extracted.DATE_EVIDENCE
        df.at[i, "INITIAL_CAPACITY_MW"] = extracted.INITIAL_CAPACITY_MW
        df.at[i, "capacity_source"] = extracted.capacity_source
        df.at[i, "MW_EVIDENCE"] = extracted.MW_EVIDENCE
        df.at[i, "RENOVATED"] = extracted.RENOVATED
        df.at[i, "INFERRED_FROM_CAMPUS"] = extracted.INFERRED_FROM_CAMPUS

        if (i + 1) % 50 == 0:
            print(f"Processed {i+1}/{n}")

    df.to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    IN_CSV = "/Users/alanbeem/Desktop/CSB430/Data-Center-Construction-and-Supporting-Utilities-Consumption/Aterio US Data Centers Dashboard/US Data Centers Dataset-Table 1.csv"
    OUT_CSV = "US_Data_Centers_enriched_ddg_free.csv"

    # TEST FIRST:
    # enrich_csv(IN_CSV, OUT_CSV.replace(".csv", "_TEST.csv"), max_rows=50)1

    # FULL RUN:
    enrich_csv(IN_CSV, OUT_CSV, max_rows=None)