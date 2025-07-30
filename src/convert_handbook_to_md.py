#!/usr/bin/env python3
"""
convert_handbook_to_md.py – High-quality PDF→Markdown converter tailored for the
UCL Computer Science Student Handbook.

Why a new script?
-----------------
`src/inspect_chunking.py` was optimised for the STS handbook. The CS handbook has
extra quirks (multi-level headings, two-column pages, long enumerated lists).
This script generalises that logic and adds:
• Dynamic heading detection (H1/H2/H3) based on relative font size.
• Column-aware reading order to handle two-column layouts.
• Better list preservation (bullets / numbered lists).
• Optional OCR fallback for pages without extractable text.

Usage
-----
python -m src.convert_handbook_to_md \
    --pdf "data/Computer Science Student Handbook 2024-25.pdf" \
    --out data/cs-student-handbook-clean.md

CLI Flags
~~~~~~~~~
--min-length    Minimum chars for a chunk to be kept (default 100).
--ocr           Enable tesseract OCR fallback for image-only pages.
--verbose       Print extra diagnostics.

The resulting markdown is suitable for downstream chunking / embedding with
`src/build_embeddings.py`.
"""
from __future__ import annotations

import argparse
import collections
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import fitz  # PyMuPDF
from langchain.docstore.document import Document

###############################################################################
# Helpers                                                                     #
###############################################################################

def analyse_font_sizes(doc: fitz.Document) -> Tuple[Dict[float, int], float]:
    """Return (histogram, body_size). Body size = most frequent span size."""
    hist = collections.defaultdict(int)
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    hist[round(span["size"], 1)] += 1
    body_size = max(hist, key=hist.get) if hist else 10.0
    return hist, body_size


def detect_two_column(page: fitz.Page) -> bool:
    """Heuristic: if text blocks cluster on left/right halves, treat as 2-column."""
    blocks = [b for b in page.get_text("dict")["blocks"] if b["type"] == 0]
    if len(blocks) < 6:
        return False
    page_width = page.rect.width
    left, right = 0, 0
    for b in blocks:
        if b["bbox"][0] < page_width * 0.55:
            left += 1
        else:
            right += 1
    # Consider two-column if both sides have at least 25% of blocks
    return left > len(blocks) * 0.25 and right > len(blocks) * 0.25


def ocr_page(page: fitz.Page) -> str:
    """Run Tesseract OCR on a rasterised page (PNG) and return text."""
    import tempfile
    import pytesseract
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = Path(tmpdir) / "page.png"
        pix = page.get_pixmap(dpi=300)
        pix.save(img_path.as_posix())
        text = pytesseract.image_to_string(img_path, lang="eng")
    return text

###############################################################################
# Main conversion logic                                                       #
###############################################################################

BULLET_REGEX = re.compile(r"^\s*[\u2022\u2023\u25E6\u2043\u2219\-]\s+")
NUMBERED_REGEX = re.compile(r"^\s*\d+([.)])\s+")


def spans_to_markdown(line_spans, link_lookup, heading_level):
    """Convert list of span dicts to a single markdown line."""
    # Step 1: stitch consecutive spans that share link
    line_text = ""
    i = 0
    while i < len(line_spans):
        span = line_spans[i]
        span_rect = fitz.Rect(span["bbox"])
        link_uri = next((uri for r, uri in link_lookup.items() if r.intersects(span_rect)), None)
        cur_text = span["text"]
        j = i + 1
        while j < len(line_spans):
            nxt = line_spans[j]
            nxt_rect = fitz.Rect(nxt["bbox"])
            nxt_uri = next((uri for r, uri in link_lookup.items() if r.intersects(nxt_rect)), None)
            if nxt_uri == link_uri:
                cur_text += nxt["text"]
                j += 1
            else:
                break
        if link_uri:
            line_text += f"[{cur_text}]({link_uri})"
        else:
            line_text += cur_text
        i = j
    # Add heading hashes if needed
    if heading_level:
        line_text = f"{'#' * heading_level} {line_text.strip()}"
    return line_text.strip()


def process_page(page: fitz.Page, body_size: float, min_heading_increase: float, ocr: bool, verbose: bool):
    """Return markdown lines for a page."""
    if verbose:
        print(f"Processing page {page.number + 1} …")

    blocks_dict = page.get_text("dict")
    if not blocks_dict["blocks"] and ocr:
        return ocr_page(page).splitlines()

    two_col = detect_two_column(page)
    blocks = [b for b in blocks_dict["blocks"] if b["type"] == 0]
    # Sort reading order: y, then x for single column; y, then column buckets for 2-col
    if two_col:
        mid_x = page.rect.width * 0.5
        blocks.sort(key=lambda b: (b["bbox"][1], 0 if b["bbox"][0] < mid_x else 1, b["bbox"][0]))
    else:
        blocks.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))

    markdown_lines: List[str] = []
    link_lookup = {fitz.Rect(lk["from"]): lk["uri"] for lk in page.get_links() if "uri" in lk}

    for block in blocks:
        for line in block["lines"]:
            spans = line["spans"]
            if not spans:
                continue
            first_span = spans[0]
            size_diff = first_span["size"] - body_size
            heading_level = 0
            if size_diff > min_heading_increase * 2:  # big title
                heading_level = 1
            elif size_diff > min_heading_increase:
                heading_level = 2
            elif size_diff > min_heading_increase * 0.5:
                heading_level = 3

            md_line = spans_to_markdown(spans, link_lookup, heading_level)
            # Preserve lists
            if not heading_level:
                if BULLET_REGEX.match(md_line):
                    md_line = "- " + BULLET_REGEX.sub("", md_line)
                elif NUMBERED_REGEX.match(md_line):
                    # Keep original numbering for ordered lists
                    pass
            markdown_lines.append(md_line)
    return markdown_lines


def reconstruct_pdf_to_markdown(pdf_path: str, ocr: bool = False, verbose: bool = False) -> str:
    doc = fitz.open(pdf_path)
    _, body_size = analyse_font_sizes(doc)
    min_heading_increase = 1.5  # pts over body_size to count as heading

    all_lines: List[str] = []
    header_candidates = collections.Counter()
    footer_candidates = collections.Counter()

    for page in doc:
        lines = process_page(page, body_size, min_heading_increase, ocr, verbose)
        if not lines:
            continue
        # Track possible headers/footers
        header_candidates[lines[0].strip()] += 1
        footer_candidates[lines[-1].strip()] += 1
        all_lines.extend(lines)
        all_lines.append("")  # page break

    doc.close()

    # Remove headers/footers repeating on > 60% of pages
    pages = len(header_candidates)
    common_headers = {h for h, c in header_candidates.items() if c > pages * 0.6 and h}
    common_footers = {f for f, c in footer_candidates.items() if c > pages * 0.6 and f}
    cleaned_lines = [ln for ln in all_lines if ln.strip() not in common_headers | common_footers]

    md_text = "\n".join(cleaned_lines)
    # Collapse repeated blank lines
    md_text = re.sub(r"\n{3,}", "\n\n", md_text)

    # Remove standalone Table of Contents section
    toc_match = re.search(r"^#*\s*Contents.*?$", md_text, flags=re.MULTILINE | re.IGNORECASE)
    if toc_match:
        start = toc_match.start()
        end = md_text.find("\n#", start + 1)
        if end != -1:
            md_text = md_text[:start] + md_text[end:]
    return md_text.strip()

###############################################################################
# Chunking helpers                                                            #
###############################################################################

def chunk_by_headings(md: str, min_length: int, source: str) -> List[Document]:
    """Split markdown by H1/H2/H3 headings into LangChain Document list."""
    splits = re.split(r"\n(?=##?#+?\s)", md)  # splits before any heading level 1–3
    docs: List[Document] = []
    for chunk in splits:
        chunk = chunk.strip()
        if len(chunk) < min_length:
            continue
        docs.append(Document(page_content=chunk, metadata={"source": source}))
    return docs

###############################################################################
# CLI                                                                        #
###############################################################################

def main():
    p = argparse.ArgumentParser(description="Convert CS Handbook PDF to clean Markdown.")
    p.add_argument("--pdf", required=True, help="Path to the handbook PDF.")
    p.add_argument("--out", required=True, help="Output markdown path.")
    p.add_argument("--min-length", type=int, default=100, help="Minimum chars per chunk to keep (for optional chunk export).")
    p.add_argument("--chunk-out", help="If set, also save heading-based chunks as a combined markdown file.")
    p.add_argument("--ocr", action="store_true", help="Enable Tesseract OCR fallback for image-only pages.")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if args.verbose:
        print("Starting conversion…")
    md_text = reconstruct_pdf_to_markdown(args.pdf, ocr=args.ocr, verbose=args.verbose)

    # Save full markdown
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md_text, encoding="utf-8")
    print(f"Saved full markdown to {out_path}")

    if args.chunk_out:
        chunks = chunk_by_headings(md_text, args.min_length, args.pdf)
        sep = "\n\n#################################################################\n\n"
        chunk_text = sep.join(d.page_content for d in chunks)
        Path(args.chunk_out).write_text(chunk_text, encoding="utf-8")
        print(f"Saved {len(chunks)} cleaned chunks to {args.chunk_out}")


if __name__ == "__main__":
    main()
