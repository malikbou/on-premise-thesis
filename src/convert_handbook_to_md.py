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

    # New step: remove common PDF icon/symbol artifacts
    line_text = re.sub(r'[\uf0b7\uf076\uf0e0\uf0a7\uf075\uf071\uf0b2\uf07d\uf07e]', '', line_text)


    # Add heading hashes if needed
    if heading_level:
        line_text = f"{'#' * heading_level} {line_text.strip()}"
    return line_text.strip()


def find_potential_tables(blocks: List[Dict], y_tolerance=5, x_tolerance=10) -> List[List[List[Dict]]]:
    """Identify groups of blocks that form tables based on alignment."""
    rows = find_table_rows(blocks, y_tolerance)
    if not rows:
        return []

    tables = []
    visited_blocks = set()

    for i, row in enumerate(rows):
        # Skip if the first block of the row has been processed
        if not row or id(row[0]) in visited_blocks:
            continue

        # A potential table starts with a row of multiple columns
        if len(row) > 1:
            current_table = [row]
            for block in row:
                visited_blocks.add(id(block))

            # Find subsequent rows that align with the current table's columns
            num_cols = len(row)
            for next_row in rows[i+1:]:
                if len(next_row) == num_cols:
                    # Check for horizontal alignment of columns
                    aligned = True
                    for col_idx in range(num_cols):
                        col_x = row[col_idx]['bbox'][0]
                        next_row_col_x = next_row[col_idx]['bbox'][0]
                        if abs(col_x - next_row_col_x) > x_tolerance:
                            aligned = False
                            break
                    if aligned:
                        current_table.append(next_row)
                        for block in next_row:
                            visited_blocks.add(id(block))
                    else:
                        break # Row doesn't align, so table ends
                else:
                    break # Different number of columns, table ends

            tables.append(current_table)

    return tables


def table_to_markdown(table: List[List[Dict]]) -> str:
    """Converts a table (list of rows of blocks) to a Markdown table string."""
    md_rows = []
    # Header
    header_cells = [blocks_to_text([cell]) for cell in table[0]]
    md_rows.append("| " + " | ".join(header_cells) + " |")
    md_rows.append("|" + "---|" * len(header_cells))
    # Body
    for row in table[1:]:
        row_cells = [blocks_to_text([cell]) for cell in row]
        md_rows.append("| " + " | ".join(row_cells) + " |")
    return "\n".join(md_rows)


def find_table_rows(blocks: List[Dict], y_tolerance=5) -> List[List[Dict]]:
    """Group blocks into rows based on vertical alignment."""
    if not blocks:
        return []
    rows = []
    # Sort blocks by their top y-coordinate
    sorted_blocks = sorted(blocks, key=lambda b: b['bbox'][1])
    current_row = [sorted_blocks[0]]
    for block in sorted_blocks[1:]:
        # If the current block's top is close to the previous one's, it's in the same row
        if abs(block['bbox'][1] - current_row[-1]['bbox'][1]) < y_tolerance:
            current_row.append(block)
        else:
            rows.append(sorted(current_row, key=lambda b: b['bbox'][0]))
            current_row = [block]
    rows.append(sorted(current_row, key=lambda b: b['bbox'][0]))
    return rows


def blocks_to_text(blocks: List[Dict]) -> str:
    """Extract and join text from a list of blocks."""
    text = ""
    for b in blocks:
        for l in b['lines']:
            for s in l['spans']:
                text += s['text'] + " "
    return text.strip()


def text_to_markdown_table(block_text: str) -> str:
    """If text looks like a table, format it as a GFM table."""
    lines = block_text.strip().split("\n")
    if len(lines) < 2:
        return block_text

    # Heuristic: find number of "columns" by counting whitespace-separated tokens
    # in each line. If the counts are consistent, it's likely a table.
    col_counts = [len(re.split(r'\s{2,}', ln.strip())) for ln in lines]
    mode_cols = max(set(col_counts), key=col_counts.count)

    if mode_cols <= 1 or col_counts.count(mode_cols) < len(lines) * 0.7:
        return block_text  # Not a table

    # It's a table - format it!
    md_table = []
    header = lines[0].strip()
    # Split header on multiple spaces to get column titles
    header_cols = [h.strip() for h in re.split(r'\s{2,}', header)]

    # Ensure header has the mode number of columns, pad if not.
    while len(header_cols) < mode_cols:
        header_cols.append("")
    md_table.append("| " + " | ".join(header_cols) + " |")
    md_table.append("|" + "---|" * len(header_cols))

    for line in lines[1:]:
        cols = [c.strip() for c in re.split(r'\s{2,}', line.strip())]
        while len(cols) < mode_cols:
            cols.append("")
        md_table.append("| " + " | ".join(cols) + " |")

    return "\n".join(md_table)


def process_page(page: fitz.Page, body_size: float, min_heading_increase: float, ocr: bool, verbose: bool):
    """Return markdown lines for a page."""
    if verbose:
        print(f"Processing page {page.number + 1} …")

    blocks_dict = page.get_text("dict")
    if not blocks_dict["blocks"] and ocr:
        return ocr_page(page).splitlines()

    two_col = detect_two_column(page)
    blocks = [b for b in blocks_dict["blocks"] if b["type"] == 0]

    # New sorting logic for two-column layouts
    if two_col:
        mid_x = page.rect.width * 0.5
        left_col = sorted([b for b in blocks if b["bbox"][0] < mid_x], key=lambda b: (b["bbox"][1], b["bbox"][0]))
        right_col = sorted([b for b in blocks if b["bbox"][0] >= mid_x], key=lambda b: (b["bbox"][1], b["bbox"][0]))
        blocks = left_col + right_col  # Process left column fully, then right
    else:
        blocks.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))


    markdown_lines: List[str] = []
    link_lookup = {fitz.Rect(lk["from"]): lk["uri"] for lk in page.get_links() if "uri" in lk}

    # New coordinate-based table detection
    potential_tables = find_potential_tables(blocks)
    table_blocks_processed = set()
    for table in potential_tables:
        table_md = table_to_markdown(table)
        markdown_lines.append(table_md)
        for row in table:
            for cell in row:
                table_blocks_processed.add(id(cell))

    for block in blocks:
        if id(block) in table_blocks_processed:
            continue

        block_text = ""
        is_heading_block = False
        heading_level = 0

        # First, determine if the block is a heading and extract all its text
        for line in block["lines"]:
            spans = line["spans"]
            if not spans:
                continue

            first_span = spans[0]
            size_diff = first_span["size"] - body_size
            if size_diff > min_heading_increase * 2:
                is_heading_block = True
                heading_level = 1
            elif size_diff > min_heading_increase:
                is_heading_block = True
                heading_level = 2
            elif size_diff > min_heading_increase * 0.5:
                is_heading_block = True
                heading_level = 3

            block_text += spans_to_markdown(spans, link_lookup, 0) + "\n"

        block_text = block_text.strip()
        if not block_text:
            continue

        # Now process the block
        if is_heading_block:
            # It's a heading, so just format the first line as such
            lines = block_text.split('\n')
            first_line = f"{'#' * heading_level} {lines[0]}"
            markdown_lines.append(first_line)
            markdown_lines.extend(lines[1:])
        else:
            # Not a heading block. Could be a table or just paragraph text.
            table_md = text_to_markdown_table(block_text)
            if table_md != block_text:
                markdown_lines.append(table_md)
            else:
                # It's not a table, so process as regular lines.
                for line in block["lines"]:
                    spans = line["spans"]
                    if not spans:
                        continue
                    md_line = spans_to_markdown(spans, link_lookup, 0)
                    if BULLET_REGEX.match(md_line):
                        md_line = "- " + BULLET_REGEX.sub("", md_line)
                    markdown_lines.append(md_line)

    return markdown_lines


def unwrap_paragraphs(lines: List[str]) -> List[str]:
    out: List[str] = []
    buf: List[str] = []

    def flush():
        if buf:
            out.append(" ".join(s.strip() for s in buf))
            buf.clear()

    for ln in lines:
        stripped = ln.rstrip("\n")

        if not stripped:  # blank line
            flush()
            out.append("")
            continue

        if stripped.startswith("#") or BULLET_REGEX.match(stripped) or NUMBERED_REGEX.match(stripped):
            flush()
            out.append(stripped)
            continue

        # Accumulate into paragraph buffer
        buf.append(stripped)

        # Optional heuristic: flush after sentence-ending punctuation
        if stripped.endswith((".", "?", "!")):
            flush()

    flush()
    return out


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

    # ------------------------------------------------------------------
    # Post-processing: remove recurring headers/footers and noisy blocks
    # ------------------------------------------------------------------

    pages = len(header_candidates)
    common_headers = {h for h, c in header_candidates.items() if c > pages * 0.6 and h}
    common_footers = {f for f, c in footer_candidates.items() if c > pages * 0.6 and f}

    def remove_noise(lines: List[str]) -> List[str]:
        """Strip headers/footers, page markers, duplicate ToC blocks, dup lines."""
        cleaned: List[str] = []
        in_toc_block = False
        for ln in lines:
            stripped = ln.strip()

            # Skip common header/footer lines
            if stripped in common_headers or stripped in common_footers:
                continue

            # Drop page markers like "Page" or "Page"
            if re.match(r'^.*?Page\s*$', stripped) and len(stripped) <= 10:
                continue

            # New, more robust TOC/navigational block removal.
            # Catches "### Handbook Index" and "Contents" pages.
            if stripped.startswith("### Handbook Index") or re.match(r"^#*\s*Contents\s*$", stripped, re.IGNORECASE):
                in_toc_block = True
                continue

            # Catches inline "On this page" sections.
            if stripped.startswith("On this page"):
                continue

            # Exit TOC block when we hit a major heading, which signals new content.
            if in_toc_block and stripped.startswith("#"):
                in_toc_block = False
                # Fall through to process the heading itself

            if in_toc_block:
                continue

            # Index-block removal heuristic from user prompt
            if stripped.count("[Section") >= 2 and len(stripped) > 200:
                continue

            # Deduplicate consecutive identical lines, but allow empty lines to repeat
            if cleaned and stripped and stripped == cleaned[-1].strip():
                continue

            cleaned.append(ln)
        return cleaned

    cleaned_lines = remove_noise(all_lines)

    # ------------------------------------------------------------
    # Soft-wrap paragraphs: merge consecutive text lines to make
    # the Markdown easier to read and produce more coherent
    # embedding input. Lists, headings and blank lines are kept
    # verbatim.
    # ------------------------------------------------------------

    cleaned_lines = unwrap_paragraphs(cleaned_lines)

    md_text = "\n".join(cleaned_lines)
    # Collapse repeated blank lines
    md_text = re.sub(r"\n{3,}", "\n\n", md_text)

    # Legacy safeguard: remove a single large TOC headed "Contents" if present
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
