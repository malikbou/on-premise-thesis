#!/usr/bin/env python3
"""
process_handbook_by_section.py - A new, structured approach to PDF conversion.

This script uses a user-defined list of sections containing complex tables to
skip them during processing, focusing on extracting high-quality text from the
rest of the document.
"""
import fitz
import argparse
import re
import collections
from typing import Dict, List, Tuple

# --- User-defined list of sections to skip ---
# Based on manual review, these sections contain complex tables.
SECTIONS_WITH_TABLES = [
    "2.3", "2.4", "4.1.1", "4.2.1", "4.3.1", "7.1.2", "8.3.1", "8.3.2", "8.3.3",
    "12.1", "12.2", "13.1.2", "13.5", "15.1.2", "17.7.1", "23.1", "24.1", "27" # Skips all of Annex (Section 27)
]

# --- Self-Contained Helper Functions ---

BULLET_REGEX = re.compile(r"^\s*[\u2022\u2023\u25E6\u2043\u2219\-]\s+")
NUMBERED_REGEX = re.compile(r"^\s*\d+([.)])\s+")

def analyse_font_sizes(doc: fitz.Document) -> Tuple[Dict[float, int], float]:
    """Return (histogram, body_size). Body size = most frequent span size."""
    hist = collections.defaultdict(int)
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0: continue
            for line in block["lines"]:
                for span in line["spans"]:
                    hist[round(span["size"], 1)] += 1
    body_size = max(hist, key=hist.get) if hist else 10.0
    return hist, body_size

def spans_to_markdown(line_spans, link_lookup, heading_level):
    """Convert list of span dicts to a single markdown line."""
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

    line_text = re.sub(r'[\uf0b7\uf076\uf0e0\uf0a7\uf075\uf071\uf0b2\uf07d\uf07e]', '', line_text)

    if heading_level:
        line_text = f"{'#' * heading_level} {line_text.strip()}"
    return line_text.strip()

def remove_noise(lines: List[str], common_headers: set, common_footers: set) -> List[str]:
    """Strip headers/footers, page markers, and other noise."""
    cleaned: List[str] = []
    for ln in lines:
        stripped = ln.strip()
        if stripped in common_headers or stripped in common_footers:
            continue
        if re.match(r'^page\s*\d+\s*of\s*\d+$', stripped, re.IGNORECASE):
            continue
        # Skip lines that are just a page number
        if re.match(r'^\d+$', stripped):
            page_num_match = True
            # Basic check to avoid removing legitimate numbered lines
            if len(stripped) > 3: page_num_match = False
            if cleaned and cleaned[-1].strip().endswith(":"): page_num_match = False
            if page_num_match: continue

        cleaned.append(ln)
    return cleaned

def unwrap_paragraphs(lines: List[str]) -> List[str]:
    """Join consecutive non-list, non-heading lines into single paragraphs."""
    out: List[str] = []
    buf: List[str] = []
    def flush():
        if buf:
            out.append(" ".join(s.strip() for s in buf))
            buf.clear()
    for ln in lines:
        stripped = ln.rstrip("\n")
        if not stripped:
            flush()
            out.append("")
            continue
        if stripped.startswith("#") or BULLET_REGEX.match(stripped) or NUMBERED_REGEX.match(stripped):
            flush()
            out.append(stripped)
            continue
        buf.append(stripped)
    flush()
    return out

# --- Main Processing Logic ---

def process_document(pdf_path: str, out_path: str):
    doc = fitz.open(pdf_path)

    # Perform a global analysis for headers, footers, and body font size first.
    print("Performing global analysis of document structure...")
    _, body_size = analyse_font_sizes(doc)
    header_candidates = collections.Counter()
    footer_candidates = collections.Counter()
    for page in doc:
        lines = [line.strip() for line in page.get_text().split('\n') if line.strip()]
        if len(lines) > 1:
            header_candidates[lines[0]] += 1
            footer_candidates[lines[-1]] += 1

    common_headers = {h for h, c in header_candidates.items() if c > len(doc) * 0.5 and len(h) < 100}
    common_footers = {f for f, c in footer_candidates.items() if c > len(doc) * 0.5 and len(f) < 100}
    print(f"Found {len(common_headers)} common headers and {len(common_footers)} common footers.")

    # Now, process the document page by page
    all_lines = []
    in_table_section = False
    min_heading_increase = 1.5

    for page in doc:
        page_lines = []
        blocks = [b for b in page.get_text("dict")["blocks"] if b.get("type") == 0]
        blocks.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
        link_lookup = {fitz.Rect(lk["from"]): lk["uri"] for lk in page.get_links() if "uri" in lk}

        for block in blocks:
            # Check if this block starts a new section
            first_line_of_block = ""
            if block["lines"] and block["lines"][0]["spans"]:
                first_line_of_block = block["lines"][0]["spans"][0]["text"].strip()

            section_match = re.match(r'^(#+\s*)?([\d\.]+)', first_line_of_block)
            if not section_match and block["lines"]:
                 # Check the full text of the first line for a section number
                 full_first_line = "".join([s['text'] for s in block["lines"][0]["spans"]])
                 section_match = re.match(r'^(#+\s*)?([\d\.]+)', full_first_line.strip())

            if section_match:
                section_num = section_match.group(2)
                # Check if this section or a parent section is in the skip list
                if any(section_num.startswith(s) for s in SECTIONS_WITH_TABLES):
                    in_table_section = True
                    page_lines.append(f"\n[SKIPPING TABLE SECTION: {section_num}]\n")
                else:
                    in_table_section = False

            if in_table_section:
                continue

            # If we're not in a table section, process the block normally
            for line in block["lines"]:
                spans = line.get("spans", [])
                if not spans: continue

                size_diff = spans[0]["size"] - body_size
                heading_level = 0
                if size_diff > min_heading_increase * 2: heading_level = 1
                elif size_diff > min_heading_increase: heading_level = 2
                elif size_diff > min_heading_increase * 0.5: heading_level = 3

                md_line = spans_to_markdown(spans, link_lookup, heading_level)
                if not heading_level:
                    if BULLET_REGEX.match(md_line):
                        md_line = "- " + BULLET_REGEX.sub("", md_line)
                page_lines.append(md_line)

        all_lines.extend(page_lines)
        all_lines.append("") # Page break

    # Final cleaning and formatting
    cleaned_lines = remove_noise(all_lines, common_headers, common_footers)
    unwrapped_lines = unwrap_paragraphs(cleaned_lines)

    full_markdown = "\n".join(unwrapped_lines)
    full_markdown = re.sub(r"\n{3,}", "\n\n", full_markdown)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_markdown.strip())

    print(f"\nProcessing complete. Output saved to {out_path}")


def main():
    p = argparse.ArgumentParser(description="Process CS Handbook section by section, skipping tables.")
    p.add_argument("--pdf", required=True, help="Path to the handbook PDF.")
    p.add_argument("--out", required=True, help="Output markdown path.")
    args = p.parse_args()
    process_document(args.pdf, args.out)

if __name__ == "__main__":
    main()
