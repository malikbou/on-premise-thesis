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
    # This function is now only responsible for headers, footers, and simple noise.
    # TOC removal is handled in the main processing loop.
    for ln in lines:
        stripped = ln.strip()

        # Skip common header/footer lines from global analysis
        if stripped in common_headers or stripped in common_footers:
            continue

        # Skip page number lines like "Page 7 of 95" or just "7"
        if re.match(r'^page\s*\d+\s*of\s*\d+$', stripped, re.IGNORECASE):
            continue
        if re.match(r'^\d+$', stripped) and len(stripped) < 4:
            continue

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


def cleanup_markdown_output(lines: List[str]) -> str:
    """
    Final cleanup of the markdown output.
    """
    # 1. Remove most common headers/footers
    common_headers = {"Quick Links", "Key contacts", "Student representation"}
    common_footers = {"Additional Resources", "Further information:"}
    clean_lines = remove_noise(lines, common_headers, common_footers)

    # 2. Unwrap paragraphs
    unwrapped_lines = unwrap_paragraphs(clean_lines)

    # 3. **NEW: Remove initial TOC section**
    # Find where real content begins and strip everything before it
    content_start_idx = 0
    for i, line in enumerate(unwrapped_lines):
        # Look for the first substantial paragraph that indicates real content
        # More specific detection - look for actual paragraph content, not just descriptions
        if (len(line) > 150 and
            any(phrase in line.lower() for phrase in [
                'dear students', 'warm welcome to', 'computer science was established',
                'provides you with key information', 'is intended to provide you'
            ]) and
            not line.startswith('#') and  # Don't match headings
            not line.startswith('[')      # Don't match link-heavy content
        ):
            # Found real content - look backwards for the section heading
            for j in range(i-1, max(0, i-15), -1):
                if unwrapped_lines[j].startswith('#'):
                    content_start_idx = j
                    break
            break

    # Remove everything before the real content starts
    if content_start_idx > 0:
        unwrapped_lines = unwrapped_lines[content_start_idx:]

    # 4. Join all lines into a single string
    full_text = '\n'.join(unwrapped_lines)

    # 5. Ensure proper spacing around headings
    full_text = re.sub(r'\n(?!^\n)(#+)', r'\n\n\1', full_text)
    full_text = full_text.lstrip() # Remove any leading newlines from the top of the file.

    # 6. Collapse three or more consecutive newlines into just two.
    full_text = re.sub(r'\n{3,}', r'\n\n', full_text)

    return full_text


def process_document(pdf_path: str, out_path: str):
    """
    Processes the entire PDF document, extracts content page by page,
    and converts it to a structured Markdown file.
    """
    doc = fitz.open(pdf_path)
    all_lines = []
    in_table_section = False
    min_heading_increase = 1.5
    found_first_real_content = False  # Track when we've moved past initial TOC

    for page in doc:
        page_lines = []
        blocks = [b for b in page.get_text("dict")["blocks"] if b.get("type") == 0]
        blocks.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
        link_lookup = {fitz.Rect(lk["from"]): lk["uri"] for lk in page.get_links() if "uri" in lk}

        for block in blocks:
            # Get raw text from block for analysis
            block_text = "".join(span["text"] for line in block.get("lines", []) for span in line.get("spans", [])).strip()

            if not block_text:
                continue

            # --- START: ENHANCED FILTERING LOGIC ---

            # FILTER 1: Remove page footers containing the page artifact character.
            if "Page" in block_text:
                continue

            # FILTER 2: Enhanced TOC and navigation filtering

            # Basic TOC markers
            TOC_MARKERS = ["Handbook Index", "Contents"]
            if any(marker.lower() in block_text.lower() for marker in TOC_MARKERS):
                continue

            # Enhanced "On this page" filtering - catch the content that follows
            if "On this page" in block_text:
                # Count section references in the block (e.g., "1.1", "2.3", etc.)
                section_refs = re.findall(r'\b\d+\.\d+\b', block_text)
                # If it contains multiple section references, it's likely a TOC
                if len(section_refs) >= 2:
                    continue
                # Also skip if it literally starts with "On this page"
                if re.match(r'^\s*On this page\b', block_text, re.IGNORECASE):
                    continue

            # FILTER 3: Initial document TOC detection
            # Skip blocks that look like the jumbled TOC at the document start
            if not found_first_real_content:
                # Count how many section references this block contains
                section_refs = re.findall(r'\b\d+\.\d+\b', block_text)
                link_heavy = len(section_refs) > 1

                # Skip if it's link-heavy navigation content
                if link_heavy:
                    continue

                # More aggressive detection of real content
                # Look for blocks that start with actual sentences (not just section titles/links)
                word_count = len(block_text.split())
                section_density = len(section_refs) / max(word_count, 1)

                # Check if this looks like real paragraph content
                # Real content should have longer sentences and proper punctuation
                sentences = re.split(r'[.!?]+', block_text)
                avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)

                # Skip blocks that are mostly section titles, links, or short phrases
                is_section_title_block = (
                    re.match(r'^#+\s*[\d\.\s]*[A-Z]', block_text) or  # Starts with heading markup
                    re.match(r'^[\d\.]+\s+[A-Z]', block_text) or      # Starts with section number
                    block_text.startswith('[') or                     # Starts with link
                    word_count < 15 or                                # Very short
                    section_density > 0.15                            # High section reference density
                )

                if is_section_title_block:
                    continue

                # If we find a substantial paragraph (>30 words, avg sentence >8 words), it's real content
                if word_count > 30 and avg_sentence_length > 8:
                    found_first_real_content = True
                # Also detect welcome messages or substantial content blocks
                elif any(phrase in block_text.lower() for phrase in ['dear students', 'welcome to', 'provides information']):
                    found_first_real_content = True

            # FILTER 4: Remove repetitive section link blocks
            # A block is removed if its text starts with "Section X" AND it is a hyperlink.
            if re.match(r'^\s*Section\s+\d+', block_text, re.IGNORECASE):
                is_link_block = False
                block_bbox = fitz.Rect(block['bbox'])
                for link in page.get_links():
                    if block_bbox.intersects(fitz.Rect(link['from'])):
                        is_link_block = True
                        break
                if is_link_block:
                    continue # This is a navigational link block. DELETE IT.

            # FILTER 5: Skip blocks that are primarily section number lists
            # (catches remaining TOC-style content)
            if found_first_real_content:  # Only apply this after we've found real content
                section_refs = re.findall(r'\b\d+\.\d+\b', block_text)
                words = block_text.split()
                if len(section_refs) >= 3 and len(words) < 50:  # Lots of section refs, not much other text
                    continue

            # --- END: ENHANCED FILTERING LOGIC ---

            # --- Table Handling ---
            # Check if this block starts a new section to be skipped (for tables)
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

                # Determine heading level based on section numbering (e.g., "1.1", "1.1.1")
                raw_text = "".join(s["text"] for s in spans).strip()
                heading_level = 0
                # Matches patterns like "1.2", "1.2.3 First word"
                match = re.match(r'^(\d[\d\.]*)\s+', raw_text)

                if match:
                    section_number = match.group(1).strip().rstrip('.')
                    heading_level = len(section_number.split('.'))
                # Fallback for major headings like "Section 1"
                elif re.match(r'^Section\s+\d+', raw_text, re.IGNORECASE):
                    heading_level = 1

                md_line = spans_to_markdown(spans, link_lookup, heading_level)
                page_lines.append(md_line)

        all_lines.extend(page_lines)

    # --- NEW: Final Cleanup and Write ---
    # After processing all pages, run the final cleanup function.
    final_output = cleanup_markdown_output(all_lines)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(final_output)


def main():
    p = argparse.ArgumentParser(description="Process CS Handbook section by section, skipping tables.")
    p.add_argument("--pdf", required=True, help="Path to the handbook PDF.")
    p.add_argument("--out", required=True, help="Output markdown path.")
    args = p.parse_args()
    process_document(args.pdf, args.out)

if __name__ == "__main__":
    main()
