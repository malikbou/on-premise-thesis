#!/usr/bin/env python3
"""
inspect_pdf_structure.py - A utility to dump the raw block and span
structure of a PDF page.

This helps in debugging PDF extraction scripts by showing the exact coordinates
and text that PyMuPDF (fitz) extracts. This is the "ground truth" that can
be used to develop more robust heuristics for table and column detection.

Usage:
-------
python -m src.inspect_pdf_structure \
    --pdf "data/Computer Science Student Handbook 2024-25.pdf" \
    --page <page_number>
"""
import argparse
from pathlib import Path
import fitz  # PyMuPDF
import json

def inspect_page_structure(pdf_path: str, page_number: int):
    """Prints the JSON structure of a single PDF page."""
    doc = fitz.open(pdf_path)

    if page_number < 1 or page_number > len(doc):
        print(f"Error: Page number must be between 1 and {len(doc)}")
        doc.close()
        return

    page = doc[page_number - 1]  # page_number is 1-based for user input

    page_dict = page.get_text("dict", sort=True)

    # We only care about text blocks for this inspection
    text_blocks = [
        b for b in page_dict.get("blocks", []) if b.get("type") == 0
    ]

    print(f"--- Structure for Page {page_number} of {Path(pdf_path).name} ---")
    print(f"Page Dimensions (width x height): {page.rect.width} x {page.rect.height}")
    print("-" * 20)

    # Use JSON for pretty printing the nested structure
    print(json.dumps(text_blocks, indent=2))

    doc.close()

def main():
    p = argparse.ArgumentParser(
        description="Inspect the raw block/span structure of a PDF page."
    )
    p.add_argument("--pdf", required=True, help="Path to the handbook PDF.")
    p.add_argument("--page", type=int, required=True, help="1-based page number to inspect.")
    args = p.parse_args()

    inspect_page_structure(args.pdf, args.page)

if __name__ == "__main__":
    main()
