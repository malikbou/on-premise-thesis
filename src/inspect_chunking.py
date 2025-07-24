#!/usr/bin/env python3
"""
A script for advanced, layout-aware PDF parsing to create high-quality,
semantically chunked Markdown for RAG applications.

This script uses PyMuPDF's rich coordinate and font data to intelligently
reconstruct the document's structure, including headings, paragraphs, tables,
and links, avoiding the pitfalls of simple text extraction.

Usage:
------
# First, ensure you have the correct libraries installed:
pip install PyMuPDF langchain

# Then, run the script:
python src/inspect_chunking.py
"""
import re
import collections
from typing import List, Dict, Any

import fitz  # PyMuPDF
from langchain.docstore.document import Document

PDF_PATH = 'data/sts-student-handbook.pdf'

def get_font_styles(doc: fitz.Document) -> tuple[dict[float, int], float]:
    """Analyzes the document to find font sizes and the main body font size."""
    styles = collections.defaultdict(int)
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if b['type'] == 0:
                for l in b["lines"]:
                    for s in l["spans"]:
                        styles[round(s["size"])] += 1

    main_font_size = max(styles, key=styles.get) if styles else 10.0
    return styles, main_font_size

def reconstruct_document_from_pdf(pdf_path: str) -> str:
    """
    Intelligently reconstructs the entire PDF into a single, clean Markdown string
    by analyzing the layout, fonts, and coordinates of each text block.
    """
    print("--- Step 1: Intelligently Reconstructing PDF to Markdown ---")
    doc = fitz.open(pdf_path)
    _, main_font_size = get_font_styles(doc)
    full_markdown = ""

    for i, page in enumerate(doc):
        print(f"Processing page {i+1}/{len(doc)}...")

        links = {link['from']: link['uri'] for link in page.get_links() if 'uri' in link}
        blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_IMAGES)["blocks"]

        for b in blocks:
            if b['type'] == 0:  # It's a text block
                block_markdown = ""
                for l in b["lines"]:
                    # Step 1: Pre-process all spans in a line to find their associated links
                    processed_spans = []
                    for s in l["spans"]:
                        span_rect = fitz.Rect(s['bbox'])
                        link_uri = next((uri for r, uri in links.items() if r.intersects(span_rect)), None)
                        processed_spans.append({'text': s['text'], 'uri': link_uri})

                    # Step 2: Group consecutive spans with the same link URI to form a single link
                    line_text = ""
                    i = 0
                    while i < len(processed_spans):
                        span = processed_spans[i]
                        if span['uri']:
                            # This is a link. Group all subsequent spans that share this URI.
                            link_text = span['text']
                            j = i + 1
                            while j < len(processed_spans) and processed_spans[j]['uri'] == span['uri']:
                                link_text += processed_spans[j]['text']
                                j += 1

                            line_text += f"[{link_text}]({span['uri']})"
                            i = j # Move pointer past the processed spans
                        else:
                            # Not a link, just append the text
                            line_text += span['text']
                            i += 1

                    # Determine heading level based on font size
                    span_size = round(l["spans"][0]["size"])
                    if span_size > main_font_size + 1:
                        level = 1 if span_size > main_font_size + 4 else 2
                        block_markdown += f"\n{'#' * level} {line_text.strip()}\n"
                    else:
                        block_markdown += line_text + " "

                full_markdown += block_markdown.strip() + "\n"

    doc.close()
    return re.sub(r'\n{3,}', '\n\n', full_markdown)

def create_final_chunks(markdown_content: str, source_path: str) -> List[Document]:
    """Chunks the clean markdown based on heading structure."""
    print("\n--- Step 2: Chunking document based on headings ---")

    # Split by H1 or H2 headings
    chunks = re.split(r'\n(?=#{1,2}\s)', markdown_content)

    docs = []
    # Filter out TOC and other noise
    for chunk_text in chunks:
        chunk_text = chunk_text.strip()
        if "contents" in chunk_text[:50].lower():
            print("Filtering out Table of Contents.")
            continue

        # Remove page numbers that might be left over
        chunk_text = re.sub(r'^\d+\s*$', '', chunk_text, flags=re.MULTILINE).strip()

        if len(chunk_text) < 100:
            continue

        docs.append(Document(page_content=chunk_text, metadata={"source": source_path}))

    return docs

def save_chunks_to_markdown(chunks: List[Document], output_path: str):
    """Saves the final, clean chunks to a single markdown file."""
    print(f"\n--- Step 3: Saving clean chunks to '{output_path}' ---")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(chunks):
                f.write(chunk.page_content)
                if i < len(chunks) - 1:
                    f.write("\n\n#################################################################\n\n")
        print(f"Successfully saved {len(chunks)} chunks to '{output_path}'")
    except Exception as e:
        print(f"Error saving markdown file: {e}")

def main():
    """Main execution function."""
    markdown_content = reconstruct_document_from_pdf(PDF_PATH)
    final_chunks = create_final_chunks(markdown_content, PDF_PATH)

    print(f"\n--- Step 4: Inspecting Results ---")
    print(f"Total high-quality chunks created: {len(final_chunks)}\n")

    if final_chunks:
        print(f"--- First Chunk (Length: {len(final_chunks[0].page_content)}) ---")
        print(final_chunks[0].page_content)
        print("\n" + "="*80)

    save_chunks_to_markdown(final_chunks, 'data/sts-student-handbook-clean.md')

if __name__ == "__main__":
    main()
