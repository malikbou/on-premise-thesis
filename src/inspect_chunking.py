#!/usr/bin/env python3
"""
A script to inspect the PDF parsing and document chunking process.

This script uses the PyMuPDF library for robust text extraction and then
applies a semantic chunking strategy to create high-quality document chunks
for a RAG pipeline.

It prints the output at each stage, allowing you to verify the quality of
the text extraction and the final chunks.

Usage:
------
# First, ensure you have the correct libraries installed:
pip install PyMuPDF langchain

# Then, run the script:
python -m src.inspect_chunking
"""
import re
from typing import List, Dict
import fitz  # PyMuPDF
from langchain.docstore.document import Document

PDF_PATH = 'data/sts-student-handbook.pdf'

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts clean, structured text from a PDF file using PyMuPDF."""
    full_text = ""
    print(f"--- Step 1: Extracting Text from '{pdf_path}' ---")
    try:
        with fitz.open(pdf_path) as doc:
            page_count = len(doc)
            for i, page in enumerate(doc):
                full_text += page.get_text() + f"\n\n--- End of Page {i+1} ---\n\n"
        print(f"Successfully extracted text from {page_count} pages.")
        return full_text
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""

def format_tables_as_markdown(text: str) -> str:
    """
    Identifies and reformats data that appears to be in a two-column table format
    into a structured Markdown table. This is especially useful for staff lists
    and key dates.
    """
    # This regex is designed to find patterns that look like two-column tables.
    # It looks for lines with a clear split, often with roles on the left and names/emails on the right.
    # Example: "Head of Department\nDr Jean-Baptiste Gouyon\nj.gouyon@ucl.ac.uk"

    # A simplified heuristic: find lines with what looks like a "key" and "value",
    # often separated by newlines and containing typical signifiers like "@" for email.
    # This is a complex problem, so we'll start with a targeted approach for staff.

    # Regex to capture a block of text that is likely a staff member entry.
    # It looks for a title, a name, and an email address, which are often separated by newlines.
    pattern = re.compile(
        r"^(?P<role>[\w\s&]+?)\n(?P<name>[\w\s-]+\n[\w\s&]+?)\n(?P<email>[\w.-]+@[\w.-]+)$",
        re.MULTILINE
    )

    # A more general pattern for two-column layouts
    # This looks for two distinct blocks of text on consecutive lines.
    general_pattern = re.compile(r"^(?P<key>.+?)\n(?P<value>.+?)$", re.MULTILINE)

    def replace_with_markdown_table(match):
        role = ' '.join(match.group('role').split())
        name = ' '.join(match.group('name').replace('\n', ' ').split())
        email = match.group('email').strip()
        return f"| {role} | {name} | {email} |"

    # For now, let's focus on a simpler replacement for general key-value pairs
    # to avoid overly complex regex. A full table formatter might require a more
    # sophisticated parsing approach (e.g., analyzing text coordinates from PyMuPDF).
    # Let's clean up multiple newlines into a more structured format.

    # A key observation is that tables are often mangled into multi-line text blocks.
    # Let's try to identify these and format them.
    # For instance, "Role\nPerson\nEmail" followed by data.
    if "Role\nPerson\nEmail" in text:
        text = text.replace("Role\nPerson\nEmail", "| Role | Person | Email |\n|---|---|---|")
        # Now, try to format the subsequent lines. This is non-trivial.
        # Let's stick to a simpler cleanup for now.

    # A more robust approach would involve coordinate-based table detection,
    # but for this text-based post-processing, we can make some improvements.

    # Let's try to fix the staff list specifically, as it has a predictable structure.
    # "Role \n Person \n Email"
    text = re.sub(r'^(.*?)\n(.*?)\n([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', r'| \1 | \2 | \3 |', text, flags=re.MULTILINE)

    return text


def parse_toc(full_text: str) -> Dict[str, str]:
    """
    Parses the table of contents from the first page to extract main section headings.
    This provides the high-level semantic structure of the document.
    """
    print("\n--- Step 2: Parsing Table of Contents ---")
    toc = {}
    # The TOC is on the first pages, ending before the Provost's Welcome.
    # We limit the search to the first 4000 characters for robustness.
    toc_text = full_text[:4000]

    # Regex to find lines like: "2 Introduction to the department [...] ..."
    # It captures the number and the title, ignoring the dots and page number.
    pattern = re.compile(r"^\s*(\d+)\s+(.*?)\s*\.{5,}", re.MULTILINE)

    for match in pattern.finditer(toc_text):
        section_number = match.group(1)
        # Clean up title from any extra whitespace
        title = ' '.join(match.group(2).split())
        toc[section_number] = title
        print(f"Found TOC entry: {section_number} -> {title}")

    return toc

def create_semantic_chunks(full_text: str, toc: Dict[str, str], source_path: str) -> List[Document]:
    """
    Creates semantically meaningful chunks based on the document's hierarchical structure.
    It uses the parsed Table of Contents to provide high-level context to each chunk.
    """
    print("\n--- Step 3: Creating Semantic Chunks ---")

    # --- Pre-processing ---
    # 1. Rejoin headings that were split onto separate lines during PDF extraction.
    # e.g., "2.1\nIntroduction" becomes "2.1 Introduction"
    rejoined_text = re.sub(r'(?m)^(\d+(\.\d+)*)\s*\n(.*?)$', r'\1 \3', full_text)
    # 2. Remove the noisy Table of Contents dotted lines from the body.
    cleaned_text = re.sub(r'\s\.{5,}\s\d+', '', rejoined_text)

    # --- Chunking based on Hierarchy ---
    # We find all headings (e.g., "1.1", "2.1.3", "14.2") to define chunk boundaries.
    heading_pattern = re.compile(r"^(?P<heading>(\d+)\.\d+.*?)$", re.MULTILINE | re.DOTALL)

    heading_matches = list(heading_pattern.finditer(cleaned_text))

    docs = []
    source_metadata = {"source": source_path}

    for i, match in enumerate(heading_matches):
        # The content of a chunk is the text between the current heading and the next one.
        start_index = match.start()
        end_index = heading_matches[i+1].start() if i + 1 < len(heading_matches) else len(cleaned_text)

        chunk_content = cleaned_text[start_index:end_index].strip()

        # Extract the heading line and the main section number (e.g., "2" from "2.1.3")
        heading_line = match.group('heading').split('\n')[0].strip()
        main_section_number = match.group(2)

        # Look up the main section title from our parsed TOC.
        main_title = toc.get(main_section_number, f"Section {main_section_number}")

        # Clean the body content and format the final output in Markdown.
        # This adds the crucial high-level context to every chunk.
        body_content = chunk_content[len(heading_line):].strip()
        body_content = re.sub(r'\n{3,}', '\n\n', body_content) # Consolidate newlines

        # --- Format Tables ---
        # Apply our table formatting logic to the body of the chunk.
        body_content = format_tables_as_markdown(body_content)

        final_chunk = (
            f"# {main_section_number} {main_title}\n\n"
            f"## {heading_line}\n\n"
            f"{body_content}"
        )

        docs.append(Document(page_content=final_chunk.strip(), metadata=source_metadata))

    # --- Filtering ---
    # Filter out chunks that are too short or lack meaningful content.
    original_count = len(docs)
    filtered_docs = []
    for doc in docs:
        if len(doc.page_content) < 150:  # Increased threshold for more substance
            continue
        if not re.search(r'[a-zA-Z]', doc.page_content):
            continue
        filtered_docs.append(doc)

    print(f"\nFiltered out {original_count - len(filtered_docs)} low-quality chunks.")
    return filtered_docs

def save_chunks_to_markdown(chunks: List[Document], output_path: str):
    """Saves the final, clean chunks to a single markdown file."""
    print(f"\n--- Step 5: Saving clean chunks to '{output_path}' ---")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                # Add each chunk's content followed by a clear separator
                f.write(chunk.page_content + "\n\n---\n\n")
        print(f"Successfully saved markdown file to '{output_path}'")
    except Exception as e:
        print(f"Error saving markdown file: {e}")

def main():
    """Main execution function."""
    # Step 1: Extract clean text from the PDF
    full_text_content = extract_text_from_pdf(PDF_PATH)
    if not full_text_content:
        return

    # Step 2: Parse the Table of Contents to get main section titles
    toc = parse_toc(full_text_content)

    # Step 3: Create and filter high-quality, semantically-aware chunks
    final_chunks = create_semantic_chunks(full_text_content, toc, PDF_PATH)

    # Step 4: Inspect the results
    print(f"\n--- Step 4: Inspecting Results ---")
    print(f"Total high-quality chunks created: {len(final_chunks)}\n")

    # Print the first 2 chunks to demonstrate the new structure
    if final_chunks:
        for i, doc in enumerate(final_chunks[:2]):
            print(f"--- Chunk {i+1} (Length: {len(doc.page_content)}) ---")
            print(doc.page_content)
            print("\n" + "="*80)

    # Step 5: Save the clean chunks to a markdown file
    save_chunks_to_markdown(final_chunks, 'data/sts-student-handbook-clean.md')

if __name__ == "__main__":
    main()
