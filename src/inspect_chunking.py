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
from typing import List
import fitz  # PyMuPDF
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

PDF_PATH = 'data/sts-student-handbook.pdf'

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts clean, structured text from a PDF file using PyMuPDF."""
    full_text = ""
    print(f"--- Step 1: Extracting Text from '{pdf_path}' ---")
    try:
        with fitz.open(pdf_path) as doc:
            page_count = len(doc)  # Get the page count before the loop
            for i, page in enumerate(doc):
                full_text += page.get_text() + f"\\n\\n--- End of Page {i+1} ---\\n\\n"
        # Now we can safely print the page count after the 'with' block has closed the doc
        print(f"Successfully extracted text from {page_count} pages.")
        return full_text
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""

def create_and_filter_chunks(full_text: str, source_path: str) -> List[Document]:
    """Chunks a document based on numbered headings and filters for quality."""
    print("\\n--- Step 2: Chunking and Filtering ---")

    # --- Pre-processing: Rejoin standalone numbers with their subsequent text ---
    # This is crucial for fixing the PDF parsing issue where numbers for headings
    # appear on separate lines from the heading text itself.
    print("Pre-processing text to rejoin headings...")
    # Pattern: a line that starts and ends with a number (e.g., "1 \n")
    # followed by a line of text. The (?m) flag is for multiline matching.
    rejoined_text = re.sub(r'(?m)^(\d+(\.\d+)*)\s*\n(.*?)$', r'\1 \3', full_text)

    # Attempt to split by numbered headings on the processed text
    heading_pattern = re.compile(r"^\d+(\.\d+)*\s", re.MULTILINE)
    heading_indices = [match.start() for match in heading_pattern.finditer(rejoined_text)]

    docs = []
    source_metadata = {"source": source_path}

    if heading_indices:
        print(f"Found {len(heading_indices)} semantic headings. Creating chunks...")
        for i in range(len(heading_indices)):
            start_index = heading_indices[i]
            end_index = heading_indices[i+1] if i+1 < len(heading_indices) else len(rejoined_text)
            chunk_text = rejoined_text[start_index:end_index].strip()
            docs.append(Document(page_content=chunk_text, metadata=source_metadata))
    else:
        print("WARN: No numbered headings found. Falling back to robust recursive chunking.")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        split_texts = splitter.split_text(rejoined_text)
        docs = [Document(page_content=t, metadata=source_metadata) for t in split_texts]

    # Filter out low-quality chunks
    original_count = len(docs)
    filtered_docs = []
    for doc in docs:
        if len(doc.page_content) < 100:
            continue
        if not re.search(r'[a-zA-Z]', doc.page_content):
            continue
        filtered_docs.append(doc)

    print(f"Filtered out {original_count - len(filtered_docs)} low-quality chunks.")
    return filtered_docs


def save_chunks_to_markdown(chunks: List[Document], output_path: str):
    """Saves the final, clean chunks to a single markdown file."""
    print(f"\\n--- Step 4: Saving clean chunks to '{output_path}' ---")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                # A simple approach: add each chunk's content followed by a separator
                f.write(chunk.page_content + "\\n\\n---\\n\\n")
        print(f"Successfully saved markdown file to '{output_path}'")
    except Exception as e:
        print(f"Error saving markdown file: {e}")


def main():
    """Main execution function."""
    # Step 1: Extract clean text from the PDF
    full_text_content = extract_text_from_pdf(PDF_PATH)
    if not full_text_content:
        return

    print(f"\\nFirst 500 characters of extracted text:\\n'{full_text_content[:500]}...'")

    # Step 2: Create and filter high-quality chunks
    final_chunks = create_and_filter_chunks(full_text_content, PDF_PATH)

    # Step 3: Inspect the results
    print(f"\\n--- Step 3: Inspecting Results ---")
    print(f"Total high-quality chunks created: {len(final_chunks)}\\n")

    # Print the first 3 chunks to see the quality
    for i, doc in enumerate(final_chunks[:3]):
        print(f"--- Chunk {i+1} (Length: {len(doc.page_content)}) ---")
        print(doc.page_content)
        print("\\n" + "="*80)

    # Step 4: Save the clean chunks to a markdown file
    save_chunks_to_markdown(final_chunks, 'data/sts-student-handbook-clean.md')


if __name__ == "__main__":
    main()
