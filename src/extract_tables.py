import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import re
import argparse
import pandas as pd

# --- Page Mappings for Target Sections ---
# Based on the PDF's structure, these are the pages where the tables are located.
# We will target these pages specifically for high-resolution OCR.
PAGE_MAPPINGS = {
    "2.3": [19, 20, 21],
    "2.4": [21, 22, 23, 24],
    "4.1.1": [44, 45],
    "4.2.1": [46],
    "4.3.1": [47, 48, 49],
    "7.1.2": [64, 65],
    "8.3.1": [74],
    "8.3.2": [75],
    "8.3.3": [76],
    "12.1": [105],
    "12.2": [106],
    "13.1.2": [111, 112],
    "13.5": [123],
    "15.1.2": [149],
    "17.7.1": [177, 178],
    "23.1": [209, 210, 211, 212],
    "24.1": [222, 223]
}

# --- OCR and Image Processing ---

def ocr_pages(doc: fitz.Document, page_numbers: list) -> str:
    """
    Renders specified pages of a PDF to high-res images and performs OCR.
    """
    full_text = ""
    for page_num in page_numbers:
        # Page numbers in fitz are 0-indexed
        page = doc.load_page(page_num - 1)
        # Render at 300 DPI for high-quality OCR
        pix = page.get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        # Use Tesseract to OCR the image
        text = pytesseract.image_to_string(img)
        full_text += f"\n\n--- OCR Content from Page {page_num} ---\n" + text
    return full_text

# --- Custom Table Parsers ---

def parse_section_2_3(text: str) -> str:
    """
    Custom parser for the complex, multi-table layout of Section 2.3.
    It identifies tables by looking for 'Role' and 'Contact' headers,
    then intelligently splits the lines into two columns.
    """
    lines = text.split('\n')
    tables_data = []
    current_rows = []
    in_table_block = False

    for line in lines:
        cleaned_line = line.strip()
        if not cleaned_line or "--- OCR Content from" in cleaned_line:
            continue

        # A header marks the start of a new table block
        if 'role' in cleaned_line.lower() and 'contact' in cleaned_line.lower():
            if current_rows:
                tables_data.append(current_rows)
            current_rows = [["Role", "Contact"]]
            in_table_block = True
            continue

        # Stop capturing if we hit the next major section title
        if re.match(r'^\d+\.\d+', cleaned_line):
            in_table_block = False
            if current_rows:
                tables_data.append(current_rows)
                current_rows = []
            continue

        if in_table_block:
            # Heuristic to split a line into a role and a contact.
            # It assumes the contact is the last set of capitalized words.
            parts = re.split(r'\s{2,}', cleaned_line)
            if len(parts) >= 2:
                role = " ".join(parts[:-1])
                contact = parts[-1]
                current_rows.append([role, contact])
            # Handle cases where OCR might put role and contact on separate lines
            elif len(current_rows) > 1 and len(current_rows[-1]) == 1:
                 current_rows[-1].append(cleaned_line)
            else:
                 current_rows.append([cleaned_line])


    if current_rows:
        tables_data.append(current_rows)

    # Convert all captured table data to Markdown
    final_markdown = ""
    for table_rows in tables_data:
        # Ensure all rows have 2 columns, padding if necessary
        processed_rows = []
        for row in table_rows:
            if len(row) == 1:
                processed_rows.append([row[0], ""])
            else:
                processed_rows.append(row)

        if not processed_rows: continue

        header = processed_rows[0]
        data = processed_rows[1:]
        df = pd.DataFrame(data, columns=header)
        final_markdown += df.to_markdown(index=False) + "\n\n"

    return final_markdown.strip()

# --- Normalization and Dispatcher ---

def normalize_ocr_text(text: str) -> list[str]:
    """
    Cleans raw OCR text by removing junk lines and merging split lines.
    """
    # This is a simplified version for now. The custom parsers will handle more logic.
    return [line.strip() for line in text.split('\n') if line.strip()]

def dispatch_parser(section: str, text: str) -> str:
    """
    Calls the appropriate custom parser based on the section number.
    Falls back to raw text if no specific parser is available.
    """
    parser_map = {
        "2.3": parse_section_2_3,
        # Future custom parsers for other sections will be added here.
    }

    parser = parser_map.get(section)

    if parser:
        print(f"    [INFO] Using custom parser for Section {section}.")
        return parser(text) # Pass the raw text to the custom parser
    else:
        # This is the fallback for sections we haven't written a parser for yet.
        return ""


# --- Main Logic ---

def extract_and_convert_tables(pdf_path: str, out_path: str):
    """
    Main function that orchestrates the OCR and parsing process.
    """
    doc = fitz.open(pdf_path)
    final_markdown = ""

    print("Starting targeted OCR table extraction with custom parsers...")

    for section, pages in PAGE_MAPPINGS.items():
        print(f"  - Processing Section {section} on page(s) {pages}...")

        # Step 1: Perform OCR on the target pages
        ocr_text = ocr_pages(doc, pages)

        # Step 2: Dispatch to the correct custom parser
        markdown_table = dispatch_parser(section, ocr_text)

        # Step 3: Append the result to the final output file
        if markdown_table:
            print(f"    [SUCCESS] Parsed a structured table for Section {section}.")
            final_markdown += f"## Table from Section {section}\n"
            final_markdown += markdown_table
            final_markdown += "\n\n"
        else:
            print(f"    [FALLBACK] No custom parser for Section {section}. Appending raw text.")
            final_markdown += f"## Raw OCR from Section {section}\n"
            final_markdown += f"```\n{ocr_text}\n```\n\n"

    # Step 4: Write the final combined Markdown to the output file
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(final_markdown)

    print(f"\nExtraction complete. All tables saved to '{out_path}'.")

def main():
    p = argparse.ArgumentParser(description="Extract tables from specific PDF sections using targeted OCR.")
    p.add_argument("--pdf", required=True, help="Path to the handbook PDF.")
    p.add_argument("--out", required=True, help="Output markdown path for extracted tables.")
    args = p.parse_args()
    extract_and_convert_tables(args.pdf, args.out)

if __name__ == "__main__":
    main()
