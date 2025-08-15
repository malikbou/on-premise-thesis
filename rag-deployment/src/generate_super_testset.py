#!/usr/bin/env python3
"""
Super Testset Generator for UCL Computer Science Handbook

This script creates a high-quality, diverse testset from the CS handbook using:
- RAGAS 0.3.1 TestsetGenerator with improved document processing
- Diverse student personas (anxious first-year, confident final-year, international, etc.)
- Smart document chunking that preserves context
- Quality control and validation
- Comprehensive coverage of handbook topics

Usage:
    python src/generate_super_testset.py --size 100 --output testset/super_testset_100q.json
"""

import argparse
import json
import os
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.testset import TestsetGenerator
from ragas.testset.persona import Persona
from ragas.run_config import RunConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Student personas for diverse question generation
STUDENT_PERSONAS = [
    Persona(
        name="anxious_first_year",
        description="A first-year undergraduate student who is anxious about university life, often worried about deadlines, grades, and fitting in. Uses informal language and asks questions that show uncertainty.",
        role_description="You are a first-year computer science student who is new to university and feeling overwhelmed. You're worried about missing deadlines, not understanding the system, and making mistakes."
    ),
    Persona(
        name="confident_final_year",
        description="A final-year student who is confident and experienced with university systems. Asks more sophisticated questions about career preparation, advanced topics, and strategic decisions.",
        role_description="You are a final-year computer science student who knows the university system well. You ask strategic questions about career preparation, advanced opportunities, and optimizing your academic path."
    ),
    Persona(
        name="international_student",
        description="An international student concerned about visa requirements, cultural differences, and specific regulations that affect non-UK students. Often asks about immigration compliance and support services.",
        role_description="You are an international computer science student on a student visa. You're particularly concerned about attendance requirements, visa compliance, and understanding UK university systems."
    ),
    Persona(
        name="part_time_student",
        description="A part-time student juggling work and studies, focused on practical questions about scheduling, flexibility, and managing competing priorities.",
        role_description="You are a part-time computer science student who works while studying. You need to understand scheduling flexibility, deadline management, and how to balance work with academic requirements."
    ),
    Persona(
        name="postgraduate_student",
        description="A postgraduate student focused on research, dissertation requirements, and advanced academic opportunities. Asks detailed questions about academic procedures and research support.",
        role_description="You are a postgraduate computer science student working on your dissertation. You ask detailed questions about research requirements, academic procedures, and advanced opportunities."
    ),
    Persona(
        name="struggling_student",
        description="A student who is having academic difficulties and needs support. Asks questions about help resources, second chances, and recovery from poor performance.",
        role_description="You are a computer science student who is struggling academically and needs help. You ask questions about support services, academic recovery, and getting back on track."
    )
]

# Note: RAGAS 0.3.1 uses a simplified generation approach
# The generator will automatically create diverse question types

def load_and_chunk_document(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Load and intelligently chunk the CS handbook document
    """
    logger.info(f"Loading document from {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Document not found: {file_path}")

    # Load the document
    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()

    logger.info(f"Loaded document with {len(documents[0].page_content)} characters")

    # Smart chunking that preserves context
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n# ",      # Main sections
            "\n## ",     # Subsections
            "\n### ",    # Sub-subsections
            "\n\n",      # Paragraphs
            "\n",        # Lines
            " ",         # Words
            ""           # Characters
        ],
        keep_separator=True,
    )

    # Split documents
    chunks = text_splitter.split_documents(documents)

    # Filter out very short chunks and chunks with mostly skipped content
    filtered_chunks = []
    for chunk in chunks:
        content = chunk.page_content.strip()

        # Skip chunks that are too short or mostly skipped content
        if (len(content) < 100 or
            content.count("[SKIPPING") > 2 or
            len(content.split()) < 20):
            continue

        # Clean up the content
        content = re.sub(r'\[SKIPPING[^\]]*\]', '', content)
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)  # Remove excessive newlines
        content = content.strip()

        if len(content) > 50:  # Still has meaningful content after cleaning
            chunk.page_content = content
            filtered_chunks.append(chunk)

    logger.info(f"Created {len(filtered_chunks)} meaningful chunks after filtering")
    return filtered_chunks

def create_knowledge_graph_from_chunks(chunks: List[Document]) -> None:
    """
    In RAGAS 0.3.1, knowledge graph creation is handled internally
    This function is kept for compatibility but doesn't need implementation
    """
    logger.info("Knowledge graph will be created automatically by RAGAS 0.3.1")
    pass

def validate_testset_quality(testset_data: List[Dict]) -> List[Dict]:
    """
    Validate and filter testset for quality
    """
    logger.info("Validating testset quality...")

    valid_samples = []
    quality_issues = {
        'too_short': 0,
        'no_context': 0,
        'generic_question': 0,
        'reference_mismatch': 0,
        'valid': 0
    }

    for sample in testset_data:
        user_input = sample.get('user_input', '').strip()
        reference = sample.get('reference', '').strip()
        contexts = sample.get('reference_contexts', [])

        # Check for quality issues
        if len(user_input) < 10:
            quality_issues['too_short'] += 1
            continue

        if not contexts or len(contexts) == 0:
            quality_issues['no_context'] += 1
            continue

        # Check for generic questions (common patterns to avoid)
        generic_patterns = [
            r'^what (is|does|are) (the|this) (section|information)',
            r'^tell me about (the|this) (section|part)',
            r'^what does section \d+ say',
            r'^explain (the|this) (section|part)',
        ]

        is_generic = any(re.search(pattern, user_input.lower()) for pattern in generic_patterns)
        if is_generic:
            quality_issues['generic_question'] += 1
            continue

        # Check if reference makes sense with context
        if len(reference) < 10:
            quality_issues['reference_mismatch'] += 1
            continue

        # Sample passes quality checks
        quality_issues['valid'] += 1
        valid_samples.append(sample)

    logger.info(f"Quality validation results: {quality_issues}")
    logger.info(f"Kept {len(valid_samples)} high-quality samples out of {len(testset_data)}")

    return valid_samples

def enhance_sample_with_metadata(sample: Dict, index: int) -> Dict:
    """
    Enhance a sample with additional metadata
    """
    # Add sample metadata
    sample['sample_id'] = f"cs_handbook_{index:03d}"
    sample['generated_at'] = datetime.now().isoformat()

    return sample

def generate_super_testset(
    document_path: str,
    testset_size: int,
    output_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    model_name: str = "gpt-4o-mini"
) -> None:
    """
    Generate a high-quality, diverse testset using RAGAS 0.3.1
    """
    logger.info(f"Starting Super Testset Generation for {testset_size} samples")

    # Initialize models
    llm = ChatOpenAI(model=model_name, temperature=0.7)  # Slightly higher temp for creativity
    embeddings = OpenAIEmbeddings()

    # Load and chunk document
    chunks = load_and_chunk_document(document_path, chunk_size, chunk_overlap)

    if len(chunks) == 0:
        raise ValueError("No valid chunks created from document")

    # Create testset generator with personas
    generator = TestsetGenerator.from_langchain(
        llm=llm,
        embedding_model=embeddings
    )

    # Set personas for diverse question generation
    generator.persona_list = STUDENT_PERSONAS

    # Generate testset using the new RAGAS 0.3.1 API
    logger.info("Generating testset with RAGAS 0.3.1...")
    testset = generator.generate_with_langchain_docs(
        documents=chunks,
        testset_size=testset_size,
        run_config=RunConfig(timeout=300, max_retries=3),
        with_debugging_logs=False,
        raise_exceptions=False
    )

    # Convert to our format
    testset_data = []
    for i, sample in enumerate(testset.samples):
        # Extract data in our expected format
        sample_data = {
            "user_input": sample.user_input,
            "reference_contexts": sample.reference_contexts,
            "reference": sample.reference,
        }

        # Add metadata
        sample_data = enhance_sample_with_metadata(sample_data, i)

        testset_data.append(sample_data)

    # Validate quality
    validated_testset = validate_testset_quality(testset_data)

    if len(validated_testset) < testset_size * 0.7:  # Less than 70% valid
        logger.warning(f"Low quality rate: {len(validated_testset)}/{testset_size} samples valid")
        logger.warning("Consider adjusting chunk size or generation parameters")

    # Save testset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(validated_testset, f, indent=2, ensure_ascii=False)

    logger.info(f"Super testset saved to {output_path}")
    logger.info(f"Generated {len(validated_testset)} high-quality samples")

    # Generate summary report
    generate_testset_report(validated_testset, output_path.replace('.json', '_report.txt'))

def generate_testset_report(testset_data: List[Dict], report_path: str) -> None:
    """
    Generate a quality report for the testset
    """
    logger.info("Generating testset quality report...")

    # Analyze testset characteristics
    question_lengths = []
    context_counts = []
    reference_lengths = []

    for sample in testset_data:
        # Track question lengths
        question_lengths.append(len(sample.get('user_input', '')))

        # Track context counts
        contexts = sample.get('reference_contexts', [])
        context_counts.append(len(contexts) if contexts else 0)

        # Track reference lengths
        reference_lengths.append(len(sample.get('reference', '')))

    # Generate report
    report = f"""
Super Testset Quality Report (RAGAS 0.3.1)
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Samples: {len(testset_data)}

=== Question Length Statistics ===
Average Length: {sum(question_lengths)/len(question_lengths):.1f} characters
Min Length: {min(question_lengths)} characters
Max Length: {max(question_lengths)} characters

=== Context Statistics ===
Average Contexts per Question: {sum(context_counts)/len(context_counts):.1f}
Min Contexts: {min(context_counts) if context_counts else 0}
Max Contexts: {max(context_counts) if context_counts else 0}

=== Reference Answer Statistics ===
Average Reference Length: {sum(reference_lengths)/len(reference_lengths):.1f} characters
Min Reference Length: {min(reference_lengths) if reference_lengths else 0}
Max Reference Length: {max(reference_lengths) if reference_lengths else 0}

=== Sample Questions ===
"""

    # Add sample questions
    sample_questions = testset_data[:5]  # First 5 samples
    for i, sample in enumerate(sample_questions, 1):
        report += f"\n{i}. Question: {sample.get('user_input', '')[:150]}...\n"
        report += f"   Reference: {sample.get('reference', '')[:100]}...\n"
        report += f"   Contexts: {len(sample.get('reference_contexts', []))}\n"

    # Save report
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    logger.info(f"Quality report saved to {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate Super Testset for UCL CS Handbook")
    parser.add_argument("--document", default="data/cs-handbook.md",
                       help="Path to the CS handbook markdown file")
    parser.add_argument("--size", type=int, default=100,
                       help="Number of test samples to generate")
    parser.add_argument("--output", default="testset/super_testset_100q.json",
                       help="Output path for the testset")
    parser.add_argument("--chunk-size", type=int, default=1000,
                       help="Chunk size for document splitting")
    parser.add_argument("--chunk-overlap", type=int, default=200,
                       help="Chunk overlap for document splitting")
    parser.add_argument("--model", default="gpt-4o-mini",
                       help="OpenAI model to use for generation")

    args = parser.parse_args()

    try:
        generate_super_testset(
            document_path=args.document,
            testset_size=args.size,
            output_path=args.output,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            model_name=args.model
        )
        print(f"✅ Super testset generation completed successfully!")
        print(f"📄 Testset saved to: {args.output}")
        print(f"📊 Report saved to: {args.output.replace('.json', '_report.txt')}")

    except Exception as e:
        logger.error(f"Testset generation failed: {e}")
        raise

if __name__ == "__main__":
    main()
