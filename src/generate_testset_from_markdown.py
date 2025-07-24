#!/usr/bin/env python3
"""
Generate a synthetic, single-hop test set from a structured Markdown file.

This script leverages the structured nature of Markdown (headings, sections)
to create high-quality, semantically coherent chunks, which in turn produce
a superior test set for RAG evaluation compared to PDF-based generation.

Key Features:
- **Markdown-Optimized**: Uses MarkdownHeaderTextSplitter for intelligent chunking.
- **High Quality**: Generates clear, grammatically correct questions by default.
- **Flexible**: Allows for different query styles (e.g., misspelled) to be
  generated for robustness testing via the `--query-style` argument.

Example usage
-------------
# Generate a high-quality, clean test set from a Markdown file
python -m src.generate_testset_from_markdown \\
    --doc data/sts-student-handbook-clean.md \\
    --size 10

# Generate a test set with deliberately misspelled questions
python -m src.generate_testset_from_markdown \\
    --doc data/sts-student-handbook-clean.md \\
    --size 10 \\
    --query-style misspelled
"""
from __future__ import annotations

import argparse
import json
import os
import re
from typing import List, Optional
from datetime import datetime

from langchain.docstore.document import Document
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper, BaseRagasLLM
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
from ragas.run_config import RunConfig
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.synthesizers.single_hop.specific import (
    SingleHopSpecificQuerySynthesizer,
)
from ragas.testset.persona import Persona


# This instruction is used when `--query-style` is set to 'perfect'.
# It forces the LLM to generate high-quality, grammatically correct questions.
PERFECT_GRAMMAR_INSTRUCTION = """
Generate a question and a corresponding answer based on the provided context.

Your task is to act as a university administrator creating an official FAQ document.
The questions you generate must be:
- **Grammatically flawless:** Adhere to perfect English grammar.
- **Clearly worded:** The question should be unambiguous and easy to understand.
- **Formal in tone:** Avoid slang, contractions, or informal language.
- **Directly answerable from the context:** The answer must be found verbatim
  in the provided text.

Do not invent any information. Do not include any misspellings or typos.
"""


def load_markdown_document(path: str) -> List[Document]:
    """Loads a single Markdown document using TextLoader."""
    if not path.lower().endswith(".md"):
        raise ValueError("This script is designed for Markdown files (.md) only.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"The specified document was not found: {path}")

    loader = TextLoader(path)
    return loader.load()


def chunk_markdown_documents(docs: List[Document]) -> List[Document]:
    """
    Chunks documents based on Markdown headings. This creates semantically
    meaningful chunks based on the document's structure.
    """
    print("Attempting to chunk documents by Markdown headings...")
    # The handbook uses '#' and '##' for its main sections.
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
    ]
    # Use the first document's content, as we expect a single file.
    markdown_text = docs[0].page_content
    source_metadata = {"source": docs[0].metadata.get("source", "Unknown")}

    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    split_docs = splitter.split_text(markdown_text)

    # The splitter adds header info to metadata, but we need to combine it
    # with the page content for the LLM and add the source.
    for doc in split_docs:
        header_content = " ".join(doc.metadata.values())
        doc.page_content = f"{header_content}\\n\\n{doc.page_content}".strip()
        doc.metadata = source_metadata

    print(f"Successfully created {len(split_docs)} semantic chunks from Markdown.")
    return split_docs


def filter_chunks(docs: List[Document], min_length: int = 50) -> List[Document]:
    """Filters out documents that are too short or lack meaningful content."""
    original_count = len(docs)
    filtered_docs = []
    for doc in docs:
        if len(doc.page_content) < min_length:
            continue  # Skip docs that are too short
        if not re.search(r'[a-zA-Z]', doc.page_content):
            continue  # Skip docs that contain no alphabetic characters
        filtered_docs.append(doc)

    print(f"Filtered out {original_count - len(filtered_docs)} low-quality chunks.")
    return filtered_docs


def configure_synthesizer(
    llm: BaseRagasLLM, instruction: Optional[str] = None
) -> SingleHopSpecificQuerySynthesizer:
    """Configures the single-hop synthesizer, optionally with a custom prompt."""
    # Explicitly tell the synthesizer to use our new list-based property
    synthesizer = SingleHopSpecificQuerySynthesizer(
        llm=llm, property_name="content_as_theme"
    )
    if instruction:
        try:
            # Use the correct prompt key found via debugging
            key = "query_answer_generation_prompt"
            prompt = synthesizer.get_prompts()[key]
            prompt.instruction = instruction
            # Unpack the dictionary into keyword arguments
            synthesizer.set_prompts(**{key: prompt})
        except KeyError:
            # This warning should no longer appear
            print(f"WARN: Could not set custom prompt for {type(synthesizer).__name__}")
    return synthesizer


def main():
    parser = argparse.ArgumentParser(
        description="Generate a synthetic Ragas test-set from a Markdown document."
    )
    parser.add_argument(
        "--doc", required=True, help="Path to the source Markdown (.md) document."
    )
    parser.add_argument(
        "--dept", default="STS", help="Dept slug used in output filename."
    )
    parser.add_argument(
        "--size", type=int, default=10, help="Number of questions to generate."
    )
    parser.add_argument("--generator-model", default="gpt-3.5-turbo-16k")
    parser.add_argument(
        "--query-style",
        choices=["perfect", "misspelled", "poor_grammar", "web_search"],
        default="perfect",
        help="The style of questions to generate. Defaults to 'perfect'."
    )
    args = parser.parse_args()

    # Load and chunk documents
    print(f"Loading document from {args.doc}…")
    raw_docs = load_markdown_document(args.doc)
    print(f"Loaded {len(raw_docs)} source document(s)")
    docs = chunk_markdown_documents(raw_docs)
    docs = filter_chunks(docs)
    print(f"Chunked and filtered to {len(docs)} passages")

    # Initialize models
    generator_llm = LangchainLLMWrapper(ChatOpenAI(model=args.generator_model, temperature=0))
    embedding_model = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

    # Create UCL student personas with detailed, problem-oriented descriptions
    personas = [
        Persona(
            name="Stressed Fresher",
            role_description="A first-year undergraduate who is overwhelmed by the start of term. You are urgently trying to figure out key deadlines for course registration, find your timetable, understand the attendance policy, and locate where to get administrative help. You ask direct, practical questions to solve immediate problems.",
        ),
        Persona(
            name="Anxious International Student",
            role_description="An international student who has just arrived in the UK. You are worried about your Student Visa requirements, especially rules about working and attendance. You also need to find out about setting up a bank account and registering with a doctor (GP). Your questions are focused on compliance and settling in.",
        ),
        Persona(
            name="Ambitious Masters Student",
            role_description="A postgraduate student planning their year. You need to know the exact dates for dissertation submission, the process for getting an extension, the rules for academic misconduct, and where to find information about PhD funding. You ask precise questions to plan your academic future.",
        ),
    ]

    # Configure the synthesizer based on the chosen query style
    instruction = PERFECT_GRAMMAR_INSTRUCTION if args.query_style == "perfect" else None
    synthesizer = configure_synthesizer(generator_llm, instruction)
    query_distribution = [(synthesizer, 1.0)]

    # Manually create a simple KnowledgeGraph to avoid default transformations
    nodes = []
    for doc in docs:
        nodes.append(Node(
            type=NodeType.CHUNK,
            properties={
                "page_content": doc.page_content,
                # Create a new property that is a list, as required by the synthesizer
                "content_as_theme": [doc.page_content],
                "document_metadata": doc.metadata
            }
        ))
    knowledge_graph = KnowledgeGraph(nodes=nodes)

    # Initialize the generator with the simple graph
    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=embedding_model,
        persona_list=personas,
        knowledge_graph=knowledge_graph,
    )

    # Generate the testset using the core `generate` method
    print("Generating test-set… this may take a few minutes.")
    run_config = RunConfig(max_workers=2)
    dataset = generator.generate(
        testset_size=args.size,
        query_distribution=query_distribution,
        run_config=run_config,
    )

    # Save the results
    out_dir = "testset"
    os.makedirs(out_dir, exist_ok=True)

    # Create a detailed, timestamped filename for better tracking
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = args.generator_model.replace(":", "_").replace("/", "_")
    style_slug = f"_{args.query_style}" if args.query_style != "perfect" else ""
    out_path = os.path.join(
        out_dir,
        f"{args.dept}_testset_from_markdown{style_slug}_{model_slug}_{timestamp}.json"
    )

    with open(out_path, "w") as f:
        json.dump(dataset.to_list(), f, indent=2)
    print(f"Saved {len(dataset.to_list())} samples to {out_path}")


if __name__ == "__main__":
    main()
