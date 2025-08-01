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
# It guides the LLM to generate high-quality, realistic student questions.
STUDENT_QUERY_INSTRUCTION = """
You are a master of prompt engineering, tasked with generating synthetic data.
Your goal is to generate a question and a corresponding answer based on the provided context, from the perspective of the given student persona.

Your task is to FULLY EMBODY the persona. The question you generate should reflect their specific situation, their emotional state (e.g., anxious, ambitious, confused), and the kind of language they would use. It should sound like a real question a student would ask, not a formal FAQ.

The answer you generate MUST:
- Be found verbatim in the provided text.
- Be concise and directly answer the user's question.
- **If the context contains a relevant URL, you must include the full URL in the answer.**

Do not invent any information. The question should be natural-sounding, but the answer must be grounded truth from the text.

**Critique and Refine Step:**
1. First, generate a draft question based on the persona and context.
2. Then, critique your own question. Does it sound like a real student? Is it too generic? Is it just rephrasing a heading?
3. Refine the question to make it more specific, persona-driven, and natural.

**Examples of BAD questions to AVOID:**
- "What does the section on Personal study time say?"
- "How are students assessed?"
- "What is the information about word counts?"

Your final output should only be the refined question and the answer.
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
        # Preserve header metadata and add the source
        doc.metadata.update(source_metadata)

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
            role_description="I'm a first-year, totally overwhelmed and feeling lost. I think I've missed a deadline for choosing my modules, and I can't find my timetable. I'm scared I'll get in trouble for my attendance. My questions are urgent and slightly panicked. For example: 'I think I messed up my module choices, who do I talk to RIGHT NOW?' or 'What's the bare minimum attendance I need so I don't get kicked out?'",
        ),
        Persona(
            name="Anxious International Student",
            role_description="I'm an international student and very worried about my visa and settling in. I need to know the rules for working part-time and what happens if my attendance drops. I also have practical questions about setting up a bank account or seeing a doctor. My questions are driven by anxiety about compliance. For example: 'I'm terrified of doing something wrong with my visa, what are the rules on attendance?' or 'How do I even start to see a doctor here?'",
        ),
        Persona(
            name="Ambitious Masters Student",
            role_description="I'm a postgraduate focused on my dissertation and future career. I need specifics on submission deadlines, extension request procedures, academic misconduct rules, and research ethics. My questions are precise and goal-oriented. For example: 'What is the formal procedure for getting an extension on my dissertation?' or 'I'm planning research with human participants, what specific ethics approval process must I follow?'",
        ),
        Persona(
            name="Financially Concerned Student",
            role_description="I'm struggling with money and worried about making ends meet. I need to know about financial support, bursaries, or loans. I'm also looking for tips on budgeting. My questions are practical and driven by financial stress. For example: 'My loan hasn't come in and I can't pay my rent, is there any emergency financial help?' or 'Who can I talk to about my money problems without being judged?'"
        ),
        Persona(
            name="Student Needing Support",
            role_description="I have a long-term health condition that affects my studies. I need to ask about reasonable adjustments like extra time in exams or flexible deadlines. I'm also interested in mental health support. My questions are about accessibility and wellbeing. For example: 'How do I arrange for extra time in my exams because of my disability?' or 'I'm really struggling with my mental health, what kind of counseling does UCL offer?'"
        ),
    ]

    # Configure the synthesizer based on the chosen query style
    instruction = STUDENT_QUERY_INSTRUCTION if args.query_style == "perfect" else None
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
