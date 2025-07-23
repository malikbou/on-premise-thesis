#!/usr/bin/env python3
"""
Generate a synthetic, single-hop test set for the UCL Student-Handbook.

This script is a streamlined and cost-effective version of the Knowledge
Graph-based generator. It focuses exclusively on creating high-quality,
single-hop questions without the expensive graph-building steps.

Key Features:
- **Cost-Efficient**: Skips the expensive Knowledge Graph transformations.
- **High Quality**: Uses semantic chunking and a custom prompt to generate
  clear, grammatically correct questions by default.
- **Flexible**: Allows for different query styles (e.g., misspelled) to be
  generated for robustness testing via the `--query-style` argument.

Example usage
-------------
# Generate a high-quality, clean test set
python -m src.generate_testset_single_hop \\
    --docs data/sts-student-handbook.pdf \\
    --size 10

# Generate a test set with deliberately misspelled questions
python -m src.generate_testset_single_hop \\
    --docs data/sts-student-handbook.pdf \\
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
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
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


def load_documents(path: str) -> List[Document]:
    """Load documents from a path (file or directory)."""
    if os.path.isdir(path):
        loader = DirectoryLoader(path, glob="**/*.*")
    elif path.lower().endswith(".pdf"):
        loader = PyPDFLoader(path)
    else:
        loader = TextLoader(path)
    return loader.load()


def chunk_documents(docs: List[Document]) -> List[Document]:
    """
    Chunks documents based on semantic headings rather than fixed size.
    It assumes headings are numbered (e.g., "1. Introduction") and uses
    these as boundaries for creating meaningful chunks.
    """
    print("Attempting to chunk documents by semantic headings...")
    full_text = "\\n\\n".join(doc.page_content for doc in docs)
    heading_pattern = re.compile(r"^\\d+(\\.\\d+)*\\s.*$", re.MULTILINE)
    heading_indices = [match.start() for match in heading_pattern.finditer(full_text)]

    if not heading_indices:
        print("WARN: No numbered headings found. Falling back to fixed-size chunking.")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        return splitter.split_documents(docs)

    semantic_chunks = []
    for i in range(len(heading_indices)):
        start_index = heading_indices[i]
        end_index = heading_indices[i+1] if i + 1 < len(heading_indices) else len(full_text)
        chunk_text = full_text[start_index:end_index].strip()

        if chunk_text:
            semantic_chunks.append(Document(
                page_content=chunk_text,
                metadata={"source": docs[0].metadata.get("source", "Unknown")}
            ))

    print(f"Successfully created {len(semantic_chunks)} semantic chunks.")
    return semantic_chunks


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
        description="Generate a synthetic single-hop Ragas test-set."
    )
    parser.add_argument(
        "--docs", required=True, help="Path to PDF or directory with documents."
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
    print(f"Loading documents from {args.docs}…")
    raw_docs = load_documents(args.docs)
    print(f"Loaded {len(raw_docs)} source documents")
    docs = chunk_documents(raw_docs)
    print(f"Chunked to {len(docs)} passages")

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
        f"{args.dept}_testset_single_hop{style_slug}_{model_slug}_{timestamp}.json"
    )

    with open(out_path, "w") as f:
        json.dump(dataset.to_list(), f, indent=2)
    print(f"Saved {len(dataset.to_list())} samples to {out_path}")


if __name__ == "__main__":
    main()
