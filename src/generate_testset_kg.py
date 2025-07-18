#!/usr/bin/env python3
"""Generate a synthetic test-set for the UCL Student-Handbook RAG demo.

This script builds a Knowledge Graph from the source documents to generate a
more sophisticated test set with a higher proportion of multi-hop questions.

Example usage
-------------
python -m src.generate_testset_kg \
    --docs data/sts-student-handbook.pdf \
    --dept STS \
    --size 10

Environment
-----------
Requires an OPENAI_API_KEY in your environment. The script defaults to
`gpt-3.5-turbo-16k` for generation and `gpt-4o` for critique, but you can
override both.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, Node
from ragas.testset.synthesizers import QueryDistribution
from ragas.testset.synthesizers.single_hop import SingleHopQuerySynthesizer
from ragas.testset.synthesizers.multi_hop import MultiHopQuerySynthesizer
from ragas.testset.persona import Persona


# Import graph construction components
from ragas.testset.transforms import (
    apply_transforms,
    Parallel,
    KeyphrasesExtractor,
    SummaryExtractor,
    EmbeddingExtractor,
)
from ragas.testset.transforms.relationship_builders.cosine import SummaryCosineSimilarityBuilder


# Increased chunk size to avoid "documents too short" error
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_documents(path: str) -> List:
    """Load documents from a path (file or directory)."""
    if os.path.isdir(path):
        # Load all PDFs / MD / txt recursively
        loader = DirectoryLoader(path, glob="**/*.*")
    elif path.lower().endswith(".pdf"):
        loader = PyPDFLoader(path)
    else:
        loader = TextLoader(path)
    return loader.load()


def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(docs)


# ---------------------------------------------------------------------------
# Custom query distribution
# ---------------------------------------------------------------------------

def custom_query_distribution() -> QueryDistribution:
    """Create a custom query distribution with 60% single-hop and 40% multi-hop questions."""
    return QueryDistribution(
        distributions={
            SingleHopQuerySynthesizer: 0.6,
            MultiHopQuerySynthesizer: 0.4,
        }
    )


# ---------------------------------------------------------------------------
# Custom student-tone prompts disabled for now because the StringPrompt API
# in Ragas ≥0.2 has changed (no 'instruction' argument). The default prompts
# shipped with Ragas will be used instead. Uncomment and adapt using the new
# PydanticPrompt API if customised wording is required.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate a synthetic Ragas test-set from a handbook PDF.")
    parser.add_argument("--docs", required=True, help="Path to PDF or directory with documents.")
    parser.add_argument("--dept", default="STS", help="Dept slug used in output filename.")
    parser.add_argument("--size", type=int, default=10, help="Number of questions to generate.")
    parser.add_argument("--generator-model", default="gpt-3.5-turbo-16k")
    parser.add_argument("--critic-model", default="gpt-4o")
    args = parser.parse_args()

    print(f"Loading documents from {args.docs} …")
    raw_docs = load_documents(args.docs)
    print(f"Loaded {len(raw_docs)} source documents")

    # Optional chunking step (Ragas generator can take raw docs but chunking is safer)
    docs = chunk_documents(raw_docs)
    print(f"Chunked to {len(docs)} passages (≈{CHUNK_SIZE} chars each)")

    # Wrap OpenAI models for Ragas
    generator_llm = LangchainLLMWrapper(ChatOpenAI(model=args.generator_model, temperature=0))
    critic_llm = LangchainLLMWrapper(ChatOpenAI(model=args.critic_model, temperature=0))
    embedding_model = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

    # --- Knowledge Graph Construction ---
    print("Building Knowledge Graph from documents…")

    # 1. Convert documents to Ragas Nodes
    nodes = [Node(properties={"page_content": doc.page_content}) for doc in docs]
    kg = KnowledgeGraph(nodes=nodes)

    # 2. Define graph transformation pipeline
    transforms = [
        Parallel(
            KeyphrasesExtractor(llm=generator_llm),
            SummaryExtractor(llm=generator_llm),
        ),
        EmbeddingExtractor(
            embedding_model=embedding_model,
            embed_property_name="summary" # Embed summaries, not full content
        ),
        SummaryCosineSimilarityBuilder(
            embedding_model=embedding_model,
            threshold=0.85 # Stricter threshold for higher quality links
        )
    ]

    # 3. Apply transformations to build the graph
    apply_transforms(kg, transforms)

    print(f"Knowledge Graph built with {len(kg.nodes)} nodes and {len(kg.relationships)} relationships.")


    # --- Testset Generation ---

    # Create UCL student personas
    persona_new_student = Persona(
        name="New Student",
        role_description="A first-year student looking for information about university policies and procedures."
    )
    persona_international = Persona(
        name="International Student",
        role_description="An international student seeking information about visa requirements, accommodation, and support services."
    )
    persona_graduate = Persona(
        name="Graduate Student",
        role_description="A graduate student interested in research opportunities, funding, and academic progression."
    )

    personas = [persona_new_student, persona_international, persona_graduate]

    generator = TestsetGenerator(
        llm=generator_llm,
        critic_llm=critic_llm,
        embedding_model=embedding_model,
        persona_list=personas,
        knowledge_graph=kg  # Pass the graph to the generator
    )

    # Create custom query distribution
    query_distribution = custom_query_distribution()

    print("Generating test-set from Knowledge Graph… this may take a few minutes.")
    dataset = generator.generate(
        testset_size=args.size,
        query_distribution=query_distribution,
    )

    out_dir = "testset"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.dept}_testset.json")

    with open(out_path, "w") as f:
        json.dump(dataset.to_dict(), f, indent=2)
    print(f"Saved {len(dataset)} samples to {out_path}")


if __name__ == "__main__":
    main()
